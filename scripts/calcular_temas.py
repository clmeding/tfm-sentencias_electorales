"""Calcula los temas (materias) de cada documento del corpus histórico y
guarda el resultado en data/temas_por_documento.csv, además de imprimir un
resumen de conteos para armar el panel de KPIs en Streamlit.

Uso (desde la raíz del repo):
    python scripts/calcular_temas.py

Requiere:
    - data/dataset_sentencias.csv   (ya existente en el repo)
    - documents/output/*.json        (texto ya extraído, ya existente)

No entrena nada ni usa embeddings -- es solo regex sobre texto ya
extraído, corre en segundos incluso con ~900 documentos.

NÚCLEO DINÁMICO (30-ago-2026): las 10 "barras propias" del panel ya NO
son una lista fija en el código. Cada vez que corres este script, se
recalculan sobre el corpus real actual (calcular_nucleo_dinamico), y el
resultado se guarda en data/nucleo_temas_actual.json. Así, si con el
tiempo cambia qué temas son más frecuentes en el corpus histórico, el
panel se actualiza solo la próxima vez que vuelvas a correr este script
-- no hace falta editar detectar_temas.py a mano. La demo de Streamlit
lee ese mismo JSON para graficar exactamente el mismo núcleo que se usó
aquí.

RESPALDO GEMINI PARA DOCUMENTOS NUEVOS (09-sep-2026): el regex nunca va a
cubrir temas que todavía no existen en su catálogo -- por diseño, solo
sabe buscar lo que ya conoce. Sobre el corpus histórico (866 docs) eso
deja 233 (27%) en "Otro / no identificado", y esa parte histórica se deja
tal cual (ya se decidió que no vale la pena reabrirla). Pero de aquí en
adelante, cada vez que llega un documento NUEVO (uno cuyo json_filename no
aparece todavía en data/temas_por_documento.csv de la corrida anterior):
  1. Se intenta primero con el regex, igual que siempre (gratis, rápido).
  2. Si el regex no encuentra nada ("Otro"), se llama UNA vez a Gemini
     (mismo modelo económico que ya se usa para resúmenes,
     gemini-flash-lite-latest) solo para ESE documento, para ver si
     propone un tema -- de catálogo o nuevo -- que el regex no contempla.
  3. El resultado se guarda en data/temas_gemini_documentos_nuevos.json,
     así que un documento nuevo solo consume UNA llamada a Gemini en toda
     su vida (corridas futuras lo leen del caché, no vuelven a llamar).
Esto se queda dentro del uso gratuito porque son documentos sueltos según
van llegando, no una corrida masiva de golpe (eso fue lo que costó dinero
en la corrida de descubrimiento de scripts/detectar_temas_gemini.py). Si
no hay GEMINI_API_KEY configurada, el script sigue funcionando solo con
regex (no truena), y esos documentos se quedan como "Otro" hasta que se
vuelva a correr con la key disponible. La columna nueva `fuente_tema` en
el CSV de salida indica si cada fila vino de regex o de este respaldo.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from detectar_temas import (  # noqa: E402
    OTRO,
    OTROS_ESPECIFICOS,
    calcular_nucleo_dinamico,
    detectar_temas_con_otro,
    detectar_temas_extendido_con_otro,
    detectar_temas_panel_con_bucket,
)
from detectar_temas_gemini import API_KEY, CONFIANZA_ACEPTABLE, llamar_gemini  # noqa: E402

DATASET_CSV = REPO_ROOT / "data" / "dataset_sentencias.csv"
DOCUMENTS_DIR = REPO_ROOT / "documents" / "output"
SALIDA_CSV = REPO_ROOT / "data" / "temas_por_documento.csv"
NUCLEO_JSON = REPO_ROOT / "data" / "nucleo_temas_actual.json"
RESUMENES_JSON = REPO_ROOT / "data" / "resumenes.json"
CACHE_GEMINI_NUEVOS = REPO_ROOT / "data" / "temas_gemini_documentos_nuevos.json"
TOP_N_NUCLEO = 10


def cargar_resumenes() -> dict[str, str]:
    """Carga data/resumenes.json (si existe) y devuelve, por json_filename,
    el texto concatenado hechos+agravios+resolucion -- para usarlo como
    señal de alta confianza en la detección de temas (ver docstring de
    detectar_temas._tema_detectado: en el resumen ya curado, una sola
    mención basta, sin necesitar UMBRAL_MENCIONES). Documentos con 'error'
    o sin resumen generado simplemente no aportan esta señal extra --
    el detector sigue funcionando igual sobre el texto completo."""
    if not RESUMENES_JSON.exists():
        return {}
    try:
        datos = json.loads(RESUMENES_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}
    resultado = {}
    for r in datos:
        if "error" in r or not r.get("json_filename"):
            continue
        texto = " ".join(
            [r.get("hechos") or "", r.get("agravios") or "", r.get("resolucion") or ""]
        ).strip()
        if texto:
            resultado[r["json_filename"]] = texto
    return resultado


def formatear_anio(valor) -> str:
    """Evita que el año salga como '2023.0' (pandas lo lee como float por
    los NaN en otras filas)."""
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return ""
    try:
        return str(int(float(valor)))
    except (TypeError, ValueError):
        return str(valor)


def etiqueta_expediente(row) -> str:
    """Igual que _etiqueta_expediente() en src/explicar_prediccion.py: si el
    expediente viene vacío en los metadatos, usa el nombre de archivo (sin
    extensión) como respaldo -- así la clave coincide con la que muestra la
    búsqueda real de precedentes, y el tema no se "pierde" para esos casos.
    """
    expediente = row.get("expediente")
    if isinstance(expediente, str) and expediente.strip():
        return expediente
    return Path(row["json_filename"]).stem


def cargar_texto(json_filename: str) -> str:
    ruta = DOCUMENTS_DIR / json_filename
    if not ruta.exists():
        return ""
    try:
        data = json.loads(ruta.read_text(encoding="utf-8"))
    except Exception:
        return ""
    return data.get("extracted_text_clean", "") or ""


def cargar_historicos() -> set[str]:
    """json_filename ya presentes en la corrida anterior de este script --
    esos documentos se dejan tal cual (regex, sin respaldo de Gemini) por
    decisión explícita: lo histórico no se reabre."""
    if not SALIDA_CSV.exists():
        return set()
    try:
        return set(pd.read_csv(SALIDA_CSV)["json_filename"])
    except Exception:
        return set()


def cargar_cache_gemini_nuevos() -> dict[str, dict]:
    if not CACHE_GEMINI_NUEVOS.exists():
        return {}
    try:
        datos = json.loads(CACHE_GEMINI_NUEVOS.read_text(encoding="utf-8"))
        return {r["json_filename"]: r for r in datos if r.get("json_filename")}
    except Exception:
        return {}


def guardar_cache_gemini_nuevos(cache: dict[str, dict]) -> None:
    CACHE_GEMINI_NUEVOS.parent.mkdir(parents=True, exist_ok=True)
    CACHE_GEMINI_NUEVOS.write_text(
        json.dumps(list(cache.values()), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    df = pd.read_csv(DATASET_CSV)

    # El panel de KPIs es sobre el corpus histórico real (se excluyen
    # Acuerdos Plenarios, igual que en el resto del pipeline de modelado).
    if "es_acuerdo_plenario" in df.columns:
        df = df[~df["es_acuerdo_plenario"]].copy()

    historicos = cargar_historicos()
    cache_gemini = cargar_cache_gemini_nuevos()

    textos_por_fila = [cargar_texto(jf) for jf in df["json_filename"]]
    resumenes_por_jf = cargar_resumenes()
    resumenes_por_fila = [
        resumenes_por_jf.get(jf, "") for jf in df["json_filename"]
    ]
    if resumenes_por_jf:
        print(
            f"Resúmenes cargados desde {RESUMENES_JSON.name}: "
            f"{len(resumenes_por_jf)} documentos con señal extra.\n"
        )

    # 1) Núcleo dinámico: los TOP_N_NUCLEO temas más frecuentes REALMENTE
    # en el corpus actual (no una lista fija en el código). Nota: el
    # núcleo se calcula SOLO sobre el texto completo (sin resumen), para
    # que las frecuencias que definen las barras propias del panel sigan
    # siendo comparables entre corridas aunque el set de resúmenes
    # disponibles cambie con el tiempo.
    nucleo_con_conteo = calcular_nucleo_dinamico(textos_por_fila, top_n=TOP_N_NUCLEO)
    nucleo = [tema for tema, _ in nucleo_con_conteo]

    NUCLEO_JSON.parent.mkdir(parents=True, exist_ok=True)
    NUCLEO_JSON.write_text(
        json.dumps(
            {
                "generado": datetime.now(timezone.utc).isoformat(),
                "total_documentos": len(df),
                "top_n": TOP_N_NUCLEO,
                "nucleo": [
                    {"tema": tema, "documentos": n} for tema, n in nucleo_con_conteo
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 2) Bucketing por documento usando ESE núcleo (no el fijo de detectar_temas.py).
    #    Documentos NUEVOS que el regex no logra clasificar reciben respaldo
    #    de Gemini (ver docstring arriba); los históricos no se reabren.
    if not API_KEY:
        print(
            "AVISO: no hay GEMINI_API_KEY configurada -- los documentos nuevos "
            "sin tema por regex se quedan como 'Otro' hasta la próxima corrida "
            "con la key disponible (no afecta a los históricos).\n"
        )

    client = None
    llamadas_gemini_nuevas = 0
    filas = []
    for (_, row), texto, texto_resumen in zip(
        df.iterrows(), textos_por_fila, resumenes_por_fila
    ):
        jf = row["json_filename"]
        temas = detectar_temas_con_otro(texto, texto_resumen=texto_resumen)
        temas_extendidos = detectar_temas_extendido_con_otro(
            texto, texto_resumen=texto_resumen
        )
        temas_panel = detectar_temas_panel_con_bucket(
            texto, nucleo=nucleo, texto_resumen=texto_resumen
        )
        fuente_tema = "regex"

        es_nuevo = jf not in historicos
        sin_tema_por_regex = temas == [OTRO]

        if es_nuevo and sin_tema_por_regex and API_KEY:
            resultado = cache_gemini.get(jf)
            if resultado is None:
                if client is None:
                    client = httpx.Client()
                try:
                    r = llamar_gemini(client, texto)
                    temas_gemini = [
                        t.strip() for t in r.get("temas", []) if isinstance(t, str) and t.strip()
                    ]
                    resultado = {
                        "json_filename": jf,
                        "temas": temas_gemini,
                        "confianza": r.get("confianza", "baja"),
                        "evidencia": r.get("evidencia", ""),
                    }
                except Exception as error:
                    resultado = {"json_filename": jf, "error": str(error)}
                cache_gemini[jf] = resultado
                guardar_cache_gemini_nuevos(cache_gemini)
                llamadas_gemini_nuevas += 1
                time.sleep(2.5)

            if (
                "error" not in resultado
                and resultado.get("confianza") in CONFIANZA_ACEPTABLE
                and resultado.get("temas")
            ):
                temas_rescatados = resultado["temas"]
                temas = temas_rescatados
                temas_extendidos = temas_rescatados
                fuente_tema = "gemini (documento nuevo, regex no encontró nada)"
                en_nucleo = [t for t in temas_rescatados if t in nucleo]
                temas_panel = en_nucleo if en_nucleo else [OTROS_ESPECIFICOS]

        filas.append(
            {
                "expediente": etiqueta_expediente(row),
                "json_filename": jf,
                "category": row.get("category", ""),
                "year": formatear_anio(row.get("year", "")),
                "sentido": row.get("sentido", "") or "no_detectado",
                "temas": ", ".join(temas),
                "temas_extendidos": ", ".join(temas_extendidos),
                "temas_panel": ", ".join(temas_panel),
                "fuente_tema": fuente_tema,
            }
        )

    if client is not None:
        client.close()

    salida = pd.DataFrame(filas)
    SALIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(SALIDA_CSV, index=False)

    total = len(salida)
    print(f"Documentos procesados: {total}")
    print(f"Guardado en: {SALIDA_CSV}")
    print(f"Núcleo actual guardado en: {NUCLEO_JSON}")
    if llamadas_gemini_nuevas:
        print(
            f"Documentos nuevos sin tema por regex, resueltos con Gemini "
            f"(respaldo gratuito, incremental): {llamadas_gemini_nuevas}"
        )
    print()
    print(
        f"Conteo para el gráfico del panel (columna temas_panel -- {TOP_N_NUCLEO} temas"
    )
    print("más frecuentes del corpus actual + bucket 'Otros temas identificados'")
    print("+ 'Otro / no identificado'):\n")
    print(f"{'TEMA':45s} {'#docs':>7s} {'%docs':>7s}")
    etiquetas_panel = nucleo + [OTROS_ESPECIFICOS, OTRO]
    for tema in etiquetas_panel:
        n = salida["temas_panel"].str.contains(tema, regex=False).sum()
        pct = 100.0 * n / total if total else 0.0
        print(f"{tema:45s} {n:7d} {pct:6.1f}%")


if __name__ == "__main__":
    main()
