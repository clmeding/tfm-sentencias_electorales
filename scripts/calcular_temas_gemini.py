"""Genera la versión final de data/temas_por_documento.csv combinando los
temas LIBRES que identificó Gemini (data/temas_gemini.json, generado por
scripts/detectar_temas_gemini.py) con el detector por palabras clave
(src/detectar_temas.py) como respaldo -- mismo espíritu que
build_dataset_modelo.py combinando sentido_gemini.json con el CSV base.

IMPORTANTE: los temas de Gemini son de vocabulario ABIERTO (no una lista
cerrada) -- este script todavía NO decide qué temas se quedan para los
gráficos del panel de KPIs, solo junta todo en un CSV con la etiqueta cruda
de cada documento. Antes de correr este script, corre
scripts/resumen_temas_libres.py para ver el universo completo de temas
descubiertos y decidir (con tu criterio jurídico) cuáles curar/fusionar; la
columna "temas" de este CSV usa por ahora la lista de 6 temas 'núcleo' ya
existente en src/detectar_temas.py SOLO como filtro de compatibilidad con
el panel actual -- actualízala ahí cuando definas tu lista curada final.

Para cada documento:
    - Si Gemini lo procesó con confianza "alta" o "media": se usa su
      resultado (lista de temas libres, o "Otro / no identificado" si
      Gemini devolvió lista vacía).
    - Si no (no se procesó, dio error, o confianza "baja"): se usa el
      detector por regex (detectar_temas_extendido_con_otro), igual que
      antes de este cambio.

Agrega una columna "fuente_tema" ("gemini" o "regex") para poder auditar,
documento por documento, de dónde salió cada etiqueta.

Requiere haber corrido antes (aunque sea parcialmente):
    python scripts/detectar_temas_gemini.py

Si data/temas_gemini.json no existe todavía, el script avisa y usa el
detector por regex para el 100% del corpus (equivalente a correr el
scripts/calcular_temas.py original).

Uso (desde la raíz del repo):
    python scripts/resumen_temas_libres.py     # ver el universo de temas primero
    python scripts/calcular_temas_gemini.py     # generar el CSV final
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from detectar_temas import (  # noqa: E402
    OTRO,
    OTROS_ESPECIFICOS,
    PATRONES_TEMA,
    detectar_temas_extendido_con_otro,
)

DATASET_CSV = REPO_ROOT / "data" / "dataset_sentencias.csv"
DOCUMENTS_DIR = REPO_ROOT / "documents" / "output"
TEMAS_GEMINI_JSON = REPO_ROOT / "data" / "temas_gemini.json"
SALIDA_CSV = REPO_ROOT / "data" / "temas_por_documento.csv"
NUCLEO_JSON = REPO_ROOT / "data" / "nucleo_temas_actual.json"

CONFIANZA_ACEPTABLE = {"alta", "media"}


def _cargar_nucleo_actual() -> list[str]:
    """Usa el mismo núcleo dinámico que guarda scripts/calcular_temas.py
    (data/nucleo_temas_actual.json) para que este CSV consolidado grafique
    el MISMO núcleo que el panel regex de rutina, en vez de una lista fija
    aparte. Si todavía no corriste calcular_temas.py, cae de vuelta a la
    lista estática de PATRONES_TEMA (y lo avisa)."""
    if NUCLEO_JSON.exists():
        try:
            data = json.loads(NUCLEO_JSON.read_text(encoding="utf-8"))
            return [item["tema"] for item in data["nucleo"]]
        except Exception:
            pass
    print(
        f"AVISO: no existe '{NUCLEO_JSON}' (corre primero "
        "scripts/calcular_temas.py para calcular el núcleo dinámico). "
        "Usando la lista estática de PATRONES_TEMA como respaldo."
    )
    return list(PATRONES_TEMA.keys())


def formatear_anio(valor) -> str:
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return ""
    try:
        return str(int(float(valor)))
    except (TypeError, ValueError):
        return str(valor)


def etiqueta_expediente(row) -> str:
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


def cargar_temas_gemini() -> dict[str, dict]:
    if not TEMAS_GEMINI_JSON.exists():
        print(
            f"AVISO: no existe '{TEMAS_GEMINI_JSON}'. Todo el corpus usará el "
            "detector por regex (corre antes scripts/detectar_temas_gemini.py "
            "si quieres usar Gemini)."
        )
        return {}
    registros = json.loads(TEMAS_GEMINI_JSON.read_text(encoding="utf-8"))
    return {r["json_filename"]: r for r in registros if "error" not in r}


def main() -> None:
    df = pd.read_csv(DATASET_CSV)
    if "es_acuerdo_plenario" in df.columns:
        df = df[~df["es_acuerdo_plenario"]].copy()

    temas_gemini = cargar_temas_gemini()
    temas_nucleo_actual_ordenado = _cargar_nucleo_actual()
    temas_nucleo_actual = set(temas_nucleo_actual_ordenado)

    filas = []
    n_gemini = 0
    n_regex = 0
    for _, row in df.iterrows():
        jf = row["json_filename"]
        registro_gemini = temas_gemini.get(jf)

        if registro_gemini and registro_gemini.get("confianza") in CONFIANZA_ACEPTABLE:
            temas_extendidos = [t.strip() for t in (registro_gemini.get("temas") or []) if t and t.strip()] or [OTRO]
            fuente = "gemini"
            n_gemini += 1
        else:
            texto = cargar_texto(jf)
            temas_extendidos = detectar_temas_extendido_con_otro(texto)
            fuente = "regex"
            n_regex += 1

        # "temas" (núcleo) filtra contra el núcleo DINÁMICO actual (el
        # mismo que calculó scripts/calcular_temas.py por frecuencia real
        # sobre el corpus regex -- ver data/nucleo_temas_actual.json) -- si
        # Gemini propuso un tema libre que no coincide con el núcleo actual,
        # no aparece aquí (pero sí en "temas_extendidos", sin filtrar).
        # Gemini es de vocabulario ABIERTO: a veces nombra el mismo tema del
        # núcleo con otras palabras ("Ejercicio del cargo" en vez de
        # "Derecho a ser votado / ejercicio del cargo"). Este alias SOLO
        # cubre variantes verificadas manualmente contra los conteos reales
        # de data/temas_gemini.json (ver conversación 09-sep-2026) -- no es
        # una fusión automática/difusa, por eso deja "Coalición electoral"
        # fuera: Gemini prácticamente nunca la nombra como tema central,
        # eso parece una discrepancia real de criterio con el regex, no un
        # problema de redacción, y merece tu revisión antes de fusionarse.
        ALIAS_NUCLEO = {
            "Remuneraciones y dietas": "Remuneraciones y dietas de cargos públicos",
            "Dietas y remuneraciones": "Remuneraciones y dietas de cargos públicos",
            "Ejercicio del cargo": "Derecho a ser votado / ejercicio del cargo",
        }
        temas_extendidos_norm = [ALIAS_NUCLEO.get(t, t) for t in temas_extendidos]
        temas_nucleo = [t for t in temas_extendidos_norm if t in temas_nucleo_actual]

        # "temas_panel" -- la columna que usa el gráfico del panel de KPIs.
        # Mismo criterio de 3 niveles que detectar_temas_panel_con_bucket()
        # en src/detectar_temas.py, pero aquí puede alimentarse de temas
        # LIBRES de Gemini (no solo regex): si hay tema núcleo -> esos; si
        # no pero se identificó CUALQUIER tema (de Gemini o del regex
        # extendido) -> "Otros temas identificados"; si de verdad no se
        # identificó nada -> "Otro / no identificado".
        if temas_nucleo:
            temas_panel = temas_nucleo
        elif temas_extendidos and temas_extendidos != [OTRO]:
            temas_panel = [OTROS_ESPECIFICOS]
        else:
            temas_panel = [OTRO]

        filas.append(
            {
                "expediente": etiqueta_expediente(row),
                "json_filename": jf,
                "category": row.get("category", ""),
                "year": formatear_anio(row.get("year", "")),
                "sentido": row.get("sentido", "") or "no_detectado",
                "temas": ", ".join(temas_nucleo),
                "temas_extendidos": ", ".join(temas_extendidos),
                "temas_panel": ", ".join(temas_panel),
                "fuente_tema": fuente,
            }
        )

    salida = pd.DataFrame(filas)
    SALIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    salida.to_csv(SALIDA_CSV, index=False)

    total = len(salida)
    print(f"Documentos procesados: {total}  (gemini: {n_gemini}, regex/respaldo: {n_regex})")
    print(f"Guardado en: {SALIDA_CSV}\n")
    print(f"Conteo para el gráfico del panel (columna temas_panel -- núcleo actual")
    print(f"de {len(temas_nucleo_actual)} temas + 'Otros temas identificados' + ")
    print("'Otro / no identificado'; ver scripts/resumen_temas_libres.py para el ")
    print("universo COMPLETO de temas libres que Gemini descubrió, antes de decidir ")
    print("si promover alguno):\n")
    print(f"{'TEMA':40s} {'#docs':>7s} {'%docs':>7s}")
    for tema in list(temas_nucleo_actual_ordenado) + [OTROS_ESPECIFICOS, OTRO]:
        n = salida["temas_panel"].str.contains(tema, regex=False).sum()
        pct = 100.0 * n / total if total else 0.0
        print(f"{tema:40s} {n:7d} {pct:6.1f}%")

    pct_otro_antes = None
    if n_gemini:
        # Comparación rápida para que veas el efecto de Gemini: cuántos de
        # los documentos donde SÍ se usó Gemini habrían quedado en "Otro"
        # con el regex, vs. con Gemini.
        otro_gemini = sum(
            1 for jf, r in temas_gemini.items()
            if r.get("confianza") in CONFIANZA_ACEPTABLE and not r.get("temas")
        )
        print(
            f"\nDe los {n_gemini} documentos etiquetados por Gemini, "
            f"{otro_gemini} quedaron en 'Otro / no identificado' "
            f"({100.0 * otro_gemini / n_gemini:.1f}%)."
        )


if __name__ == "__main__":
    main()
