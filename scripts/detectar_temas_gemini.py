"""Identifica el TEMA (materia) de TODOS los documentos del corpus histórico
usando la API de Gemini, de forma ABIERTA (no restringida a una lista
cerrada) -- la idea es descubrir el universo real de temas que trata el
corpus, incluyendo asuntos que el detector por palabras clave
(src/detectar_temas.py) nunca contempló. La curación de qué temas se
quedan para los gráficos del panel de KPIs (agrupar variantes, descartar
temas de un solo documento, etc.) es un paso POSTERIOR y separado -- este
script solo hace el descubrimiento.

Mismo patrón que scripts/detectar_sentido_gemini.py (que ya usaste para
ampliar el corpus con "sentido"): NO modifica ningún archivo existente,
genera un archivo aparte, resumible, con una fila por documento intentado:

    data/temas_gemini.json
    [
        {
            "json_filename": "...",
            "temas": ["Fiscalización / Recursos públicos", ...],
            "confianza": "alta" | "media" | "baja",
            "evidencia": "frase corta que justifica el/los tema(s) elegido(s)",
        },
        ...
    ]

`scripts/calcular_temas_gemini.py` combina este archivo con el detector por
regex para producir el data/temas_por_documento.csv final (usa Gemini
cuando confianza es "alta"/"media", regex de respaldo en el resto).

IMPORTANTE -- costo y alcance: a diferencia de detectar_sentido_gemini.py
(que solo corría sobre los ~307 documentos "no_detectado"), este script
corre sobre TODO el corpus (por defecto, salvo Acuerdos Plenarios) porque
pediste mejorar el tema en todos los documentos, no solo llenar huecos. Eso
son ~850+ llamadas al modelo económico "gemini-flash-lite-latest" (el mismo
que ya usaste antes) -- barato por llamada, pero no gratis. Si prefieres
acotar el gasto, puedes correr primero con --solo-otro para procesar
solamente los documentos que el regex ya marca como "Otro / no
identificado" (~la fracción que más te interesa mejorar) y decidir después
si vale la pena correr el resto.

Preparación (una sola vez, se reutiliza la misma key ya usada antes):
    export GEMINI_API_KEY="tu-api-key-aqui"

Uso (desde la raíz del repo):
    python scripts/detectar_temas_gemini.py                # corpus completo
    python scripts/detectar_temas_gemini.py --solo-otro     # solo "Otro" del regex (más barato)
    python scripts/detectar_temas_gemini.py --limite 50     # prueba rápida con 50 documentos

Guarda el avance después de CADA documento -- si se interrumpe (o se corta
la cuota), se puede volver a correr y continúa donde se quedó.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from detectar_temas import (  # noqa: E402
    OTRO,
    PATRONES_TEMA_TODOS,
    detectar_temas_extendido_con_otro,
)

CSV_INVENTARIO = REPO_ROOT / "data" / "dataset_sentencias.csv"
CORPUS_DIR = REPO_ROOT / "documents" / "output"
SALIDA = REPO_ROOT / "data" / "temas_gemini.json"

API_KEY = os.environ.get("GEMINI_API_KEY")

MODELO = "gemini-flash-lite-latest"  # mismo modelo económico usado en detectar_sentido_gemini.py
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELO}:generateContent"

CONFIANZA_ACEPTABLE = {"alta", "media"}

# Estos 17 NO son una lista cerrada -- son EJEMPLOS que ya sabemos que
# existen en el corpus (los 6 del panel de KPIs + 11 adicionales), para
# darle a Gemini una referencia de nivel de detalle y terminología esperada
# (evitar que un tema salga demasiado genérico como "Derecho electoral" o
# demasiado específico como el nombre de una persona). Gemini puede -- y
# debe -- proponer temas nuevos que no estén en esta lista si el asunto
# central del caso no calza bien con ninguno de estos ejemplos.
TEMAS_EJEMPLO = list(PATRONES_TEMA_TODOS.keys())

# Definiciones cortas para reducir falsos positivos por mención de paso (el
# mismo problema que ya diagnosticaste con el regex: p.ej. "coalición"
# aparece en el nombre de un partido sin que el caso trate de coaliciones).
# Ajusta esta redacción si tienes un criterio más preciso para alguno.
DEFINICIONES_TEMA = {
    "Violencia política de género": "el caso trata de violencia ejercida contra una mujer POR SU CONDICIÓN DE GÉNERO en el ejercicio de un cargo o candidatura (no cualquier conflicto entre una mujer y otra persona).",
    "Actos anticipados de campaña": "se analiza si actos de un aspirante o precandidato constituyeron actos anticipados de campaña o precampaña, fuera de los plazos legales.",
    "Fiscalización / Recursos públicos": "el caso trata del origen, destino, tope o rendición de cuentas de recursos públicos o privados de un partido/candidato (gastos de campaña, rebase de topes, fiscalización).",
    "Paridad de género": "el caso trata del cumplimiento (o incumplimiento) del principio de paridad de género en la postulación de candidaturas.",
    "Nulidad de elección": "se solicita o analiza la nulidad de una elección o de un cómputo municipal/distrital por irregularidades graves.",
    "Propaganda gubernamental": "el caso trata de propaganda de un gobierno o funcionario público (incluyendo informes de labores) que pudiera constituir promoción personalizada indebida.",
    "Vida interna de partidos políticos": "el conflicto es interno de un partido político (estatutos, militancia, afiliación, disciplina interna), no una disputa electoral entre partidos distintos.",
    "Financiamiento y prerrogativas": "el caso trata de prerrogativas o financiamiento público/privado de partidos, distinto de un procedimiento de fiscalización por gasto irregular.",
    "Registro y sustitución de candidaturas": "el caso trata del registro, sustitución o cancelación del registro de una candidatura específica, o de candidaturas comunes.",
    "Desistimiento": "el actor se desistió del medio de impugnación y el caso gira en torno a esa decisión procesal.",
    "Usos y costumbres / Sistema normativo indígena": "el caso involucra un sistema normativo indígena, usos y costumbres, o una consulta a un pueblo indígena.",
    "Integración de órganos electorales": "el caso trata de la integración de un consejo electoral, ayuntamiento, o la toma de protesta de sus integrantes.",
    "Acción afirmativa": "el caso trata de una acción afirmativa (medida especial de inclusión) distinta de paridad de género pura.",
    "Debates y encuestas electorales": "el caso trata de un debate entre candidatos o una encuesta electoral/de preferencias.",
    "Reelección": "el caso trata de la reelección de una persona en un cargo de elección popular.",
    "Credencial para votar": "el caso gira en torno a la credencial para votar de una persona.",
    "Recuento y nulidad de casilla": "el caso trata del recuento de votos o la nulidad de la votación recibida en una o varias casillas específicas.",
}

PROMPT_SISTEMA = """Eres un asistente jurídico especializado en sentencias del Tribunal Electoral del Estado de Nayarit (TEEN), para un proyecto académico de tesis de maestría.

El texto que te doy proviene de OCR y puede tener errores de reconocimiento de caracteres. Tu tarea es identificar el TEMA CENTRAL (materia principal) de la sentencia -- de qué trata realmente el asunto, no menciones de paso.

Estos son EJEMPLOS de temas ya identificados en otros documentos de este mismo corpus, con el nivel de detalle y el estilo de redacción esperado (2 a 6 palabras, terminología jurídico-electoral estándar en español, NO son la única lista posible):
{temas_con_definiciones}

Usa uno de estos ejemplos TAL CUAL (mismo texto exacto) si de verdad aplica al caso. Si el asunto central del caso NO está bien representado por ninguno de estos ejemplos, PROPON TÚ MISMO un tema nuevo, corto (2 a 6 palabras), en español, con el mismo estilo de los ejemplos (terminología jurídico-electoral estándar, ni demasiado genérico como "Derecho electoral" ni demasiado específico como un nombre propio o número de expediente). El objetivo es identificar TODOS los temas reales del corpus, no limitarse a los ejemplos -- así que no dudes en proponer uno nuevo cuando corresponda.

Reglas importantes:
- NO elijas ni inventes un tema solo porque una palabra aparece una o dos veces de forma incidental (ej. "coalición" en el nombre de un partido, o una mención de paso a un tema que no es el objeto de la controversia).
- Si genuinamente no puedes identificar un tema central con algo de certeza (texto muy dañado por OCR, caso atípico sin materia clara), responde con la lista vacía [] -- es una respuesta válida, se interpretará como "Otro / no identificado".
- Elige entre 1 y 3 temas por documento, en orden de relevancia (normalmente basta con 1).

Responde ÚNICAMENTE con un objeto JSON válido con exactamente estas claves, sin texto adicional, sin markdown:
- "temas": lista de 0 a 3 strings (cada uno un tema, de los ejemplos o propuesto por ti), o lista vacía si no identificas ninguno con certeza.
- "confianza": "alta", "media" o "baja".
- "evidencia": frase corta (máx 25 palabras) del texto que justifica el/los tema(s) elegido(s), o por qué no aplica ninguno."""


def _construir_bloque_temas() -> str:
    lineas = []
    for tema in TEMAS_EJEMPLO:
        definicion = DEFINICIONES_TEMA.get(tema, "")
        lineas.append(f"- {tema}: {definicion}" if definicion else f"- {tema}")
    return "\n".join(lineas)


def cargar_documentos(solo_otro: bool, limite: int | None) -> list[dict]:
    df = pd.read_csv(CSV_INVENTARIO)
    if "es_acuerdo_plenario" in df.columns:
        df = df[~df["es_acuerdo_plenario"]].copy()

    documentos = []
    faltantes = []
    for _, row in df.iterrows():
        json_path = CORPUS_DIR / row["json_filename"]
        if not json_path.exists():
            faltantes.append(row["json_filename"])
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        texto = data.get("extracted_text_clean", "") or ""

        if solo_otro:
            temas_regex = detectar_temas_extendido_con_otro(texto)
            if temas_regex != [OTRO]:
                continue

        documentos.append({"json_filename": row["json_filename"], "texto": texto})

    if faltantes:
        print(f"AVISO: {len(faltantes)} documentos del inventario no tienen archivo en {CORPUS_DIR}/.")

    if limite:
        documentos = documentos[:limite]

    return documentos


def llamar_gemini(client: httpx.Client, texto: str, intentos: int = 5) -> dict:
    texto_recortado = texto[:60000]
    prompt = PROMPT_SISTEMA.format(temas_con_definiciones=_construir_bloque_temas())
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{prompt}\n\nTEXTO DE LA SENTENCIA:\n{texto_recortado}"}],
            }
        ],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }
    espera = 8
    for _ in range(intentos):
        resp = client.post(URL, params={"key": API_KEY}, json=payload, timeout=90)
        if resp.status_code == 429:
            time.sleep(espera)
            espera = min(espera * 2, 60)
            continue
        resp.raise_for_status()
        data = resp.json()
        texto_respuesta = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(texto_respuesta)
    resp.raise_for_status()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--solo-otro",
        action="store_true",
        help="Procesa solo los documentos que el detector por regex ya marca como 'Otro / no identificado' (más barato).",
    )
    parser.add_argument("--limite", type=int, default=None, help="Procesa como máximo N documentos (para pruebas).")
    args = parser.parse_args()

    if not API_KEY:
        sys.exit(
            "ERROR: no encontré la variable de entorno GEMINI_API_KEY.\n"
            "Defínela antes de correr el script (la misma que usas para detectar_sentido_gemini.py)."
        )

    documentos = cargar_documentos(solo_otro=args.solo_otro, limite=args.limite)
    print(f"Documentos a intentar clasificar con Gemini: {len(documentos)}")
    if args.solo_otro:
        print("(modo --solo-otro: solo documentos que el regex marca como 'Otro / no identificado')")

    resultados = []
    if SALIDA.exists():
        resultados = json.loads(SALIDA.read_text(encoding="utf-8"))
    resultados = [r for r in resultados if "error" not in r]
    ya_hechos = {r["json_filename"] for r in resultados}
    pendientes = [d for d in documentos if d["json_filename"] not in ya_hechos]

    print(f"Ya procesados: {len(ya_hechos)} | pendientes en esta corrida: {len(pendientes)}")

    if not pendientes:
        print("No hay documentos pendientes -- todo está al día.")
        return

    SALIDA.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        for i, doc in enumerate(pendientes, 1):
            jf = doc["json_filename"]
            try:
                r = llamar_gemini(client, doc["texto"])
                temas = [t.strip() for t in r.get("temas", []) if isinstance(t, str) and t.strip()]
                confianza = r.get("confianza", "baja")
                resultados.append(
                    {
                        "json_filename": jf,
                        "temas": temas,
                        "confianza": confianza,
                        "evidencia": r.get("evidencia", ""),
                    }
                )
                etiqueta = ", ".join(temas) if temas else OTRO
                print(f"[{i}/{len(pendientes)}] {jf} -> {etiqueta} ({confianza})")
            except Exception as error:
                print(f"[{i}/{len(pendientes)}] ERROR en {jf}: {error}", file=sys.stderr)
                resultados.append({"json_filename": jf, "error": str(error)})

            SALIDA.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(2.5)

    usables = [r for r in resultados if "error" not in r and r.get("confianza") in CONFIANZA_ACEPTABLE]
    errores = [r for r in resultados if "error" in r]
    print(f"\nListo. Total intentados: {len(resultados)} | usables (confianza alta/media): {len(usables)} | errores: {len(errores)}")
    print("Siguiente paso: python scripts/calcular_temas_gemini.py")


if __name__ == "__main__":
    main()
