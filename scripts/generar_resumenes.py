"""
Genera resúmenes (hechos, agravios, resolución) para las sentencias del
TEEN, llamando directamente a la API de Gemini (Google AI Studio).

Reemplaza a los dos scripts anteriores (preparar_entrada_resumenes.py +
generar_resumenes_gemini.py): ahora es UN SOLO comando. Cada vez que
proceses documentos nuevos con el pipeline principal (main.py), corre
este script después -- solo generará el resumen de los documentos que
todavía no lo tengan, sin repetir los que ya están hechos.

Alcance: TODAS las sentencias (es_acuerdo_plenario == False), sin
importar si el clasificador de sentido las detectó o no -- son 873
documentos en total (a diferencia del set de 566 usado para entrenar el
modelo, que solo incluye los que sí tienen sentido detectado). Por
decisión del proyecto, el resumen se genera para las 873.

Preparación (una sola vez):
    pip install httpx pandas

Antes de correr, define tu API key como variable de entorno (NUNCA la
escribas directamente en el código ni la subas a GitHub):

    macOS / Linux:
        export GEMINI_API_KEY="tu-api-key-aqui"
    Windows (PowerShell):
        $env:GEMINI_API_KEY="tu-api-key-aqui"

Cómo obtener la key gratis: entra a https://aistudio.google.com, inicia
sesión, ve a "Get API key" -> "Create API key" -> "Create new project"
(así queda en capa gratuita, sin tarjeta de crédito).

Uso (desde la raíz del repo, después de procesar documentos nuevos):
    python scripts/generar_resumenes.py

El script guarda el avance después de CADA documento, así que si se
interrumpe (Ctrl+C, se acaba la cuota gratis por hoy, etc.) puedes volver
a correrlo y continúa exactamente donde se quedó, sin repetir trabajo ni
perder lo ya generado.
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

CSV_INVENTARIO = Path("data/dataset_sentencias.csv")
CORPUS_DIR = Path("documents/output")
SALIDA = Path("data/resumenes.json")

API_KEY = os.environ.get("GEMINI_API_KEY")

MODELO = "gemini-flash-lite-latest"  # modelo económico / capa gratuita
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELO}:generateContent"

PROMPT_SISTEMA = """Eres un asistente jurídico especializado en resumir sentencias del Tribunal Electoral del Estado de Nayarit (TEEN), para un proyecto académico de tesis de maestría.

A partir del texto completo de la sentencia que te doy, genera un resumen estructurado en tres campos, en español, con estilo formal y objetivo (como el de una nota de jurisprudencia), SIN inventar información que no esté en el texto:

- "hechos": 2-4 oraciones que expliquen el contexto y origen del asunto (quién promovió, contra qué acto o autoridad, y por qué).
- "agravios": 2-3 oraciones que resuman los argumentos o reclamos principales de la parte promovente.
- "resolucion": 2-4 oraciones que expliquen qué resolvió el Tribunal y sus razones principales.

Responde ÚNICAMENTE con un objeto JSON válido con exactamente esas tres claves (hechos, agravios, resolucion), sin texto adicional, sin markdown, sin comillas triples."""


def cargar_documentos_en_alcance() -> list[dict]:
    """Lee el inventario (CSV) y devuelve una lista con
    {json_filename, expediente, texto} para TODAS las sentencias
    (excluyendo Acuerdos Plenarios), leyendo el texto ya extraído
    directamente de documents/output/*.json.
    """
    df = pd.read_csv(CSV_INVENTARIO)
    en_alcance = df[df["es_acuerdo_plenario"] == False]

    documentos = []
    faltantes = []
    for _, row in en_alcance.iterrows():
        json_path = CORPUS_DIR / row["json_filename"]
        if not json_path.exists():
            faltantes.append(row["json_filename"])
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        expediente = row.get("expediente", "")
        # pd.isna() cubre tanto NaN (float) como None -- algunos documentos
        # no tienen expediente registrado en el inventario.
        if pd.isna(expediente):
            expediente = ""
        documentos.append(
            {
                "json_filename": row["json_filename"],
                "expediente": expediente,
                "texto": data.get("extracted_text_clean", "") or "",
            }
        )

    if faltantes:
        print(
            f"AVISO: {len(faltantes)} documentos del inventario no tienen "
            f"archivo en {CORPUS_DIR}/ todavía (¿falta correr el pipeline "
            "principal sobre ellos?)."
        )

    return documentos


def llamar_gemini(
    client: httpx.Client,
    texto: str,
    expediente: str,
    intentos: int = 5,
) -> dict:
    texto_recortado = texto[:60000]  # margen de seguridad de contexto
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            f"{PROMPT_SISTEMA}\n\nExpediente: {expediente}\n\n"
                            f"TEXTO DE LA SENTENCIA:\n{texto_recortado}"
                        )
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }
    espera = 8
    for _ in range(intentos):
        resp = client.post(URL, params={"key": API_KEY}, json=payload, timeout=90)
        if resp.status_code == 429:
            # límite de solicitudes por minuto de la capa gratuita: esperar y reintentar
            time.sleep(espera)
            espera = min(espera * 2, 60)
            continue
        resp.raise_for_status()
        data = resp.json()
        texto_respuesta = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(texto_respuesta)
    resp.raise_for_status()


def main() -> None:
    if not API_KEY:
        sys.exit(
            "ERROR: no encontré la variable de entorno GEMINI_API_KEY.\n"
            "Defínela antes de correr el script (ver instrucciones al "
            "inicio de este archivo)."
        )

    documentos = cargar_documentos_en_alcance()
    print(
        "Documentos en alcance (todas las sentencias, sin Acuerdos "
        f"Plenarios): {len(documentos)}"
    )

    resultados = []
    if SALIDA.exists():
        resultados = json.loads(SALIDA.read_text(encoding="utf-8"))

    # las entradas con error no cuentan como terminadas: se reintentan
    resultados = [r for r in resultados if "error" not in r]
    ya_hechos = {r["json_filename"] for r in resultados}
    pendientes = [d for d in documentos if d["json_filename"] not in ya_hechos]

    print(f"Ya generados: {len(ya_hechos)} | pendientes en esta corrida: {len(pendientes)}")

    if not pendientes:
        print("No hay resúmenes pendientes -- todo está al día.")
        return

    SALIDA.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        for i, doc in enumerate(pendientes, 1):
            jf = doc["json_filename"]
            exp = doc["expediente"]
            try:
                resumen = llamar_gemini(client, doc["texto"], exp)
                resultados.append(
                    {
                        "json_filename": jf,
                        "expediente": exp,
                        "hechos": resumen.get("hechos", ""),
                        "agravios": resumen.get("agravios", ""),
                        "resolucion": resumen.get("resolucion", ""),
                    }
                )
                print(f"[{i}/{len(pendientes)}] OK: {jf}")
            except Exception as error:
                print(f"[{i}/{len(pendientes)}] ERROR en {jf}: {error}", file=sys.stderr)
                resultados.append({"json_filename": jf, "expediente": exp, "error": str(error)})

            SALIDA.write_text(
                json.dumps(resultados, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            time.sleep(2.5)  # margen de cortesía para no exceder el límite gratuito por minuto

    errores = [r for r in resultados if "error" in r]
    print(f"\nListo. Total guardados: {len(resultados)} | errores: {len(errores)}")


if __name__ == "__main__":
    main()
