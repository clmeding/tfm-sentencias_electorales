"""Genera embeddings semánticos para cada documento del dataset de
modelado usando la API de embeddings de Gemini (gemini-embedding-001),
en vez del modelo local sentence-transformers (generar_embeddings.py).

Por qué cambiar a Gemini: unifica el pipeline en una sola API (ya se usa
Gemini para los resúmenes, scripts/generar_resumenes.py), y permite
aprovechar un modelo de embeddings más reciente sin mantener una
dependencia pesada (~470 MB) de descarga local.

Límite de contexto: gemini-embedding-001 acepta hasta 2,048 tokens por
solicitud. Los documentos de este corpus (insumo de texto de ~2,000 a
15,000+ palabras) exceden ese límite, así que se parte cada documento en
fragmentos de ~1,300 palabras (margen conservador para no pasar de 2,048
tokens) y se promedian (mean pooling) los embeddings normalizados de
cada fragmento para obtener un solo vector por documento.

Dimensión de salida: 768 (de las opciones recomendadas 768/1536/3072).
Se eligió el valor más chico recomendado porque el set de entrenamiento
es pequeño (cientos de documentos) -- una dimensión menor reduce el
riesgo de sobreajuste en el clasificador y mantiene el archivo de
embeddings compacto.

Costo aproximado: a $0.15 por millón de tokens (capa de pago, estándar),
procesar ~7.8M tokens (873 documentos completos) cuesta ~$1.2 USD. La
capa gratuita tiene límites de 1,000 solicitudes/día y 30k tokens/minuto
para este modelo -- a esa velocidad tardaría varios días en completar
todo el corpus.

Preparación (una sola vez, se reutiliza la misma key de scripts/generar_resumenes.py):
    export GEMINI_API_KEY="tu-api-key-aqui"

Uso (desde la raíz del repo, después de build_dataset_modelo.py):
    python scripts/generar_embeddings_gemini.py

Guarda el avance después de CADA documento en data/embeddings_gemini_raw.json
(un vector por documento ya promediado); si se interrumpe, se puede volver
a correr y continúa donde se quedó. Al terminar, escribe los archivos
finales modelo/embeddings_documentos.npy y modelo/dataset_modelo_labels.parquet
en el mismo formato que usa scripts/entrenar_modelo.py.
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

WORDS_PER_CHUNK = 1300
MODELO = "gemini-embedding-001"
OUTPUT_DIM = 768
TASK_TYPE = "CLASSIFICATION"  # los embeddings alimentan un clasificador de sentido
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELO}:embedContent"

API_KEY = os.environ.get("GEMINI_API_KEY")

DATASET_MODELO = Path("data/dataset_modelo.parquet")
PROGRESO = Path("data/embeddings_gemini_raw.json")
EMBEDDINGS_OUT = Path("modelo/embeddings_documentos.npy")
LABELS_OUT = Path("modelo/dataset_modelo_labels.parquet")


def chunk_text(text: str, words_per_chunk: int = WORDS_PER_CHUNK) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    return [" ".join(words[i : i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]


def normalizar(vec: np.ndarray) -> np.ndarray:
    norma = np.linalg.norm(vec)
    return vec / norma if norma > 0 else vec


def embed_chunk(client: httpx.Client, texto: str, intentos: int = 5) -> list[float]:
    payload = {
        "model": f"models/{MODELO}",
        "content": {"parts": [{"text": texto}]},
        "taskType": TASK_TYPE,
        "outputDimensionality": OUTPUT_DIM,
    }
    espera = 8
    for _ in range(intentos):
        resp = client.post(URL, params={"key": API_KEY}, json=payload, timeout=90)
        if resp.status_code == 429:
            time.sleep(espera)
            espera = min(espera * 2, 60)
            continue
        resp.raise_for_status()
        return resp.json()["embedding"]["values"]
    resp.raise_for_status()


def embed_documento(client: httpx.Client, texto: str) -> list[float]:
    """Parte el documento en fragmentos, obtiene el embedding normalizado
    de cada uno y devuelve el promedio (mean pooling), también normalizado."""
    chunks = chunk_text(texto)
    vectores = [normalizar(np.array(embed_chunk(client, c), dtype=np.float32)) for c in chunks]
    promedio = np.mean(vectores, axis=0)
    return normalizar(promedio).tolist()


def main() -> None:
    if not API_KEY:
        sys.exit(
            "ERROR: no encontré la variable de entorno GEMINI_API_KEY.\n"
            "Defínela antes de correr el script (la misma que usas para generar_resumenes.py)."
        )

    df = pd.read_parquet(DATASET_MODELO)
    print(f"Documentos a procesar: {len(df)}")

    progreso: dict[str, list] = {}
    if PROGRESO.exists():
        progreso = json.loads(PROGRESO.read_text(encoding="utf-8"))

    pendientes = [
        (i, row) for i, row in df.iterrows() if row["json_filename"] not in progreso
    ]
    print(f"Ya generados: {len(progreso)} | pendientes en esta corrida: {len(pendientes)}")

    if pendientes:
        with httpx.Client() as client:
            for n, (i, row) in enumerate(pendientes, 1):
                jf = row["json_filename"]
                try:
                    vector = embed_documento(client, row["texto_insumo"])
                    progreso[jf] = vector
                    print(f"[{n}/{len(pendientes)}] OK: {jf}")
                except Exception as error:
                    print(f"[{n}/{len(pendientes)}] ERROR en {jf}: {error}", file=sys.stderr)

                if n % 5 == 0 or n == len(pendientes):
                    PROGRESO.write_text(json.dumps(progreso), encoding="utf-8")

                time.sleep(0.5)

    faltantes = [jf for jf in df["json_filename"] if jf not in progreso]
    if faltantes:
        print(f"\nAVISO: {len(faltantes)} documento(s) no obtuvieron embedding (revisa errores arriba) y se excluyen del archivo final.")

    df_final = df[df["json_filename"].isin(progreso.keys())].reset_index(drop=True)
    doc_embeddings = np.array([progreso[jf] for jf in df_final["json_filename"]], dtype=np.float32)

    EMBEDDINGS_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_OUT, doc_embeddings)
    df_final[["json_filename", "category", "expediente", "sentido", "metodo_corte"]].to_parquet(LABELS_OUT, index=False)
    print(f"\nGuardado: {EMBEDDINGS_OUT} ({doc_embeddings.shape}) y {LABELS_OUT} ({len(df_final)} filas)")


if __name__ == "__main__":
    main()
