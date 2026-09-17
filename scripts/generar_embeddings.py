"""Genera embeddings semánticos (modelo multilingüe con soporte de español,
basado en arquitectura BERT) para cada documento del dataset de modelado.

Cada documento se parte en fragmentos de ~90 palabras (límite práctico del
modelo: 128 tokens por fragmento) y se promedian los embeddings de todos
sus fragmentos para obtener un solo vector de 768 dimensiones por
documento (mean pooling). Esto permite representar documentos de
cualquier longitud con un modelo que solo puede leer fragmentos cortos
a la vez.

La primera ejecución descarga el modelo (~470 MB) desde Hugging Face;
las siguientes usan la copia local en caché.

Uso:
    python scripts/generar_embeddings.py
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

WORDS_PER_CHUNK = 90
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

DATASET_MODELO = Path("data/dataset_modelo.parquet")
EMBEDDINGS_OUT = Path("modelo/embeddings_documentos.npy")
LABELS_OUT = Path("modelo/dataset_modelo_labels.parquet")


def chunk_text(text: str, words_per_chunk: int = WORDS_PER_CHUNK) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def main() -> None:
    df = pd.read_parquet(DATASET_MODELO)
    print(f"Documentos a procesar: {len(df)}")

    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()

    all_chunks: list[str] = []
    doc_chunk_ranges: list[tuple[int, int]] = []

    for text in df["texto_insumo"]:
        chunks = chunk_text(text)
        start = len(all_chunks)
        all_chunks.extend(chunks)
        doc_chunk_ranges.append((start, len(all_chunks)))

    print(f"Total de fragmentos a codificar: {len(all_chunks)}")

    start_time = time.time()
    chunk_embeddings = model.encode(
        all_chunks,
        batch_size=64,
        show_progress_bar=True,
    )
    elapsed = time.time() - start_time
    print(f"Embeddings de fragmentos calculados en {elapsed:.1f}s")

    doc_embeddings = np.zeros((len(df), dim), dtype=np.float32)
    for i, (start, end) in enumerate(doc_chunk_ranges):
        doc_embeddings[i] = chunk_embeddings[start:end].mean(axis=0)

    EMBEDDINGS_OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_OUT, doc_embeddings)
    columnas = ["json_filename", "category", "expediente", "sentido"]
    if "metodo_corte" in df.columns:
        columnas.append("metodo_corte")
    else:
        print(
            "AVISO: data/dataset_modelo.parquet no tiene columna 'metodo_corte' "
            "-- scripts/comparar_escenarios_corte.py no podrá filtrar por "
            "escenario. Si la necesitas, vuelve a correr build_dataset_modelo.py "
            "con la versión que la genera."
        )
    df[columnas].to_parquet(LABELS_OUT, index=False)
    print(f"\nGuardado: {EMBEDDINGS_OUT} y {LABELS_OUT}")


if __name__ == "__main__":
    main()
