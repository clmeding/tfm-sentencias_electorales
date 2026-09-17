"""Reconstruye, de forma reproducible y sobre el corpus completo (856
documentos, no una muestra de 40), la comparación que originalmente
justificó usar TF-IDF en vez de embeddings semánticos para "precedentes
similares" (ver docstring de generar_precedentes_tfidf.py).

Qué mide
--------
Para cada documento del corpus, busca su vecino más cercano (excluyéndose
a sí mismo) bajo dos representaciones -- TF-IDF y embedding semántico
(mpnet, el mismo que usa el clasificador) -- y registra si ese vecino
comparte la misma "category" (proxy de "mismo asunto", igual que en la
prueba original). Se hace por validación cruzada tipo leave-one-out:
cada uno de los 856 documentos actúa una vez como consulta.

También reporta la misma métrica para el top-4 vecinos, que es el número
real que usa producción (MAX_PRECEDENTES_POR_LADO en explicar_prediccion.py).

Esto NO reemplaza el mecanismo de producción (que sigue usando TF-IDF vía
generar_precedentes_tfidf.py) -- es un script de diagnóstico/evidencia,
igual que evaluar_kfold.py o comparar_modelos_clasificacion.py.

Uso:
    python scripts/comparar_precedentes_tfidf_vs_semantico.py

Requiere modelo/embeddings_documentos.npy, modelo/tfidf_matriz_corpus.npz
y modelo/dataset_modelo_labels.parquet (columna "category").
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

MODELO_DIR = Path("modelo")
TOP_K_PRODUCCION = 4


def vecino_mas_cercano_por_fila(sim: np.ndarray) -> np.ndarray:
    """Índice del vecino más similar excluyendo la diagonal (self-match)."""
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)
    return sim.argmax(axis=1)


def top_k_vecinos_por_fila(sim: np.ndarray, k: int) -> np.ndarray:
    sim = sim.copy()
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :k]


def evaluar_representacion(sim: np.ndarray, categorias: np.ndarray, nombre: str, k: int) -> dict:
    n = len(categorias)

    vecino1 = vecino_mas_cercano_por_fila(sim)
    mismatches_top1 = int(np.sum(categorias[vecino1] != categorias))

    vecinosK = top_k_vecinos_por_fila(sim, k)
    total_checks = 0
    mismatches_topK = 0
    for i in range(n):
        for j in vecinosK[i]:
            total_checks += 1
            if categorias[j] != categorias[i]:
                mismatches_topK += 1

    resultado = {
        "representacion": nombre,
        "top1_vecino_categoria_distinta": mismatches_top1,
        "top1_total_documentos": n,
        "top1_tasa_error": round(mismatches_top1 / n, 4),
        f"top{k}_vecino_categoria_distinta": mismatches_topK,
        f"top{k}_total_checks": total_checks,
        f"top{k}_tasa_error": round(mismatches_topK / total_checks, 4),
    }
    print(
        f"  [{nombre}] vecino más cercano (top-1): {mismatches_top1}/{n} con categoría distinta "
        f"({resultado['top1_tasa_error']:.1%})"
    )
    print(
        f"  [{nombre}] top-{k} vecinos (el que usa producción): {mismatches_topK}/{total_checks} "
        f"con categoría distinta ({resultado[f'top{k}_tasa_error']:.1%})"
    )
    return resultado


def main() -> None:
    embeddings = np.load(MODELO_DIR / "embeddings_documentos.npy")
    tfidf = sparse.load_npz(MODELO_DIR / "tfidf_matriz_corpus.npz")
    labels_df = pd.read_parquet(MODELO_DIR / "dataset_modelo_labels.parquet")
    categorias = labels_df["category"].to_numpy()

    print(f"Documentos: {len(categorias)} | categorías únicas: {len(set(categorias))}\n")

    print("Calculando similitud semántica (embeddings mpnet, 768-dim)...")
    sim_semantica = cosine_similarity(embeddings)
    print("Calculando similitud TF-IDF...")
    sim_tfidf = cosine_similarity(tfidf)

    print(f"\n{'=' * 70}\nResultados (leave-one-out sobre los {len(categorias)} documentos actuales)\n{'=' * 70}")
    r_semantica = evaluar_representacion(sim_semantica, categorias, "embedding_semantico_mpnet", TOP_K_PRODUCCION)
    r_tfidf = evaluar_representacion(sim_tfidf, categorias, "tfidf_produccion", TOP_K_PRODUCCION)

    print(f"\n{'=' * 70}\nComparación directa con el hallazgo original (muestra de 40, ver "
          f"generar_precedentes_tfidf.py):\n"
          f"  Original -- documento completo + semántico: 18/40 (45.0%)\n"
          f"  Original -- documento completo + TF-IDF:     16/40 (40.0%)\n"
          f"  Reproducido hoy (856 docs, vecino más cercano):\n"
          f"    Semántico: {r_semantica['top1_vecino_categoria_distinta']}/{r_semantica['top1_total_documentos']} "
          f"({r_semantica['top1_tasa_error']:.1%})\n"
          f"    TF-IDF:    {r_tfidf['top1_vecino_categoria_distinta']}/{r_tfidf['top1_total_documentos']} "
          f"({r_tfidf['top1_tasa_error']:.1%})")

    salida = MODELO_DIR / "comparacion_precedentes_tfidf_vs_semantico.json"
    salida.write_text(
        json.dumps({"semantico": r_semantica, "tfidf": r_tfidf}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nGuardado detalle completo en {salida}")


if __name__ == "__main__":
    main()
