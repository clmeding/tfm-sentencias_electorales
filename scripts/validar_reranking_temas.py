"""Valida si reordenar precedentes por tema compartido (usar_temas=True en
precedentes_relevantes()) mejora los resultados frente al orden actual
(solo similitud TF-IDF, usar_temas=False) -- ANTES de activarlo por
defecto en la demo.

Mismo criterio de validacion que el usado para justificar el cambio de
embeddings semanticos a TF-IDF (ver scripts/generar_precedentes_tfidf.py):
se usa `category` como proxy de "mismo asunto" sobre una muestra de
documentos reales, y se compara que tan seguido el/los precedente(s)
mostrados tienen una categoria DISTINTA a la del documento evaluado (menos
es mejor).

Para cada documento de la muestra:
  1. Se corre precedentes_relevantes() con usar_temas=False (orden actual).
  2. Se corre precedentes_relevantes() con usar_temas=True (orden nuevo).
  3. Se compara, en ambos casos, que fraccion de los precedentes
     mostrados (a_favor + otros, top max_por_lado de cada lado) tiene
     una categoria distinta a la del documento evaluado.

Se usa el `sentido` REAL (no el predicho por el clasificador) como
`sentido_predicho` en la llamada -- asi la comparacion aisla el efecto del
reordenamiento por tema y no mezcla errores del clasificador.

Uso (desde la raiz del repo, con el entorno virtual activado):
    python scripts/validar_reranking_temas.py
    python scripts/validar_reranking_temas.py --n 60 --seed 7

Requiere (ademas de lo que ya requiere explicar_prediccion.py):
    data/temas_por_documento.csv   (scripts/calcular_temas.py)
Si no existe, el script avisa y termina sin comparar nada -- correr antes:
    python scripts/calcular_temas.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import pandas as pd  # noqa: E402

import explicar_prediccion as ep  # noqa: E402

DATA_PATH = REPO_ROOT / "data" / "dataset_modelo.parquet"


def _fraccion_categoria_distinta(precedentes: dict, categoria_query) -> tuple[int, int]:
    """Devuelve (num_distinta, total) sobre a_favor + otros."""
    total = 0
    distinta = 0
    for lado in ("a_favor", "otros"):
        for p in precedentes[lado]:
            total += 1
            if p.get("category") != categoria_query:
                distinta += 1
    return distinta, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40, help="Tamano de la muestra (default 40, igual que el diagnostico previo TF-IDF vs embeddings)")
    parser.add_argument("--seed", type=int, default=7, help="random_state para la muestra (default 7)")
    args = parser.parse_args()

    if not ep.TEMAS_CSV.exists():
        print(
            f"ERROR: no existe '{ep.TEMAS_CSV}'. Corre primero:\n"
            "    python scripts/calcular_temas.py\n"
            "antes de poder validar el reordenamiento por tema."
        )
        raise SystemExit(1)

    if not DATA_PATH.exists():
        print(f"ERROR: no existe '{DATA_PATH}'. Corre antes scripts/build_dataset_modelo.py.")
        raise SystemExit(1)

    df = pd.read_parquet(DATA_PATH)
    if "category" not in df.columns:
        print("ERROR: dataset_modelo.parquet no tiene columna 'category'.")
        raise SystemExit(1)

    muestra = df.sample(min(args.n, len(df)), random_state=args.seed)

    print(f"Muestra: {len(muestra)} documentos (seed={args.seed})\n")

    filas_resultado = []
    total_distinta_actual = 0
    total_distinta_nuevo = 0
    total_precedentes_actual = 0
    total_precedentes_nuevo = 0
    cambia_top1 = 0
    sin_precedentes_actual = 0
    sin_precedentes_nuevo = 0

    for i, (_, row) in enumerate(muestra.iterrows(), start=1):
        texto = row["texto_insumo"]
        sentido_real = row["sentido"]
        categoria_real = row.get("category")
        jf = row["json_filename"]

        actual = ep.precedentes_relevantes(
            texto, sentido_real, excluir_json_filename=jf, usar_temas=False
        )
        nuevo = ep.precedentes_relevantes(
            texto, sentido_real, excluir_json_filename=jf, usar_temas=True
        )

        d_actual, t_actual = _fraccion_categoria_distinta(actual, categoria_real)
        d_nuevo, t_nuevo = _fraccion_categoria_distinta(nuevo, categoria_real)

        total_distinta_actual += d_actual
        total_distinta_nuevo += d_nuevo
        total_precedentes_actual += t_actual
        total_precedentes_nuevo += t_nuevo

        if t_actual == 0:
            sin_precedentes_actual += 1
        if t_nuevo == 0:
            sin_precedentes_nuevo += 1

        top1_actual = actual["a_favor"][0]["json_filename"] if actual["a_favor"] else None
        top1_nuevo = nuevo["a_favor"][0]["json_filename"] if nuevo["a_favor"] else None
        if top1_actual != top1_nuevo:
            cambia_top1 += 1

        filas_resultado.append(
            {
                "json_filename": jf,
                "category": categoria_real,
                "temas_documento": ", ".join(nuevo.get("temas_documento", [])) or "(ninguno)",
                "distinta_actual": f"{d_actual}/{t_actual}",
                "distinta_nuevo": f"{d_nuevo}/{t_nuevo}",
                "cambio_top1": top1_actual != top1_nuevo,
            }
        )

        print(
            f"[{i:2d}/{len(muestra)}] {jf:45s} categoria={str(categoria_real):20s} "
            f"temas_doc={', '.join(nuevo.get('temas_documento', [])) or '(ninguno)':45s} "
            f"actual={d_actual}/{t_actual}  nuevo={d_nuevo}/{t_nuevo}"
            + ("  <-- cambia top1" if top1_actual != top1_nuevo else "")
        )

    print("\n" + "=" * 78)
    print("RESUMEN")
    print("=" * 78)
    pct_actual = 100.0 * total_distinta_actual / total_precedentes_actual if total_precedentes_actual else float("nan")
    pct_nuevo = 100.0 * total_distinta_nuevo / total_precedentes_nuevo if total_precedentes_nuevo else float("nan")
    print(
        f"Orden actual  (solo similitud):  {total_distinta_actual}/{total_precedentes_actual} "
        f"precedentes con categoria distinta ({pct_actual:.1f}%)"
    )
    print(
        f"Orden nuevo   (tema + similitud): {total_distinta_nuevo}/{total_precedentes_nuevo} "
        f"precedentes con categoria distinta ({pct_nuevo:.1f}%)"
    )
    print(f"\nDocumentos sin ningun precedente (orden actual): {sin_precedentes_actual}/{len(muestra)}")
    print(f"Documentos sin ningun precedente (orden nuevo):  {sin_precedentes_nuevo}/{len(muestra)}")
    print(f"\nCasos donde cambia el precedente top-1 (a_favor): {cambia_top1}/{len(muestra)}")

    print(
        "\nInterpretacion: si el % de 'categoria distinta' del orden NUEVO es "
        "menor que el del orden ACTUAL, el reordenamiento por tema esta "
        "ayudando (menos precedentes de asunto distinto llegan al top que se "
        "muestra en la demo). Si es igual o peor, o si aumenta mucho el numero "
        "de documentos sin precedentes, no conviene activarlo por defecto -- "
        "puede ser que el detector de temas por palabras clave (src/"
        "detectar_temas.py) tenga muy baja cobertura (muchos 'Otro') para que "
        "la senal sea util todavia."
    )

    salida_csv = REPO_ROOT / "data" / "validacion_reranking_temas.csv"
    pd.DataFrame(filas_resultado).to_csv(salida_csv, index=False)
    print(f"\nDetalle por documento guardado en: {salida_csv}")


if __name__ == "__main__":
    main()
