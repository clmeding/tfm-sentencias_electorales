"""Diagnóstico de errores del clasificador de sentido: matriz de confusión
y detalle por clase, usando predicciones "out-of-fold" de validación
cruzada (cada documento se predice solo cuando estaba en el fold de
prueba, nunca en el de entrenamiento -- así se puede armar una matriz de
confusión sobre el 100% del dataset sin hacer trampa).

Objetivo: entender SI el problema es "el modelo es débil en general" o
"hay un par de clases específicas que se confunden entre sí" -- eso
determina si conviene tunear hiperparámetros, agregar features, o
simplemente agrupar clases que en la práctica son casi indistinguibles
por texto.

Requiere que ya existan (mismos archivos que evaluar_kfold.py y
comparar_escenarios_corte.py):
    modelo/embeddings_documentos.npy
    modelo/dataset_modelo_labels.parquet   (con columna metodo_corte)

Uso:
    python scripts/diagnostico_confusion.py [--k 5] [--escenario completo|sin_aproximado|ambos]

Para cada escenario imprime:
  - Reporte de clasificación agregado (precision/recall/F1 por clase,
    sobre todas las predicciones out-of-fold).
  - Matriz de confusión completa (guardada también como CSV).
  - Para cada clase: con qué otra(s) clase(s) se confunde más seguido,
    en términos absolutos y como % de sus propios documentos.

No reentrena el modelo oficial ni toca modelo/*.joblib -- es puramente
diagnóstico. Guarda:
    modelo/matriz_confusion_<escenario>.csv
    modelo/diagnostico_confusion_<escenario>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

EMBEDDINGS_PATH = Path("modelo/embeddings_documentos.npy")
LABELS_PATH = Path("modelo/dataset_modelo_labels.parquet")
MODELO_DIR = Path("modelo")

ESCENARIOS = {
    "completo": None,
    "sin_aproximado": "aproximado_gemini",
}


def diagnosticar_escenario(X: np.ndarray, y_raw: np.ndarray, nombre_escenario: str, k_deseado: int) -> dict:
    print(f"\n{'=' * 70}\nEscenario: {nombre_escenario}\n{'=' * 70}")
    print(f"Documentos: {X.shape[0]} | Dimensiones del embedding: {X.shape[1]}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    clases = list(le.classes_)
    n_clases = len(clases)

    conteos = pd.Series(y_raw).value_counts()
    min_clase = int(conteos.min())
    k = min(k_deseado, min_clase)
    if k < k_deseado:
        print(
            f"AVISO: la clase con menos ejemplos tiene solo {min_clase} "
            f"documento(s) -- se reduce k de {k_deseado} a {k}."
        )
    if k < 2:
        print(f"ERROR: no se puede evaluar '{nombre_escenario}', alguna clase tiene menos de 2 ejemplos.")
        return {"escenario": nombre_escenario, "error": "clase con menos de 2 ejemplos"}

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ])

    # Predicciones "out-of-fold": cada documento predicho solo por el modelo
    # entrenado sin él (sin fuga de información), sobre el 100% del dataset.
    y_pred = cross_val_predict(pipeline, X, y, cv=skf, n_jobs=None)

    print(f"\nReporte de clasificación agregado ({k} folds, out-of-fold):")
    reporte_txt = classification_report(
        y, y_pred, target_names=clases, zero_division=0, labels=range(n_clases)
    )
    print(reporte_txt)
    reporte_dict = classification_report(
        y, y_pred, target_names=clases, zero_division=0, labels=range(n_clases), output_dict=True
    )

    cm = confusion_matrix(y, y_pred, labels=range(n_clases))
    cm_df = pd.DataFrame(cm, index=clases, columns=clases)

    print("Matriz de confusión (filas = clase real, columnas = clase predicha):")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(cm_df.to_string())

    cm_path = MODELO_DIR / f"matriz_confusion_{nombre_escenario}.csv"
    cm_df.to_csv(cm_path, encoding="utf-8")
    print(f"\nGuardada matriz de confusión completa en {cm_path}")

    # Para cada clase real, top-2 clases con las que más se confunde
    # (excluyendo la diagonal, es decir, los aciertos).
    print("\nPrincipales confusiones por clase (excluyendo aciertos):")
    confusiones_por_clase = {}
    for i, clase in enumerate(clases):
        total_real = cm[i].sum()
        fila = cm[i].copy()
        aciertos = fila[i]
        fila[i] = 0  # oculta la diagonal para encontrar los errores
        if fila.sum() == 0:
            print(f"  {clase:28s} sin errores ({aciertos}/{total_real} correctos)")
            confusiones_por_clase[clase] = {"aciertos": int(aciertos), "total": int(total_real), "confusiones": []}
            continue

        top_idx = np.argsort(fila)[::-1]
        top_confusiones = []
        detalle_txt = []
        for j in top_idx:
            if fila[j] == 0:
                break
            pct = 100 * fila[j] / total_real if total_real else 0
            top_confusiones.append({"predicha_como": clases[j], "casos": int(fila[j]), "pct_de_la_clase": round(float(pct), 1)})
            detalle_txt.append(f"{clases[j]} ({fila[j]} casos, {pct:.0f}%)")
            if len(top_confusiones) >= 3:
                break

        pct_acierto = 100 * aciertos / total_real if total_real else 0
        print(f"  {clase:28s} acierto {pct_acierto:.0f}% ({aciertos}/{total_real}) -- se confunde con: {', '.join(detalle_txt)}")
        confusiones_por_clase[clase] = {
            "aciertos": int(aciertos),
            "total": int(total_real),
            "pct_acierto": round(float(pct_acierto), 1),
            "confusiones": top_confusiones,
        }

    resultado = {
        "escenario": nombre_escenario,
        "n_documentos": int(X.shape[0]),
        "k": k,
        "reporte_por_clase": reporte_dict,
        "confusiones_por_clase": confusiones_por_clase,
    }

    json_path = MODELO_DIR / f"diagnostico_confusion_{nombre_escenario}.json"
    json_path.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Guardado diagnóstico completo en {json_path}")

    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="Número de folds deseado (default: 5)")
    parser.add_argument(
        "--escenario", choices=["completo", "sin_aproximado", "ambos"], default="ambos",
        help="Qué escenario diagnosticar (default: ambos)",
    )
    args = parser.parse_args()

    X_full = np.load(EMBEDDINGS_PATH)
    labels_df = pd.read_parquet(LABELS_PATH)

    if "metodo_corte" not in labels_df.columns:
        raise SystemExit(
            "ERROR: modelo/dataset_modelo_labels.parquet no tiene la columna 'metodo_corte'."
        )

    escenarios_a_correr = ESCENARIOS if args.escenario == "ambos" else {args.escenario: ESCENARIOS[args.escenario]}

    for nombre, metodo_excluir in escenarios_a_correr.items():
        if metodo_excluir is None:
            mask = np.ones(len(labels_df), dtype=bool)
        else:
            mask = (labels_df["metodo_corte"] != metodo_excluir).to_numpy()

        X = X_full[mask]
        y_raw = labels_df.loc[mask, "sentido"].to_numpy()
        diagnosticar_escenario(X, y_raw, nombre, args.k)


if __name__ == "__main__":
    main()
