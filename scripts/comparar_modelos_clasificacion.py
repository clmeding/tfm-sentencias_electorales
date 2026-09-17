"""Compara la regresión logística (modelo actual) contra dos alternativas
-- SVM lineal y random forest -- bajo el MISMO protocolo de validación
cruzada k-fold que scripts/evaluar_kfold.py, sobre los mismos embeddings
y los mismos folds.

Por qué este script existe
---------------------------
La regresión logística se eligió por razonamiento metodológico (corpus
pequeño frente a 768 dimensiones, función convexa, no añade opacidad
extra sobre el embedding, barata para que LIME la consulte repetidamente)
pero nunca se comparó empíricamente contra otros algoritmos sobre este
corpus. Este script cierra ese hueco: entrena los tres modelos sobre
las MISMAS particiones de cada fold (mismo random_state=42), para que la
comparación sea justa, y reporta accuracy / F1 macro / F1 ponderado /
F1 por clase de cada uno.

Solo corre el escenario "completo" (856 documentos, sin excluir el
método de corte aproximado) para mantenerlo simple y rápido -- si el
resultado cambia qué modelo conviene, se puede extender después al
escenario "sin_aproximado".

Uso:
    python scripts/comparar_modelos_clasificacion.py [--k 5]

Requiere haber corrido antes build_dataset_modelo.py y generar_embeddings.py
(o la versión Gemini), igual que evaluar_kfold.py.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

EMBEDDINGS_PATH = Path("modelo/embeddings_documentos.npy")
LABELS_PATH = Path("modelo/dataset_modelo_labels.parquet")
MODELO_DIR = Path("modelo")


def construir_modelos() -> dict:
    """Una fábrica por modelo (no una instancia compartida) para que cada
    fold entrene un clasificador nuevo, sin arrastrar estado entre folds."""
    return {
        "regresion_logistica": lambda: LogisticRegression(
            max_iter=2000, class_weight="balanced", C=1.0
        ),
        "svm_lineal": lambda: LinearSVC(
            class_weight="balanced", random_state=42, max_iter=5000, dual="auto"
        ),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }


def evaluar_modelo_kfold(X: np.ndarray, y: np.ndarray, clases: list[str],
                          fabrica_modelo, nombre_modelo: str,
                          folds: list[tuple[np.ndarray, np.ndarray]]) -> dict:
    accs, f1_macros, f1_weighteds = [], [], []
    reportes_por_fold = []
    tiempos = []

    for i, (train_idx, test_idx) in enumerate(folds, start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = fabrica_modelo()
        t0 = time.time()
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        tiempos.append(time.time() - t0)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        accs.append(acc)
        f1_macros.append(f1_macro)
        f1_weighteds.append(f1_weighted)

        rep = classification_report(
            y_test, y_pred, target_names=clases, zero_division=0,
            labels=range(len(clases)), output_dict=True,
        )
        reportes_por_fold.append(rep)

        print(
            f"  [{nombre_modelo}] Fold {i}/{len(folds)}: "
            f"exactitud={acc:.3f}  F1 macro={f1_macro:.3f}  "
            f"F1 ponderado={f1_weighted:.3f}  ({tiempos[-1]:.1f}s)"
        )

    accs, f1_macros, f1_weighteds = np.array(accs), np.array(f1_macros), np.array(f1_weighteds)

    f1_por_clase = {}
    for clase in clases:
        valores = [
            rep[clase]["f1-score"]
            for rep in reportes_por_fold
            if clase in rep and rep[clase]["support"] > 0
        ]
        f1_por_clase[clase] = round(float(np.mean(valores)), 4) if valores else None

    return {
        "modelo": nombre_modelo,
        "accuracy_media": round(float(accs.mean()), 4),
        "accuracy_std": round(float(accs.std()), 4),
        "f1_macro_media": round(float(f1_macros.mean()), 4),
        "f1_macro_std": round(float(f1_macros.std()), 4),
        "f1_weighted_media": round(float(f1_weighteds.mean()), 4),
        "f1_weighted_std": round(float(f1_weighteds.std()), 4),
        "tiempo_promedio_fit_predict_seg": round(float(np.mean(tiempos)), 2),
        "f1_por_clase": f1_por_clase,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="Número de folds deseado (default: 5)")
    args = parser.parse_args()

    X = np.load(EMBEDDINGS_PATH)
    labels_df = pd.read_parquet(LABELS_PATH)
    y_raw = labels_df["sentido"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    clases = list(le.classes_)

    print(f"Documentos: {X.shape[0]} | Dimensiones del embedding: {X.shape[1]}")
    print(f"Clases ({len(clases)}): {clases}\n")

    conteos = pd.Series(y_raw).value_counts()
    min_clase = int(conteos.min())
    k = min(args.k, min_clase)
    if k < args.k:
        print(
            f"AVISO: la clase con menos ejemplos tiene solo {min_clase} documento(s) "
            f"-- se reduce k de {args.k} a {k}.\n"
        )

    # Un solo conjunto de folds, reutilizado para los tres modelos --
    # así la comparación es sobre EXACTAMENTE las mismas particiones.
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    folds = list(skf.split(X, y))

    modelos = construir_modelos()
    resultados = []
    for nombre_modelo, fabrica in modelos.items():
        print(f"\n{'=' * 60}\nModelo: {nombre_modelo}\n{'=' * 60}")
        resultados.append(evaluar_modelo_kfold(X, y, clases, fabrica, nombre_modelo, folds))

    print(f"\n{'=' * 70}\nComparación final ({k}-fold, mismas particiones para los 3 modelos)\n{'=' * 70}")
    resumen_df = pd.DataFrame([
        {
            "modelo": r["modelo"],
            "accuracy": f"{r['accuracy_media']:.4f} ± {r['accuracy_std']:.4f}",
            "f1_macro": f"{r['f1_macro_media']:.4f} ± {r['f1_macro_std']:.4f}",
            "f1_weighted": f"{r['f1_weighted_media']:.4f} ± {r['f1_weighted_std']:.4f}",
            "tiempo_fit_predict_seg": r["tiempo_promedio_fit_predict_seg"],
        }
        for r in resultados
    ])
    print(resumen_df.to_string(index=False))

    print("\nF1 por clase (comparación lado a lado):")
    tabla_clases = pd.DataFrame({r["modelo"]: r["f1_por_clase"] for r in resultados})
    print(tabla_clases.to_string())

    salida = MODELO_DIR / "comparacion_modelos_clasificacion.json"
    salida.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado detalle completo en {salida}")


if __name__ == "__main__":
    main()
