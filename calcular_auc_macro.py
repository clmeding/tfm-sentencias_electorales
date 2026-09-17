"""Calcula el AUC macro (One-vs-Rest) del clasificador de sentido, usando
exactamente el mismo split de validación cruzada (k=5, random_state=42) que
scripts/evaluar_kfold.py, escenario "completo".

No reentrena el modelo oficial ni toca modelo/*.joblib -- es puramente
diagnóstico, igual que evaluar_kfold.py.

Uso (correr desde la raíz del repo, después de generar embeddings):
    python calcular_auc_macro.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, StandardScaler

EMBEDDINGS_PATH = Path("modelo/embeddings_documentos.npy")
LABELS_PATH = Path("modelo/dataset_modelo_labels.parquet")
K = 5


def main() -> None:
    X = np.load(EMBEDDINGS_PATH)
    labels_df = pd.read_parquet(LABELS_PATH)
    y_raw = labels_df["sentido"].to_numpy()

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    lb = LabelBinarizer()
    lb.fit(y)

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)

    aucs = []
    for i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(X_train_scaled, y_train)
        y_proba = clf.predict_proba(X_test_scaled)

        y_test_bin = lb.transform(y_test)
        # Algunas clases pueden faltar en el fold de prueba; roc_auc_score
        # con average="macro" y multi_class="ovr" requiere columnas
        # consistentes, así que se filtran las clases sin ningún positivo
        # en este fold de prueba.
        clases_presentes = y_test_bin.sum(axis=0) > 0
        auc = roc_auc_score(
            y_test_bin[:, clases_presentes],
            y_proba[:, clases_presentes],
            average="macro",
            multi_class="ovr",
        )
        aucs.append(auc)
        print(f"  Fold {i}/{K}: AUC macro={auc:.3f}")

    aucs = np.array(aucs)
    print(f"\nPromedio de {K} folds (media ± desv. estándar):")
    print(f"  AUC macro: {aucs.mean():.3f} ± {aucs.std():.3f}")


if __name__ == "__main__":
    main()
