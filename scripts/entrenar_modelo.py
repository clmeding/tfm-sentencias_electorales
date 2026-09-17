"""Entrena y evalúa el clasificador de sentido sobre los embeddings ya
generados (scripts/generar_embeddings.py debe correrse antes que este).

Primero evalúa con una partición entrenamiento/prueba estratificada
(80/20) para medir el desempeño real del modelo, y después reentrena
con el 100% de los datos para dejar el modelo final listo para usarse
en predicciones nuevas (src/explicar_prediccion.py lo carga desde aquí).

Uso:
    python scripts/entrenar_modelo.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

EMBEDDINGS_PATH = Path("modelo/embeddings_documentos.npy")
LABELS_PATH = Path("modelo/dataset_modelo_labels.parquet")
MODELO_DIR = Path("modelo")


def main() -> None:
    X = np.load(EMBEDDINGS_PATH)
    labels_df = pd.read_parquet(LABELS_PATH)
    y_raw = labels_df["sentido"].values

    print(f"Documentos: {X.shape[0]} | Dimensiones del embedding: {X.shape[1]}")
    print(f"Clases: {sorted(set(y_raw))}\n")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    # --- Evaluación honesta con partición train/test ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    scaler_eval = StandardScaler()
    X_train_scaled = scaler_eval.fit_transform(X_train)
    X_test_scaled = scaler_eval.transform(X_test)

    clf_eval = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf_eval.fit(X_train_scaled, y_train)
    y_pred = clf_eval.predict(X_test_scaled)

    print(f"\nExactitud (accuracy): {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1 macro: {f1_score(y_test, y_pred, average='macro'):.3f}")
    print(f"F1 ponderado: {f1_score(y_test, y_pred, average='weighted'):.3f}\n")
    print("Reporte de clasificación por clase:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))
    print(
        "NOTA: las clases con muy pocos ejemplos en todo el corpus "
        "(cumplida, incumplida, declara_improcedente, parcialmente_cumplida "
        "-- todas con menos de 10 casos) tienen métricas poco confiables "
        "por falta de datos, no por un problema del modelo."
    )

    # --- Modelo final: reentrenado con el 100% de los datos disponibles ---
    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X)
    clf_final = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf_final.fit(X_scaled_final, y)

    MODELO_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_final, MODELO_DIR / "clasificador.joblib")
    joblib.dump(scaler_final, MODELO_DIR / "scaler.joblib")
    joblib.dump(le, MODELO_DIR / "label_encoder.joblib")
    print(f"\nModelo final (entrenado con los {len(y)} documentos etiquetados) guardado en {MODELO_DIR}/")


if __name__ == "__main__":
    main()
