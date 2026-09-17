"""Valida el clasificador de sentido con validación cruzada estratificada
(k-fold) en vez de una sola partición 80/20.

Por qué: con un corpus de unos pocos cientos de documentos y varias clases
con menos de 10 ejemplos, una sola partición train/test es muy sensible a
qué documentos exactos caen del lado de prueba -- cambiar el tamaño del
corpus en unas pocas decenas de filas puede mover la exactitud varios
puntos aunque el modelo no haya cambiado. k-fold promedia varias
particiones distintas del mismo dataset, dando una estimación mucho más
estable (y su desviación estándar entre folds) para reportar en la tesis.

Este script NO reentrena el modelo "oficial" ni toca modelo/*.joblib --
es puramente diagnóstico. Usa los mismos embeddings y la misma columna
metodo_corte que scripts/comparar_escenarios_corte.py, así que debe correr
después de:
    python scripts/build_dataset_modelo.py
    python scripts/generar_embeddings.py   (o generar_embeddings_gemini.py,
    cualquiera que ya se haya usado para producir
    modelo/embeddings_documentos.npy y modelo/dataset_modelo_labels.parquet)

Uso:
    python scripts/evaluar_kfold.py [--k 5]

Para cada escenario (completo / sin_aproximado) hace k particiones
estratificadas, entrena y evalúa en cada una, y reporta la media ± la
desviación estándar de exactitud, F1 macro y F1 ponderado entre los k
folds, más el F1 promedio por clase. Guarda el detalle en
modelo/comparacion_kfold.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

EMBEDDINGS_PATH = Path("modelo/embeddings_documentos.npy")
LABELS_PATH = Path("modelo/dataset_modelo_labels.parquet")
MODELO_DIR = Path("modelo")

# nombre_escenario -> valor de metodo_corte a excluir (None = no excluir nada)
ESCENARIOS = {
    "completo": None,
    "sin_aproximado": "aproximado_gemini",
}


def evaluar_escenario_kfold(X: np.ndarray, y_raw: np.ndarray, nombre_escenario: str, k_deseado: int) -> dict:
    print(f"\n{'=' * 60}\nEscenario: {nombre_escenario}\n{'=' * 60}")
    print(f"Documentos: {X.shape[0]} | Dimensiones del embedding: {X.shape[1]}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"Clases ({len(le.classes_)}): {list(le.classes_)}\n")

    conteos = pd.Series(y_raw).value_counts()
    print("Documentos por clase en este escenario:")
    print(conteos.to_string())

    min_clase = int(conteos.min())
    k = min(k_deseado, min_clase)
    if k < k_deseado:
        print(
            f"\nAVISO: la clase con menos ejemplos tiene solo {min_clase} "
            f"documento(s) -- se reduce k de {k_deseado} a {k} para poder "
            f"estratificar sin dejar ningún fold sin representación de esa clase."
        )
    if k < 2:
        print(
            f"\nERROR: no se puede hacer validación cruzada en el escenario "
            f"'{nombre_escenario}' -- alguna clase tiene menos de 2 ejemplos."
        )
        return {
            "escenario": nombre_escenario,
            "n_documentos": int(X.shape[0]),
            "k": k,
            "error": "clase con menos de 2 ejemplos, no se pudo evaluar",
        }

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    accs, f1_macros, f1_weighteds = [], [], []
    reportes_por_fold = []

    for i, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        accs.append(acc)
        f1_macros.append(f1_macro)
        f1_weighteds.append(f1_weighted)

        rep = classification_report(
            y_test, y_pred, target_names=le.classes_, zero_division=0,
            labels=range(len(le.classes_)), output_dict=True,
        )
        reportes_por_fold.append(rep)

        print(f"  Fold {i}/{k}: exactitud={acc:.3f}  F1 macro={f1_macro:.3f}  F1 ponderado={f1_weighted:.3f}")

    accs, f1_macros, f1_weighteds = np.array(accs), np.array(f1_macros), np.array(f1_weighteds)

    print(f"\nPromedio de {k} folds (media ± desv. estándar):")
    print(f"  Exactitud:    {accs.mean():.3f} ± {accs.std():.3f}")
    print(f"  F1 macro:     {f1_macros.mean():.3f} ± {f1_macros.std():.3f}")
    print(f"  F1 ponderado: {f1_weighteds.mean():.3f} ± {f1_weighteds.std():.3f}")

    # Promedia el F1 de cada clase entre los folds donde apareció en el test
    print("\nF1 promedio por clase (entre los folds donde la clase tuvo ejemplos de prueba):")
    f1_por_clase = {}
    for clase in le.classes_:
        valores = [
            rep[clase]["f1-score"]
            for rep in reportes_por_fold
            if clase in rep and rep[clase]["support"] > 0
        ]
        if valores:
            f1_por_clase[clase] = {"media": float(np.mean(valores)), "n_folds": len(valores)}
            print(f"  {clase:28s} {np.mean(valores):.3f}  (presente en {len(valores)}/{k} folds)")
        else:
            f1_por_clase[clase] = {"media": None, "n_folds": 0}
            print(f"  {clase:28s} sin ejemplos de prueba en ningún fold")

    return {
        "escenario": nombre_escenario,
        "n_documentos": int(X.shape[0]),
        "k": k,
        "accuracy_media": round(float(accs.mean()), 4),
        "accuracy_std": round(float(accs.std()), 4),
        "f1_macro_media": round(float(f1_macros.mean()), 4),
        "f1_macro_std": round(float(f1_macros.std()), 4),
        "f1_weighted_media": round(float(f1_weighteds.mean()), 4),
        "f1_weighted_std": round(float(f1_weighteds.std()), 4),
        "f1_por_clase": f1_por_clase,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5, help="Número de folds deseado (default: 5)")
    args = parser.parse_args()

    X_full = np.load(EMBEDDINGS_PATH)
    labels_df = pd.read_parquet(LABELS_PATH)

    if "metodo_corte" not in labels_df.columns:
        raise SystemExit(
            "ERROR: modelo/dataset_modelo_labels.parquet no tiene la columna "
            "'metodo_corte'. Corra primero scripts/generar_embeddings.py "
            "(o la versión de Gemini) para regenerarla."
        )

    resultados = []
    for nombre, metodo_excluir in ESCENARIOS.items():
        if metodo_excluir is None:
            mask = np.ones(len(labels_df), dtype=bool)
        else:
            mask = (labels_df["metodo_corte"] != metodo_excluir).to_numpy()

        X = X_full[mask]
        y_raw = labels_df.loc[mask, "sentido"].to_numpy()
        resultados.append(evaluar_escenario_kfold(X, y_raw, nombre, args.k))

    print(f"\n{'=' * 60}\nComparación final (k-fold)\n{'=' * 60}")
    resumen_df = pd.DataFrame([
        {
            "escenario": r["escenario"],
            "n_documentos": r["n_documentos"],
            "k": r.get("k"),
            "accuracy": f"{r.get('accuracy_media', float('nan')):.4f} ± {r.get('accuracy_std', float('nan')):.4f}" if "accuracy_media" in r else "error",
            "f1_macro": f"{r.get('f1_macro_media', float('nan')):.4f} ± {r.get('f1_macro_std', float('nan')):.4f}" if "f1_macro_media" in r else "error",
            "f1_weighted": f"{r.get('f1_weighted_media', float('nan')):.4f} ± {r.get('f1_weighted_std', float('nan')):.4f}" if "f1_weighted_media" in r else "error",
        }
        for r in resultados
    ])
    print(resumen_df.to_string(index=False))

    resumen_path = MODELO_DIR / "comparacion_kfold.json"
    resumen_path.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado detalle completo en {resumen_path}")


if __name__ == "__main__":
    main()
