"""Compara el desempeño del clasificador de sentido en dos escenarios:

  A) "completo": incluye TODOS los documentos del dataset de modelado,
     incluidos los que se cortaron con el método "aproximado_gemini"
     (corte conservador del último 15% del texto, usado cuando no se
     encontró el marcador exacto del resolutivo que dio Gemini).
  B) "sin_aproximado": excluye esos documentos, quedándose solo con los
     que tienen un punto de corte confiable (encabezado RESUELVE, la
     transición "por lo fundado y motivado", o el marcador verbatim de
     Gemini encontrado tal cual en el texto).

Sirve como prueba de sensibilidad para el TFM: si el desempeño es
parecido en ambos escenarios, el corte aproximado no está introduciendo
ruido significativo y se puede reportar el modelo completo con
confianza. Si cae mucho en el escenario "completo" (o sube mucho al
quitar esos documentos), es señal de que sí está introduciendo ruido -- 
o de que el escenario B tiene una fuga que hace que la tarea sea artificialmente 
más fácil.

Requiere que ya se haya corrido, en este orden:
    python scripts/build_dataset_modelo.py
    python scripts/generar_embeddings_gemini.py   (versión que guarda
    la columna metodo_corte en modelo/dataset_modelo_labels.parquet)

Uso:
    python scripts/comparar_escenarios_corte.py

Para cada escenario:
  - Evalúa con partición 80/20 estratificada (mismos parámetros que
    scripts/entrenar_modelo.py, para que la comparación sea justa).
  - Reentrena con el 100% de los datos de ese escenario y guarda el
    modelo final en modelo/<escenario>/.
Al final guarda un resumen comparativo en modelo/comparacion_escenarios.json.

Nota: este script NO decide cuál escenario usar en producción -- eso lo
decide usted según los resultados, para reportarlo en la tesis. El
modelo "oficial" que carga src/explicar_prediccion.py sigue siendo el
que genera scripts/entrenar_modelo.py (en modelo/clasificador.joblib);
si decide adoptar el escenario "sin_aproximado" como definitivo, copie
los archivos de modelo/sin_aproximado/ a modelo/ (instrucciones abajo).
"""
from pathlib import Path
import json

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

# nombre_escenario -> valor de metodo_corte a excluir (None = no excluir nada)
ESCENARIOS = {
    "completo": None,
    "sin_aproximado": "aproximado_gemini",
}


def evaluar_escenario(X: np.ndarray, y_raw: np.ndarray, nombre_escenario: str) -> dict:
    print(f"\n{'=' * 60}\nEscenario: {nombre_escenario}\n{'=' * 60}")
    print(f"Documentos: {X.shape[0]} | Dimensiones del embedding: {X.shape[1]}")
    print(f"Clases: {sorted(set(y_raw))}\n")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    conteos = pd.Series(y_raw).value_counts()
    clases_escasas = conteos[conteos < 2].index.tolist()
    stratify_arg = y
    if clases_escasas:
        print(
            f"AVISO: clase(s) con menos de 2 ejemplos en este escenario "
            f"{clases_escasas} -- no se puede estratificar la partición, "
            f"se hace una partición aleatoria simple en su lugar."
        )
        stratify_arg = None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_arg
        )
    except ValueError as e:
        print(f"AVISO: no se pudo estratificar ({e}); usando partición aleatoria simple.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )

    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    scaler_eval = StandardScaler()
    X_train_scaled = scaler_eval.fit_transform(X_train)
    X_test_scaled = scaler_eval.transform(X_test)

    clf_eval = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf_eval.fit(X_train_scaled, y_train)
    y_pred = clf_eval.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"\nExactitud (accuracy): {acc:.3f}")
    print(f"F1 macro: {f1_macro:.3f}")
    print(f"F1 ponderado: {f1_weighted:.3f}\n")
    print("Reporte de clasificación por clase:")
    print(
        classification_report(
            y_test, y_pred, target_names=le.classes_, zero_division=0, labels=range(len(le.classes_))
        )
    )

    # Reentrena con el 100% de los datos de este escenario y guarda el modelo final
    scaler_final = StandardScaler()
    X_scaled_final = scaler_final.fit_transform(X)
    clf_final = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf_final.fit(X_scaled_final, y)

    out_dir = MODELO_DIR / nombre_escenario
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf_final, out_dir / "clasificador.joblib")
    joblib.dump(scaler_final, out_dir / "scaler.joblib")
    joblib.dump(le, out_dir / "label_encoder.joblib")
    print(f"Modelo final de este escenario guardado en {out_dir}/")

    return {
        "escenario": nombre_escenario,
        "n_documentos": int(X.shape[0]),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "accuracy": round(float(acc), 4),
        "f1_macro": round(float(f1_macro), 4),
        "f1_weighted": round(float(f1_weighted), 4),
    }


def main() -> None:
    X_full = np.load(EMBEDDINGS_PATH)
    labels_df = pd.read_parquet(LABELS_PATH)

    if "metodo_corte" not in labels_df.columns:
        raise SystemExit(
            "ERROR: modelo/dataset_modelo_labels.parquet no tiene la columna "
            "'metodo_corte'. Vuelva a correr scripts/generar_embeddings_gemini.py "
            "con la versión actualizada que guarda esa columna (borre antes "
            "modelo/dataset_modelo_labels.parquet y modelo/embeddings_documentos.npy "
            "para forzar que se regeneren con la columna nueva; no hace falta "
            "volver a llamar a la API, data/embeddings_gemini_raw.json ya tiene "
            "los vectores calculados)."
        )

    resultados = []
    for nombre, metodo_excluir in ESCENARIOS.items():
        if metodo_excluir is None:
            mask = np.ones(len(labels_df), dtype=bool)
        else:
            mask = (labels_df["metodo_corte"] != metodo_excluir).to_numpy()

        X = X_full[mask]
        y_raw = labels_df.loc[mask, "sentido"].to_numpy()
        resultados.append(evaluar_escenario(X, y_raw, nombre))

    print(f"\n{'=' * 60}\nComparación final\n{'=' * 60}")
    comp_df = pd.DataFrame(resultados)
    print(comp_df.to_string(index=False))

    resumen_path = MODELO_DIR / "comparacion_escenarios.json"
    resumen_path.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado resumen en {resumen_path}")
    print(
        "\nPara adoptar un escenario como el modelo oficial (el que usa "
        "src/explicar_prediccion.py), copie sus archivos a la raíz de modelo/, "
        "por ejemplo:\n"
        "  cp modelo/sin_aproximado/*.joblib modelo/"
    )


if __name__ == "__main__":
    main()
