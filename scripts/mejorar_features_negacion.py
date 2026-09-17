"""Prueba si agregar señales léxicas explícitas (frases clave y negaciones)
a los embeddings semánticos mejora la clasificación, sobre todo para los
pares que el diagnóstico de matriz de confusión mostró más confundidos:

  - existencia_infraccion <-> inexistencia_infraccion (negación pura)
  - confirma / revoca / modifica (triángulo de mayor volumen de error)
  - desecha / sobresee / declara_improcedente (familia de "no llega al fondo")

Por qué: los embeddings semánticos (mpnet) capturan bien el tema general
del texto, pero son conocidos por representar mal las negaciones --
"se acredita la existencia de la infracción" y "no se acredita la
existencia de la infracción" quedan muy cerca en el espacio vectorial
aunque signifiquen lo contrario. Este script añade ~14 features léxicas
explícitas (conteo normalizado de frases clave, tomadas literalmente de
las definiciones jurídicas ya usadas en el prompt de Gemini) y compara,
con la MISMA validación cruzada que ya usas, si el modelo mejora al
tener también esa señal.

Requiere, además de lo que ya usan evaluar_kfold.py y
diagnostico_confusion.py:
    data/dataset_modelo.parquet   (columna 'texto_insumo' y 'json_filename';
    es la misma tabla que generar_embeddings.py usa como fuente -- si la
    borraste, tenés que volver a correr build_dataset_modelo.py primero)

Uso:
    python scripts/mejorar_features_negacion.py [--k 5] [--escenario completo|sin_aproximado|ambos]

No reentrena el modelo oficial ni toca modelo/*.joblib -- es puramente
diagnóstico/experimental. Guarda:
    modelo/comparacion_features_negacion.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

EMBEDDINGS_PATH = Path("modelo/embeddings_documentos.npy")
LABELS_PATH = Path("modelo/dataset_modelo_labels.parquet")
DATASET_MODELO_PATH = Path("data/dataset_modelo.parquet")
MODELO_DIR = Path("modelo")

ESCENARIOS = {
    "completo": None,
    "sin_aproximado": "aproximado_gemini",
}

# Frases clave tomadas literalmente de definiciones_sentido_teen.md (las
# mismas definiciones jurídicas ya incorporadas al prompt de Gemini).
# Cada grupo cuenta cuántas veces aparece cualquiera de sus variantes,
# normalizado por longitud del documento -- le da al modelo una señal
# léxica explícita que complementa (no reemplaza) al embedding semántico.
GRUPOS_KEYWORDS = {
    "kw_confirma": [r"se confirma", r"agravios? (?:resultan?|son) infundados?", r"no se modifica"],
    "kw_revoca": [r"se revoca", r"queda sin efecto", r"sin efectos? jur[ií]dicos"],
    "kw_modifica": [r"se modifica", r"parcialmente fundado", r"parcialmente la raz[oó]n"],
    "kw_desecha": [r"se desecha", r"no se admite a tr[aá]mite", r"por extempor[aá]neo", r"se tiene por no presentado"],
    "kw_sobresee": [r"se sobresee", r"qued[oó] sin materia", r"desistimiento"],
    "kw_improcedente": [r"declara improcedente", r"es improcedente", r"carece de inter[eé]s jur[ií]dico", r"falta de competencia"],
    "kw_fundado": [r"se declara fundado", r"resulta fundado"],
    "kw_infundado": [r"se declara infundado", r"resulta infundado"],
    "kw_cumplida": [r"se tiene por cumplida", r"acat[oó] (?:en su totalidad|totalmente)"],
    "kw_incumplida": [r"se declara(?: el)? incumplimiento", r"no acat[oó]"],
    "kw_parcial_cumplida": [r"cumplimiento parcial", r"parcialmente cumplida"],
    "kw_existencia_infraccion": [r"se acredita la existencia", r"existencia de la infracci[oó]n", r"se impone (?:una )?sanci[oó]n"],
    "kw_inexistencia_infraccion": [r"inexistencia de la infracci[oó]n", r"no se acredita", r"no qued[oó] acreditad[oa]", r"no se prob[oó]"],
    "kw_negacion_general": [r"\bno\b", r"\bsin\b", r"\bcarece\b"],
}

# Pares/familias que el diagnóstico anterior mostró más confundidos --
# se revisa su confusión específica antes y después de agregar las features.
PARES_A_VIGILAR = [
    ("existencia_infraccion", "inexistencia_infraccion"),
    ("confirma", "revoca"),
    ("confirma", "modifica"),
    ("revoca", "modifica"),
    ("desecha", "sobresee"),
    ("desecha", "declara_improcedente"),
]


def contar_ocurrencias(texto_lower: str, patrones: list[str]) -> int:
    return sum(len(re.findall(p, texto_lower)) for p in patrones)


def construir_features_keywords(textos: list[str]) -> pd.DataFrame:
    filas = []
    for texto in textos:
        texto = texto or ""
        texto_lower = texto.lower()
        n_palabras = max(len(texto.split()), 1)
        fila = {
            nombre: 1000 * contar_ocurrencias(texto_lower, patrones) / n_palabras
            for nombre, patrones in GRUPOS_KEYWORDS.items()
        }
        filas.append(fila)
    return pd.DataFrame(filas)


def evaluar_con_cv(X: np.ndarray, y: np.ndarray, k: int) -> tuple[np.ndarray, dict]:
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)),
    ])
    y_pred = cross_val_predict(pipeline, X, y, cv=skf)
    metrics = {
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "f1_macro": round(float(f1_score(y, y_pred, average="macro")), 4),
        "f1_weighted": round(float(f1_score(y, y_pred, average="weighted")), 4),
    }
    return y_pred, metrics


def confusiones_de_pares(y_true_raw: np.ndarray, y_pred_raw: np.ndarray, pares: list[tuple[str, str]]) -> dict:
    resultado = {}
    for a, b in pares:
        a_como_b = int(((y_true_raw == a) & (y_pred_raw == b)).sum())
        b_como_a = int(((y_true_raw == b) & (y_pred_raw == a)).sum())
        total_a = int((y_true_raw == a).sum())
        total_b = int((y_true_raw == b).sum())
        resultado[f"{a}<->{b}"] = {
            f"{a}_predicho_como_{b}": a_como_b,
            f"{b}_predicho_como_{a}": b_como_a,
            f"pct_{a}_mal_hacia_{b}": round(100 * a_como_b / total_a, 1) if total_a else None,
            f"pct_{b}_mal_hacia_{a}": round(100 * b_como_a / total_b, 1) if total_b else None,
        }
    return resultado


def evaluar_escenario(X_emb: np.ndarray, X_kw: pd.DataFrame, y_raw: np.ndarray, nombre_escenario: str, k_deseado: int) -> dict:
    print(f"\n{'=' * 70}\nEscenario: {nombre_escenario}\n{'=' * 70}")
    print(f"Documentos: {X_emb.shape[0]}")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    conteos = pd.Series(y_raw).value_counts()
    min_clase = int(conteos.min())
    k = min(k_deseado, min_clase)
    if k < k_deseado:
        print(f"AVISO: se reduce k de {k_deseado} a {k} por la clase más chica ({min_clase} docs).")
    if k < 2:
        print(f"ERROR: no se puede evaluar '{nombre_escenario}'.")
        return {"escenario": nombre_escenario, "error": "clase con menos de 2 ejemplos"}

    print("\n-- Solo embeddings (línea base) --")
    y_pred_base, metrics_base = evaluar_con_cv(X_emb, y, k)
    print(f"  Exactitud={metrics_base['accuracy']}  F1 macro={metrics_base['f1_macro']}  F1 ponderado={metrics_base['f1_weighted']}")

    X_hibrido = np.hstack([X_emb, X_kw.to_numpy()])
    print("\n-- Embeddings + features de palabras clave/negación --")
    y_pred_hib, metrics_hib = evaluar_con_cv(X_hibrido, y, k)
    print(f"  Exactitud={metrics_hib['accuracy']}  F1 macro={metrics_hib['f1_macro']}  F1 ponderado={metrics_hib['f1_weighted']}")

    delta = {
        "accuracy": round(metrics_hib["accuracy"] - metrics_base["accuracy"], 4),
        "f1_macro": round(metrics_hib["f1_macro"] - metrics_base["f1_macro"], 4),
        "f1_weighted": round(metrics_hib["f1_weighted"] - metrics_base["f1_weighted"], 4),
    }
    print(f"\nDiferencia (híbrido - línea base): {delta}")

    y_pred_base_raw = le.inverse_transform(y_pred_base)
    y_pred_hib_raw = le.inverse_transform(y_pred_hib)

    pares_disponibles = [
        (a, b) for a, b in PARES_A_VIGILAR
        if a in set(y_raw) and b in set(y_raw)
    ]
    conf_base = confusiones_de_pares(y_raw, y_pred_base_raw, pares_disponibles)
    conf_hib = confusiones_de_pares(y_raw, y_pred_hib_raw, pares_disponibles)

    print("\nConfusiones en los pares vigilados (línea base -> híbrido):")
    for par in conf_base:
        print(f"  {par}:")
        print(f"    antes:   {conf_base[par]}")
        print(f"    después: {conf_hib[par]}")

    return {
        "escenario": nombre_escenario,
        "n_documentos": int(X_emb.shape[0]),
        "k": k,
        "solo_embeddings": metrics_base,
        "embeddings_mas_keywords": metrics_hib,
        "diferencia": delta,
        "confusiones_pares_antes": conf_base,
        "confusiones_pares_despues": conf_hib,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--escenario", choices=["completo", "sin_aproximado", "ambos"], default="ambos")
    args = parser.parse_args()

    X_full = np.load(EMBEDDINGS_PATH)
    labels_df = pd.read_parquet(LABELS_PATH)

    if "metodo_corte" not in labels_df.columns:
        raise SystemExit("ERROR: falta la columna 'metodo_corte' en dataset_modelo_labels.parquet.")
    if "json_filename" not in labels_df.columns:
        raise SystemExit("ERROR: falta la columna 'json_filename' en dataset_modelo_labels.parquet.")

    if not DATASET_MODELO_PATH.exists():
        raise SystemExit(
            f"ERROR: no se encontró {DATASET_MODELO_PATH}. Este script necesita el texto "
            "original de cada documento (columna 'texto_insumo') para construir las features "
            "de palabras clave. Si lo borraste, vuelve a correr scripts/build_dataset_modelo.py."
        )

    texto_df = pd.read_parquet(DATASET_MODELO_PATH, columns=["json_filename", "texto_insumo"])

    # Alinea el texto con las filas de labels_df/embeddings por json_filename
    # (no por posición) para evitar cualquier desfase entre corridas.
    merge_df = labels_df[["json_filename"]].merge(texto_df, on="json_filename", how="left")
    if merge_df["texto_insumo"].isna().any():
        faltantes = int(merge_df["texto_insumo"].isna().sum())
        print(
            f"AVISO: {faltantes} documento(s) de dataset_modelo_labels.parquet no se "
            f"encontraron en {DATASET_MODELO_PATH} por json_filename -- se les asigna "
            "texto vacío (sus features de palabras clave quedarán en 0)."
        )
    textos = merge_df["texto_insumo"].fillna("").tolist()

    print("Construyendo features de palabras clave/negación...")
    X_kw_full = construir_features_keywords(textos)
    print(f"Features construidas: {list(X_kw_full.columns)}\n")

    escenarios_a_correr = ESCENARIOS if args.escenario == "ambos" else {args.escenario: ESCENARIOS[args.escenario]}

    resultados = []
    for nombre, metodo_excluir in escenarios_a_correr.items():
        if metodo_excluir is None:
            mask = np.ones(len(labels_df), dtype=bool)
        else:
            mask = (labels_df["metodo_corte"] != metodo_excluir).to_numpy()

        X_emb = X_full[mask]
        X_kw = X_kw_full.loc[mask].reset_index(drop=True)
        y_raw = labels_df.loc[mask, "sentido"].to_numpy()
        resultados.append(evaluar_escenario(X_emb, X_kw, y_raw, nombre, args.k))

    print(f"\n{'=' * 70}\nResumen comparativo\n{'=' * 70}")
    for r in resultados:
        if "error" in r:
            continue
        print(
            f"{r['escenario']:16s} base: acc={r['solo_embeddings']['accuracy']} "
            f"f1_macro={r['solo_embeddings']['f1_macro']}  |  "
            f"híbrido: acc={r['embeddings_mas_keywords']['accuracy']} "
            f"f1_macro={r['embeddings_mas_keywords']['f1_macro']}  |  "
            f"delta acc={r['diferencia']['accuracy']:+.4f} delta f1_macro={r['diferencia']['f1_macro']:+.4f}"
        )

    out_path = MODELO_DIR / "comparacion_features_negacion.json"
    out_path.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado detalle completo en {out_path}")


if __name__ == "__main__":
    main()
