"""Ajusta un vectorizador TF-IDF sobre el corpus de entrenamiento y guarda
la matriz TF-IDF resultante -- esto reemplaza los embeddings de
sentence-transformers como base de la búsqueda de "precedentes similares"
(NO afecta al clasificador de sentido, que sigue usando los embeddings de
generar_embeddings.py).

Por qué se cambia de espacio vectorial solo para precedentes
--------------------------------------------------------------
El embedding de todo el documento (Antecedentes + Consideraciones) es una
buena señal para CLASIFICAR el sentido, pero es una mala señal para medir
"qué tan parecido es el ASUNTO de dos sentencias": todas las sentencias de
este tribunal comparten muchísimo lenguaje de formato/procedimiento
("de las constancias que obran en el expediente...", "en términos del
artículo...", encabezados de trámite, etc.), así que dos documentos con
asuntos completamente distintos (una queja por dietas y una por violencia
política de género, por ejemplo) terminan con embeddings casi idénticos
(similitud 0.95-1.00) solo por compartir ese lenguaje de plantilla. Esto se
comprobó con evidencia real: en la demo, "sentencias parecidas" mostraba
casos con 95-100% de similitud cuyo asunto de fondo no tenía nada que ver.

Se probó extraer solo la sección de "HECHOS" con reglas (regex sobre
encabezados HECHOS/ANTECEDENTES/RESULTANDO) para aislar la narrativa del
caso, pero falló la validación (ver diagnóstico interno): esos encabezados
suelen capturar trámite procesal (notificaciones, fechas, oficios), no el
resumen del asunto -- la similitud contra el "hechos" real generado por
Gemini fue muy baja (0.59 promedio) y el acierto de categoría del vecino
más cercano empeoró (30/40 con categoría distinta, peor que el documento
completo).

TF-IDF sí funciona porque hace lo contrario de un embedding semántico denso:
en vez de "entender" el texto, sencillamente le baja peso automáticamente a
las palabras que aparecen en casi todos los documentos (justo el lenguaje
de plantilla) y le sube peso a las palabras que son distintivas de cada
caso (nombres de partes, montos, conceptos concretos del asunto: "dietas",
"espectaculares", "violencia política de género", etc.), sin necesitar
ningún modelo de lenguaje ni encabezados de sección. Es simétrico por
construcción: el mismo vectorizador se aplica igual a un documento del
corpus que a un documento nuevo jamás visto.

Validación (muestra de 40 documentos reales, categoría como proxy de
"mismo asunto" -- ver diagnóstico interno, no incluido en este paquete por
tamaño):
    - Documento completo + embedding semántico (actual):  18/40 vecino con categoría distinta
    - Sección "HECHOS" por regex + embedding semántico:    30/40  (PEOR -- descartado)
    - Hechos reales (resumen Gemini) + embedding semántico: 14/40  (mejor posible, pero requiere Gemini por documento -- no viable)
    - Documento completo + TF-IDF (este script):           16/40  (mejora clara, sin LLM, simétrico)

Uso:
    python scripts/generar_precedentes_tfidf.py

Requiere haber corrido antes build_dataset_modelo.py (data/dataset_modelo.parquet).
Genera en modelo/:
    - tfidf_vectorizer.joblib   (vectorizador ajustado, reutilizable para documentos nuevos)
    - tfidf_matriz_corpus.npz   (matriz TF-IDF dispersa del corpus, alineada por fila con
                                  dataset_modelo_labels.parquet / embeddings_documentos.npy)
"""
from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dataset_modelo.parquet"
MODELO_DIR = BASE_DIR / "modelo"

# Mismos parámetros validados en el diagnóstico. min_df=2 descarta palabras
# que aparecen en un solo documento (ruido de OCR, nombres propios únicos
# que no ayudan a encontrar vecinos); ngram_range=(1,2) captura conceptos de
# dos palabras ("violencia política", "pago de dietas"); sublinear_tf
# amortigua el efecto de palabras repetidas muchas veces en un documento
# largo.
TFIDF_PARAMS = dict(
    max_features=8000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True,
)


def main() -> None:
    df = pd.read_parquet(DATA_PATH)
    if "texto_insumo" not in df.columns:
        raise ValueError(
            f"'{DATA_PATH}' no tiene columna 'texto_insumo'. Corre antes "
            "build_dataset_modelo.py."
        )

    vectorizador = TfidfVectorizer(**TFIDF_PARAMS)
    matriz = vectorizador.fit_transform(df["texto_insumo"].fillna(""))

    MODELO_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizador, MODELO_DIR / "tfidf_vectorizer.joblib")
    sparse.save_npz(MODELO_DIR / "tfidf_matriz_corpus.npz", matriz)

    print(f"Documentos: {matriz.shape[0]} | Vocabulario: {matriz.shape[1]}")
    print(f"Guardado: {MODELO_DIR / 'tfidf_vectorizer.joblib'}")
    print(f"Guardado: {MODELO_DIR / 'tfidf_matriz_corpus.npz'}")
    print(
        "\nIMPORTANTE: esta matriz debe quedar alineada fila-a-fila con "
        "dataset_modelo_labels.parquet / embeddings_documentos.npy -- si "
        "vuelves a correr build_dataset_modelo.py y cambia el orden o el "
        "número de filas, vuelve a correr este script también."
    )


if __name__ == "__main__":
    main()
