"""Enriquecimiento de precedentes con link al documento original en
trieen.mx y resumen automático (hechos / agravios / resolución).

Los precedentes que devuelve `explicar_prediccion.precedentes_relevantes()`
solo incluyen expediente, sentido, categoría y similitud -- lo mínimo que
necesita el modelo. Este módulo agrega la capa de presentación (link +
resumen) sin tocar la lógica de similitud, usando dos fuentes que ya
existen en el repo y no requieren ninguna llamada nueva a ninguna API:

  - data/dataset_sentencias.csv  -> nombre de archivo original y
    categoría de cada documento, para reconstruir la URL en trieen.mx
    con el mismo formato que usa scripts/download_trieen.py.
  - data/resumenes.json          -> resumen de las 873 sentencias del
    corpus (hechos / agravios / resolución), generado por
    scripts/generar_resumenes.py.

Uso:
    from precedentes_datos import enriquecer_precedentes
    precedentes = enriquecer_precedentes(precedentes_relevantes(...)["a_favor"])
"""

import json
from pathlib import Path
from urllib.parse import quote

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_SENTENCIAS_CSV = DATA_DIR / "dataset_sentencias.csv"
RESUMENES_JSON = DATA_DIR / "resumenes.json"

# Mismo template que scripts/download_trieen.py -- si ese script cambia
# de convención algún día, hay que replicar el cambio aquí también.
BASE_FILES_URL = "https://trieen.mx/wp-content/uploads/filestrieen"

_metadata_por_json_filename: dict[str, dict] | None = None
_resumenes_por_json_filename: dict[str, dict] | None = None


def _limpio(valor):
    """Normaliza valores que vienen de pandas (NaN, cadenas vacías) a
    None, para no confundirlos con un valor real más adelante -- ojo:
    `float('nan')` es "truthy" en Python, así que un `if not valor` NO
    detecta un NaN por sí solo."""
    if valor is None:
        return None
    if isinstance(valor, float) and pd.isna(valor):
        return None
    if isinstance(valor, str) and not valor.strip():
        return None
    return valor


def _cargar_metadata() -> dict[str, dict]:
    """Carga source_file/category por json_filename desde el inventario
    (data/dataset_sentencias.csv). Se cachea en memoria porque no cambia
    durante una sesión de Streamlit."""
    global _metadata_por_json_filename
    if _metadata_por_json_filename is None:
        if DATASET_SENTENCIAS_CSV.exists():
            df = pd.read_csv(DATASET_SENTENCIAS_CSV)
            _metadata_por_json_filename = {
                row["json_filename"]: {
                    "source_file": row.get("source_file"),
                    "category": row.get("category"),
                }
                for _, row in df.iterrows()
            }
        else:
            _metadata_por_json_filename = {}
    return _metadata_por_json_filename


def _cargar_resumenes() -> dict[str, dict]:
    """Carga los resúmenes ya generados, indexados por json_filename.
    Los documentos con error de generación (campo 'error' presente) se
    omiten -- no hay resumen confiable que mostrar para ellos."""
    global _resumenes_por_json_filename
    if _resumenes_por_json_filename is None:
        if RESUMENES_JSON.exists():
            datos = json.loads(RESUMENES_JSON.read_text(encoding="utf-8"))
            _resumenes_por_json_filename = {
                r["json_filename"]: r for r in datos if "error" not in r
            }
        else:
            _resumenes_por_json_filename = {}
    return _resumenes_por_json_filename


def url_documento_original(category, source_file) -> str | None:
    """Reconstruye la URL del documento original en trieen.mx. Devuelve
    None si falta la categoría o el nombre de archivo (p. ej. documentos
    antiguos sin ese metadato)."""
    category = _limpio(category)
    source_file = _limpio(source_file)
    if category is None or source_file is None:
        return None
    return f"{BASE_FILES_URL}/{category}/{quote(source_file)}"


def enriquecer_precedentes(precedentes: list[dict]) -> list[dict]:
    """Devuelve una copia de la lista de precedentes, agregando a cada
    uno:
        - url_original: link al documento en trieen.mx (o None si no se
          pudo reconstruir).
        - resumen: dict con hechos/agravios/resolucion (o None si el
          documento no tiene resumen generado o tuvo error al generarlo).
    No modifica los campos originales (expediente, sentido, similitud).
    """
    metadata = _cargar_metadata()
    resumenes = _cargar_resumenes()

    enriquecidos = []
    for precedente in precedentes:
        jf = precedente.get("json_filename")
        meta = metadata.get(jf, {})

        category = _limpio(precedente.get("category")) or _limpio(meta.get("category"))
        source_file = _limpio(meta.get("source_file"))

        enriquecido = dict(precedente)
        enriquecido["url_original"] = url_documento_original(category, source_file)
        enriquecido["resumen"] = resumenes.get(jf)
        enriquecidos.append(enriquecido)

    return enriquecidos
