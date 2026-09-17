"""Carga (Load) los outputs ya generados del pipeline de Extracción y
Transformación en una base de datos SQLite única, integrando las
distintas fuentes por `json_filename`.

Este script es la etapa "L" (Loading) del ETL: no vuelve a extraer ni
transformar nada, toma los archivos planos que ya produce el pipeline
(dataset_sentencias.csv, resumenes.json, temas_por_documento.csv) y los
integra en `data/corpus.db` para que puedan consultarse con SQL en vez
de solo como archivos sueltos.

Uso:
    python scripts/cargar_sqlite.py

Requiere: pip install pandas (ya está en requirements.txt)
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "corpus.db"

DATASET_CSV = DATA_DIR / "dataset_sentencias.csv"
RESUMENES_JSON = DATA_DIR / "resumenes.json"
TEMAS_CSV = DATA_DIR / "temas_por_documento.csv"


def cargar_documentos(conn: sqlite3.Connection) -> int:
    """Tabla principal: metadatos e inventario de cada documento."""
    df = pd.read_csv(DATASET_CSV)
    df.to_sql("documentos", conn, if_exists="replace", index=False)
    return len(df)


def cargar_resumenes(conn: sqlite3.Connection) -> int:
    """Tabla de resúmenes (hechos / agravios / resolución) por documento."""
    if not RESUMENES_JSON.exists():
        print(f"  Aviso: no existe {RESUMENES_JSON}, se omite la tabla de resúmenes.")
        return 0

    registros = json.loads(RESUMENES_JSON.read_text(encoding="utf-8"))
    df = pd.DataFrame(registros)
    df.to_sql("resumenes", conn, if_exists="replace", index=False)
    return len(df)


def cargar_temas(conn: sqlite3.Connection) -> int:
    """Tabla de temas detectados por documento."""
    if not TEMAS_CSV.exists():
        print(f"  Aviso: no existe {TEMAS_CSV}, se omite la tabla de temas.")
        return 0

    df = pd.read_csv(TEMAS_CSV)
    df.to_sql("temas", conn, if_exists="replace", index=False)
    return len(df)


def crear_indices(conn: sqlite3.Connection) -> None:
    """Índices sobre la llave de unión, para que los JOIN sean rápidos."""
    cur = conn.cursor()
    for tabla in ("documentos", "resumenes", "temas"):
        try:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{tabla}_json_filename "
                f"ON {tabla} (json_filename)"
            )
        except sqlite3.OperationalError:
            pass  # la tabla no existe (no se cargó esa fuente)
    conn.commit()


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    print(f"Cargando en {DB_PATH} ...\n")
    n_docs = cargar_documentos(conn)
    print(f"  documentos: {n_docs} filas")
    n_resumenes = cargar_resumenes(conn)
    print(f"  resumenes:  {n_resumenes} filas")
    n_temas = cargar_temas(conn)
    print(f"  temas:      {n_temas} filas")

    crear_indices(conn)
    conn.close()

    print("\nListo. Ejemplo de consulta integrada:")
    print(
        "  SELECT d.expediente, d.sentido, r.resolucion, t.temas\n"
        "  FROM documentos d\n"
        "  LEFT JOIN resumenes r ON d.json_filename = r.json_filename\n"
        "  LEFT JOIN temas t ON d.json_filename = t.json_filename\n"
        "  WHERE d.sentido != 'no_detectado';"
    )


if __name__ == "__main__":
    main()
