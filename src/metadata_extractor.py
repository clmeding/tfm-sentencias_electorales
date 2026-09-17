"""Extracción de metadatos estructurales a partir de la ruta y el nombre del archivo.

Estos metadatos (categoría, expediente, año, fecha, tipo de documento) como base para 
las siguientes etapas del pipeline: enlazar cada sentencia con su "sentido" (confirma/
revoca/modifica/sobresee) y alimentar el clasificador de aprendizaje automático.
"""

import re
from pathlib import Path
from typing import Any


# Ejemplos que debe capturar: TEE-JDCN-46-2024, TEE-PES-33-2024, TEE-AP-01-2025, SC-E-JDCN-55-2014, IEEN-CLE-RRV-005-2024
# (1). (?:TEE|SC-E|IEEN) : Siglas institucionales, que pueden ser TEE, SC-E o IEEN.
# (2). [-\s]? : Puede haber un guion o un espacio después de (1) y (3).
# (3). [A-Z]{2,6} : Siglas del tipo de procedimiento, que pueden ser de 2 a 6 letras mayúsculas (por ejemplo, JDCN, PES).
# (4). \d{1,4} : Número de expediente, que puede tener entre 1 y 4 dígitos.
# (5). [-/] : Puede haber un guion o una diagonal (/) después de (4).
# (6). \d{4} : Año del expediente, que debe tener exactamente 4 dígitos.
EXPEDIENTE_PATTERN = re.compile(
    r"(?:TEE|SC-E|IEEN)[-\s]?[A-Z]{2,6}[-\s]?\d{1,4}[-/]\d{4}",
    re.IGNORECASE,
)

# Fechas en formato dd-mm-aaaa dentro del nombre del archivo.
DATE_PATTERN = re.compile(r"(\d{2})-(\d{2})-(\d{4})")

# Año de 4 dígitos como respaldo cuando no hay expediente reconocible.
YEAR_PATTERN = re.compile(r"(19|20)\d{2}")

# Normarliza la escritura de términos similares para evitar perder información en la etapa de clasificación de documentos.
DOCUMENT_TYPE_KEYWORDS = {
    "acuerdo plenario": "acuerdo_plenario",
    "incidente": "incidente",
    "incidental": "incidente",
    "sentencia": "sentencia",
    "resolucion": "resolucion",
    "resolución": "resolucion",
    "version publica": "version_publica",
    "versión pública": "version_publica",
}


def extract_metadata(
    file_path: Path,
    input_root: Path,
) -> dict[str, Any]:
    """Deriva categoría, expediente, año, fecha y tipo de documento."""

    relative_parts = file_path.relative_to(input_root).parts

    # Obtiene la categoría temática extrayendo la carpeta contenedora 
    # de la ruta relativa (ej. documents/input/<CATEGORIA>/archivo.pdf --> <CATEGORIA>/archivo.pdf).
    category = relative_parts[0] if len(relative_parts) > 1 else None # Si la categoría esta en una subcarpeta, toma el nombre 
    # de la carpeta. Sí no hay carpeta, asigna None.

    filename = file_path.stem # Extrar el nombre del archivo sin la extensión.
    filename_lower = filename.lower() # Cambia el texto a minúsculas.

    expediente_match = EXPEDIENTE_PATTERN.search(filename)
    expediente = (
        expediente_match.group(0).upper().replace(" ", "-") # Convierte el expediente a mayúsculas y reemplaza espacios por guiones.
        if expediente_match
        else None
    )

    date_match = DATE_PATTERN.search(filename)
    document_date = (
        f"{date_match.group(3)}-{date_match.group(2)}-{date_match.group(1)}"
        if date_match
        else None
    )

    year_match = YEAR_PATTERN.search(filename)
    year = (
        document_date[:4]
        if document_date
        else (year_match.group(0) if year_match else None)
    )

    document_type = next(
        (
            value
            for keyword, value in DOCUMENT_TYPE_KEYWORDS.items()
            if keyword in filename_lower
        ),
        "no_identificado",
    )

    has_acumulados = "acumulad" in filename_lower # Verifica si el nombre del archivo contiene la palabra "acumulad" (indicando que es un documento que agrupa varias resoluciones).

    return {
        "category": category,
        "expediente": expediente,
        "year": year,
        "document_date": document_date,
        "document_type": document_type,
        "has_acumulados": has_acumulados,
    }
