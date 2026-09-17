"""Extracción de texto de documentos DOCX."""

from pathlib import Path
from typing import Any

from docx import Document

from cleaner import clean_text


def extract_docx(file_path: Path) -> dict[str, Any]:
    """Extrae los párrafos con contenido de un documento DOCX."""

    document = Document(file_path) # Carga el documento DOCX usando la biblioteca python-docx
    paragraphs: list[dict[str, Any]] = [] # Lista vacía para almacenar la información de cada párrafo del documento.

    # Itera sobre cada párrafo del documento, obteniendo el número de párrafo y el objeto de párrafo.
    for paragraph_number, paragraph in enumerate(
        document.paragraphs,
        start=1,
    ):
        raw_text = paragraph.text
        cleaned_text = clean_text(raw_text) # Limpia el texto crudo, convierte todo a minúsculas, elimina espacios en blanco y ruido visual 
        # (Signos de puntuación, comas, puntos, etc.) y devuelve el texto limpio.

        # Si el texto limpio está vacío, se omite el párrafo y se continúa con el siguiente.
        if not cleaned_text:
            continue

        paragraphs.append(
            {
                "paragraph": paragraph_number,
                "style": (
                    paragraph.style.name # Guarda el nombre del estilo del párrafo si existe, ayuda a identificar la estructura del documento, como títulos, subtítulos, etc.
                    if paragraph.style
                    else None
                ),
                "raw_text": raw_text,
                "text": cleaned_text,
            }
        )

    return {
        "extraction_method": "python-docx",
        "status": "completed",
        "needs_ocr": False,
        "page_count": None,
        "units": paragraphs,
    }