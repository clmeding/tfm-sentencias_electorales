"""Extracción directa de texto de PDF y detección de OCR."""

from pathlib import Path
from typing import Any

import pymupdf # Libreria para trabajar con archivos PDF.

from cleaner import clean_text, count_useful_characters # Funciones auxiliares para limpiar el texto y contar caracteres útiles.


MINIMUM_CHARACTERS_PER_PAGE = 100 # Define el umbral mínimo de caracteres útiles por página para considerar que la página tiene suficiente texto.
OCR_PAGE_RATIO_THRESHOLD = 0.60 # Define el umbral de proporción de páginas con poco texto para marcar el PDF como pendiente de OCR.


def extract_pdf(file_path: Path) -> dict[str, Any]:
    """
    Extrae el texto embebido de un PDF.

    Si al menos el 60 % de las páginas contiene menos de
    100 caracteres útiles, el PDF se marca como pendiente de OCR.
    """

    pages: list[dict[str, Any]] = [] # Lista vacia para almacenar la información de cada página del PDF.
    pages_with_little_text = 0 # Contador de páginas con poco texto.

    with pymupdf.open(file_path) as document: # Abre el archivo PDF usando pymupdf.
        total_pages = len(document) # Obtiene el número total de páginas del PDF.

        # Itera sobre cada página del PDF, obteniendo el número de página y el objeto de la página.
        for page_number, page in enumerate( 
            document,
            start=1,
        ):
            raw_text = page.get_text(
                "text",
                sort=True,
            )

            cleaned_text = clean_text(raw_text) # Limpia el texto crudo, convierte todo a minúsculas, elimina espacios en blanco y ruido visual 
            # (Signos de puntuación, comas, puntos, etc.) y devuelve el texto limpio.

            # Cuenta los caracteres útiles en el texto limpio.
            useful_characters = count_useful_characters(
                cleaned_text
            ) 

            # Compara si los caracteres útiles son menores al umbral mínimo, si es así incrementa el contador de páginas con poco texto.
            if useful_characters < MINIMUM_CHARACTERS_PER_PAGE:
                pages_with_little_text += 1

            pages.append(
                {
                    "page": page_number,
                    "raw_text": raw_text,
                    "text": cleaned_text,
                    "useful_characters": useful_characters,
                }
            )

    # Calcula la proporción de páginas deficientes con respecto al total de páginas del PDF.
    low_text_ratio = (
        pages_with_little_text / total_pages
        if total_pages > 0 else 1.0 # Si el total de páginas es cero, se asigna un valor de 1.0 para evitar división por cero.
    )

    # Determina si el PDF necesita OCR basado en la proporción de páginas con poco texto.
    needs_ocr = (
        low_text_ratio >= OCR_PAGE_RATIO_THRESHOLD
    )

    return {
        "extraction_method": "pymupdf",
        "status": (
            "needs_ocr"
            if needs_ocr
            else "completed"
        ),
        "needs_ocr": needs_ocr,
        "page_count": total_pages,
        "pages_with_little_text": pages_with_little_text,
        "low_text_ratio": round(low_text_ratio, 4),
        "units": pages,
    }