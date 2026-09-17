"""Funciones para normalizar y evaluar el texto extraído."""

import re
import unicodedata


# Localiza el final del membrete institucional al inicio del documento.
# Como el OCR suele corromper el texto del escudo, se usa la palabra "NAYARIT"
# como referencia fija y confiable para recortar y eliminar esa basura inicial.
LETTERHEAD_MARKER_PATTERN = re.compile(r"\bNAYARIT\b", re.IGNORECASE)

# Limita la búsqueda a los primeros 180 caracteres para evitar borrar 
# por error menciones legítimas de "Nayarit" más adelante en el documento 
# (como fechas o domicilios). Si no aparece, el texto no se modifica.
LETTERHEAD_MAX_SEARCH_CHARS = 180


def clean_text(text: str) -> str:
    """Realiza una limpieza conservadora del texto."""

    text = unicodedata.normalize("NFC", text) # Estandariza la representación de caracteres con tildes y eñes para que tengan el mismo código Unicode.
    text = text.replace("\u00ad", "") # Elimina guiones suaves (soft hyphens) que pueden aparecer en el texto OCR y que no son visibles pero afectan la lectura.
    text = re.sub(r"[ \t]+", " ", text) # Reemplaza espacios múltiples o tabulaciones(Tab) por un solo espacio.
    text = re.sub(r" *\n *", "\n", text) # Elimina espacios al inicio y final de cada línea, dejando solo un salto de línea.
    text = re.sub(r"\n{3,}", "\n\n", text) # Reemplaza saltos de línea múltiples (3 o más) por solo dos, para evitar párrafos excesivamente separados.

    return text.strip()


def strip_scanned_letterhead(text: str) -> dict:
    """
    Elimina el membrete institucional corrupto por el OCR al inicio del documento.

    Retorna un diccionario con:
        - text: El texto resultante (recortado o intacto).
        - letterhead_removed: Indicador booleano (True/False) para auditoría.
        - removed_prefix: El fragmento exacto que se eliminó para revisión rápida.
    """

    match = LETTERHEAD_MARKER_PATTERN.search(
        text[:LETTERHEAD_MAX_SEARCH_CHARS]
    )

    if match is None:
        return {
            "text": text,
            "letterhead_removed": False,
            "removed_prefix": None,
        }

    removed_prefix = text[:match.end()] # Corta y elimina el fragmento del escudo y membrete institucional incluyendo la palabra "NAYARIT".
    remaining_text = text[match.end():].lstrip() # Devuelve el resto del texto junto con un indicador de que se eliminó el membrete.

    return {
        "text": remaining_text,
        "letterhead_removed": True,
        "removed_prefix": removed_prefix,
    }


def count_useful_characters(text: str) -> int:
    """Cuenta las letras y los números presentes en el texto."""

    return sum(character.isalnum() for character in text)
