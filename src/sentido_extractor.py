"""
Detección basada en reglas del "sentido" de una sentencia.

Este es un primer nivel de etiquetado, para para construir la variable objetivo del clasificador 
mediante expresiones regulares. 

Si en una misma sentencia se cruzan varias palabras clave por referencias históricas o menciones 
a otros juicios, para evitar ambigüedades: selecciona la decisión jurídica de mayor jerarquía.
El sistema evalúa el texto aplicando un estricto orden de jerarquía jurídica:
    revoca > modifica > confirma > desecha > sobresee > declara_improcedente > 
    declara_infundado > declara_fundado

Adicionalmente, el pipeline registra la evidencia contextual ('sentido_evidencia') para facilitar 
las métricas de control de calidad del corpus.    
"""

import re
from typing import Any


# Encabezado "RESUELVE" tolerante a OCR: letras espaciadas, un punto o dos puntos, y también posibles
# variaciones del prefijo "RES" (ej. "VUELVE", "SUELVE" en vez de "RESUELVE"). La expresión regular se
# construye de la siguiente manera:
# 1. [A-ZÁÉÍÓÚÑ]{0,3} : encuentra entre cero y tres letras mayúsculas. Ayuda a encontrar posibles
# variaciones como “RESUELVE”, “SUELVE” o “VUELVE”. 
# 2. \s* : permite que existan cero o más espacios entre las letras.
# 3. [:.]? : permite que después de la palabra aparezca opcionalmente un punto o dos puntos.
RESOLUTIVO_HEADER_PATTERN = re.compile(
    r"[A-ZÁÉÍÓÚÑ]{0,3}\s*U\s*E\s*L\s*V\s*E\s*[:.]?",
    re.IGNORECASE,
)

# En ausencia del encabezado "RESUELVE". Utiliza la frase de transición "por lo expuesto y fundado, se:"
FUNDADO_TRANSITION_PATTERN = re.compile(
    r"por\s+lo\s+(anteriormente\s+)?expuesto\s+y\s+fundado,?\s*se\s*[:.]?",
    re.IGNORECASE,
)

# Detecta documentos que en realidad son "Acuerdos Plenarios": Por alcance del proyecto, estos documentos
#  se excluyen del corpus de modelado.
ACUERDO_PLENARIO_FILENAME_PATTERN = re.compile(r"ACUERDO\s+PLENARIO", re.IGNORECASE)
ACUERDA_HEADER_PATTERN = re.compile(r"\bACUERDA\s*[:.]", re.IGNORECASE)

# Extrae la categoría del procedimiento(AP/JIN/JDCN/MII/PES/RV) directamente del nombre de archivo cuando el 
# metadato 'category' viene vacío.
CATEGORY_FROM_FILENAME_PATTERN = re.compile(
    r"TEE-(AP|JIN|JDCN|MII|PES|RV)-", re.IGNORECASE
)

# La palabra "agravio" no siempre aparece cerca de "fundado"/"infundado". Como patrón alternativo, se 
# busca la frase "declara(n)" seguido directamente de fundado/infundado. A continuación se descompone
# la expresión regular para explicar su definición:
# (se\s+)?declaran?\s+infundad[oa]s?\b --> Ej. "se declara infundado", "declaran infundadas", "Declara infundado"
# 1. (se\s+)? : busca se seguido de uno o más espacios, pero el ? hace que sea opcional.
# 2. declaran? : busca la palabra "declara" o "declaran", el ? hace que la n sea opcional.
# 3. \s+ : permite que existan uno o más espacios entre las letras.
# 4. infundad[oa]s? : busca la palabra "infundado" o "infundada" con [oa] permite que sea cualquiera 
# de las dos y la s? hace que singular o plural sea opcional.
# 5. \b : marca que la palabra termina ahí.
# |infundad[oa]s?\s+(el|los|las)?\s*agravio --> Ej. "infundadas las agravios", "infundado el agravio", "infundada agravios"
# 6. (el|los|las)? : permite que la palabra "el", "los" o "las" aparezca opcionalmente después de "infundado/infundada".
# 7. \s* : permite que existan cero o más espacios entre las letras.
DECLARA_INFUNDADO_PATTERN = re.compile(
    r"(se\s+)?declaran?\s+infundad[oa]s?\b|infundad[oa]s?\s+(el|los|las)?\s*agravio",
    re.IGNORECASE,
)

# (?<![a-záéíóúñ]) : evita que "fundado" haga match dentro de otra palabra.
DECLARA_FUNDADO_PATTERN = re.compile(
    r"(se\s+)?declaran?\s+fundad[oa]s?\b|(?<![a-záéíóúñ])fundad[oa]s?\s+(el|los|las)?\s*agravio",
    re.IGNORECASE,
)

# Orden de prioridad: si aparece más de un sentido, se reporta el de mayor jerarquía (ej. una revocación
# o modificación es más informativa que una simple confirmación cuando ambas palabras aparecen).
# (se\s+)? : hace que el prefijo "se " sea opcional porque el resolutivo a veces inicia directamente con la palabra resolutiva.
SENTIDO_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("revoca", re.compile(r"(se\s+)?revoca", re.IGNORECASE)),
    ("modifica", re.compile(r"(se\s+)?modif[a-z=]*", re.IGNORECASE)),
    ("confirma", re.compile(r"(se\s+)?conf[i=]rma", re.IGNORECASE)),
    ("desecha", re.compile(r"(se\s+)?desecha", re.IGNORECASE)),
    ("sobresee", re.compile(r"(se\s+)?sobresee", re.IGNORECASE)),
    ("declara_improcedente", re.compile(r"improcedente", re.IGNORECASE)),
    ("declara_infundado", DECLARA_INFUNDADO_PATTERN),
    ("declara_fundado", DECLARA_FUNDADO_PATTERN),
]

# Patrones para incidentes de incumplimiento (evalúan estado de cumplimiento).
# (?<!in) : evita que "cumplida" haga match dentro de la palabra "incumplida".
INCUMPLIDA_PATTERN = re.compile(r"incumplid[oa]s?", re.IGNORECASE)
PARCIALMENTE_CUMPLIDA_PATTERN = re.compile(r"parcialmente\s+(fundad[oa]|cumplid[oa]s?)", re.IGNORECASE)
CUMPLIDA_PATTERN = re.compile(r"(?<!in)cumplid[oa]s?", re.IGNORECASE)
SIN_MATERIA_PATTERN = re.compile(r"(ha\s+)?quedado\s+sin\s+materia", re.IGNORECASE)


INCUMPLIMIENTO_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("incumplida", INCUMPLIDA_PATTERN),
    ("parcialmente_cumplida", PARCIALMENTE_CUMPLIDA_PATTERN),
    ("cumplida", CUMPLIDA_PATTERN),
    ("queda_sin_materia", SIN_MATERIA_PATTERN),
]

# Patrones para Procedimientos Especiales Sancionadores (PES): evalúan existencia de infracciones. 
# Ej. El resolutivo + "se" (Opcional) + "declara"/"acreditó"/"acredita" + "la" + "existencia"/"inexistencia"
#  + "de" +"la"/"las" + "infracción"/"violación" (singular o plural).
INEXISTENCIA_INFRACCION_PATTERN = re.compile(
    r"(se\s+)?(declara|acredit[oó])\s+la\s+inexistencia\s+de\s+(la|las)\s+(infracci[oó]n(es)?|violaci[oó]n(es)?)",
    re.IGNORECASE,
)
EXISTENCIA_INFRACCION_PATTERN = re.compile(
    r"(se\s+)?(declara|acredit[oó])\s+la\s+existencia\s+de\s+(la|las)\s+(infracci[oó]n(es)?|violaci[oó]n(es)?)",
    re.IGNORECASE,
)


PES_SANCIONADOR_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("inexistencia_infraccion", INEXISTENCIA_INFRACCION_PATTERN),
    ("existencia_infraccion", EXISTENCIA_INFRACCION_PATTERN),
    ("sobresee", re.compile(r"(se\s+)?sobresee", re.IGNORECASE)),
] + SENTIDO_PATTERNS  # fallback: algunos PES resuelven con lenguaje general.

# Mapeo de patrones por tipología de expediente.
PATTERN_SETS: dict[str, list[tuple[str, re.Pattern]]] = {
    "general": SENTIDO_PATTERNS,
    "incumplimiento": INCUMPLIMIENTO_PATTERNS,
    "pes_sancionador": PES_SANCIONADOR_PATTERNS,
}

# Términos puramente procesales que NO deben confundirse con un sentido final.
PROCEDURAL_ONLY_PATTERN = re.compile(
    r"^\s*(se\s+acumulan|agréguense|reg[ií]strese|fórmese|túrnese)",
    re.IGNORECASE,
)


def fix_common_ocr_artifacts(text: str) -> str:
    """
    Resuelve el error tipográfico donde la ligadura "fi"  es interpretada incorrectamente como un 
    signo de igual "=" (ej. "con=rma" -> "confirma").
    """

    return re.sub(r"(?<=[a-záéíóúñ])=(?=[a-záéíóúñ])", "fi", text, flags=re.IGNORECASE)


def extract_resolutivo_section(text: str) -> str | None:
    """
    Devuelve el texto desde el encabezado RESUELVE hasta el final del documento.

    Si aparece más de una vez (por ejemplo, en el índice y en el cuerpo), se usa la última aparición,
    que normalmente corresponde al resolutivo real.
    """

    matches = list(RESOLUTIVO_HEADER_PATTERN.finditer(text)) # Devuelve todas las coincidencias del patrón RESOLUTIVO_HEADER_PATTERN en el texto.

    if matches:
        last_match = matches[-1] # Selecciona la última coincidencia.
        return text[last_match.end():]

    # En ausencia del encabezado "RESUELVE". Utiliza la frase de transición "por lo expuesto y fundado, se:"
    fallback_matches = list(FUNDADO_TRANSITION_PATTERN.finditer(text))

    if fallback_matches:
        last_match = fallback_matches[-1] # Selecciona la última coincidencia.
        return text[last_match.end():]

    return None


def resolve_category(filename: str, category: str | None = None) -> str | None:
    """
    Devuelve la categoría del procedimiento (AP/JIN/JDCN/MII/PES/RV), usando el metadato ("category") 
    si existe y, si viene vacío, extrayéndola directamente del nombre de archivo.
    """

    if category:
        return category 

    match = CATEGORY_FROM_FILENAME_PATTERN.search(filename)
    return match.group(1).upper() if match else None


def is_acuerdo_plenario(filename: str, text: str | None = None) -> bool:
    """
    Determina si el documento corresponde a un Acuerdo Plenario y no a una sentencia de fondo.
    
    Devuelve True si el documento es un Acuerdo Plenario. Condiciones:
    1. El nombre del archivo contiene la frase "ACUERDO PLENARIO" (ej. ("ACUERDO PLENARIO TEE-AP-...")
    2. El documento contiene el encabezado "ACUERDA:" y nunca aparece un encabezado tipo RESUELVE.
    3. False en cualquier otro caso.
    """

    # Detecta si el nombre de archivo contiene la frase "ACUERDO PLENARIO".
    if ACUERDO_PLENARIO_FILENAME_PATTERN.search(filename): 
        return True

    # Detecta si el documento contiene el encabezado "ACUERDA:" y nunca aparece un encabezado tipo RESUELVE.
    if text and ACUERDA_HEADER_PATTERN.search(text) and not RESOLUTIVO_HEADER_PATTERN.search(text):
        return True

    return False


def infer_procedure_category(filename: str, category: str | None = None) -> str:
    """
    Infiere la categoría de procedimiento a partir del nombre de archivo
    y/o la categoría del expediente (AP/JIN/JDCN/MII/PES/RV), para elegir el
    conjunto de patrones correcto en detect_sentido().
    """

    upper_name = filename.upper()
    resolved_category = resolve_category(filename, category)

    if "INCIDENT" in upper_name or "EXCITATIVA" in upper_name:
        return "incumplimiento"

    if resolved_category is not None and resolved_category.upper() == "PES":
        return "pes_sancionador"

    return "general"


def detect_sentido(text: str, procedure_category: str = "general") -> dict[str, Any]:
    """
    Detecta el sentido de la resolución a partir del texto completo del documento.

        text: texto completo extraído del documento (OCR o nativo).
        procedure_category: 'general', 'incumplimiento' o 'pes_sancionador'.
            
            Usar infer_procedure_category() para derivar la categoría de procedimiento del documento
            antes de llamar a esta función. **


    Procesa el texto completo, extrae la sección correspondiente al "RESUELVE" y aplica expresiones 
    regulares según la categoría del procedimiento para identificar el sentido de la resolución de 
    forma jerárquica.

    Args:
        text (str): Texto completo del documento.
        procedure_category (str, optional): Categoría del procedimiento para seleccionar el conjunto 
        de patrones adecuado. Las opciones estándar son 'general', 'incumplimiento' o 'pes_sancionador'.
        Por defecto es "general". 

            ** Usar infer_procedure_category() para derivar la categoría de procedimiento del documento
            antes de llamar a esta función. **

    Returns:
        dict[str, Any]: Diccionario con el resultado del análisis estructurado con las siguientes claves:
            - sentido (str): Etiqueta del sentido detectado o "no_detectado".
            - sentido_metodo (str): Método o motivo del resultado (ej. "regex_resuelve", "sin_seccion_resuelve",
              "sin_coincidencias").
            - sentido_evidencia (str | None): Fragmento de texto extraído como soporte o evidencia.
            - sentidos_detectados (list[str], opcional): Listado completo de todas las coincidencias de sentido
              encontradas, ordenadas por prioridad.
    """

    cleaned_text = fix_common_ocr_artifacts(text) # Limpia el texto.
    resolutivo_section = extract_resolutivo_section(cleaned_text) # Extrae la sección del resolutivo desde el 
    # encabezado RESUELVE hasta el final del documento.

    # Si no se encuentra la sección resolutiva, se devuelve un diccionario indicando que no se detectó sentido.
    if resolutivo_section is None:
        return {
            "sentido": "no_detectado",
            "sentido_metodo": "sin_seccion_resuelve",
            "sentido_evidencia": None,
        }

    if not resolutivo_section.strip():
        return {
            "sentido": "no_detectado",
            "sentido_metodo": "sin_coincidencias",
            "sentido_evidencia": None,
        }

    patterns = PATTERN_SETS.get(procedure_category, SENTIDO_PATTERNS) # Determina el conjunto de patrones a usar según
    # la categoría de procedimiento (procedure_category). Si no se encuentra la categoría, se usa el conjunto de patrones 
    # por defecto (SENTIDO_PATTERNS).

    found: list[tuple[str, str]] = []

    for label, pattern in patterns:
        match = pattern.search(resolutivo_section) # Busca la primera coincidencia del patrón en la sección resolutiva.

        # Sí se encuentra una coincidencia:
        if match:
            start = max(match.start() - 60, 0) # Se toma un fragmento de texto de 60 caracteres antes.
            end = min(match.end() + 60, len(resolutivo_section)) # Se toma un fragmento de texto de 60 caracteres después.
            snippet = resolutivo_section[start:end].replace("\n", " ").strip() #Limpia saltos de linea, unifica el texto en una sola línea.
            found.append((label, snippet)) # Agrega la etiqueta y el fragmento de texto a la lista.

    # Sí la sección resolutiva existe pero no hizo match con las expresiones regulares, devuelve: "no_detectado" y:
    if not found:
        return {
            "sentido": "no_detectado",
            "sentido_metodo": "sin_coincidencias",
            "sentido_evidencia": resolutivo_section[:200].replace("\n", " ").strip(), # Fragmento de los primeros 200 caracteres de la sección resolutiva como evidencia.
        }

    # El orden de las listas de patrones ya refleja la prioridad; nos
    # quedamos con la primera etiqueta que haya encontrado coincidencia.
    priority_labels = [label for label, _ in patterns]
    found_labels = {label: snippet for label, snippet in found}

    # Recorre la lista de etiquetas en orden de prioridad y devuelve la primera coincidencia encontrada junto con su evidencia.
    for label in priority_labels:
        if label in found_labels:
            return {
                "sentido": label,
                "sentido_metodo": "regex_resuelve", # Indicador de detección por expresiones regulares.
                "sentido_evidencia": found_labels[label], # Fragmento de texto que contiene el sentido detectado.
                "sentidos_detectados": [label for label, _ in found], # Lista de todos los sentidos detectados,
            }

    return {
        "sentido": "no_detectado",
        "sentido_metodo": "sin_coincidencias",
        "sentido_evidencia": None,
    }
