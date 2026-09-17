"""
Detección basada en reglas de los temas de una sentencia.

Etiqueta cada documento con el/los tema(s) que trata (un documento puede tratar de varias materias 
a la vez). Los temas califican sí: 
(a) la suma de menciones de sus patrones alcanza UMBRAL_MENCIONES 
(b) coincide con un patrón de alta precisión con una sola mención
(c) aparece en el resumen (hechos/agravios/resolución) de la sentencia generado por una API de un 
modelo de lenguaje. --> una sola mención es suficiente.

Los documentos que no alcanzan el umbral para ningún tema conocido se clasifican como "Otro / no identificado".

UMBRAL_MENCIONES : Número mínimo de menciones requeridas para validar un tema. Evita falso positivos por menciones aisladas.
PATRONES_TEMA : Los 10 temas más frecuentes en el corpus de sentencias.
PATRONES_TEMA_ADICIONALES : 7 temas adicionales de menos frecuencia.
PATRONES_TEMA_TODOS = PATRONES_TEMA + PATRONES_TEMA_ADICIONALES : Todos los temas detectables en una sola lista. 
PATRONES_ALTA_PRECISION : Frases suficientemente específicas como para que una sola mención sea 
suficiente para dar por detectado el tema. Ej. "convenio de coalición"
"""

import re

UMBRAL_MENCIONES = 3

# Diccionario principal de expresiones regulares para identificar un tema.
PATRONES_TEMA = {
    "Violencia política de género": [
        r"violencia\s+pol[ií]tica",
        r"raz[oó]n\s+de\s+g[eé]nero",
        r"violencia\s+de\s+g[eé]nero",
    ],
    "Actos anticipados de campaña": [
        r"actos?\s+anticipados?\s+de\s+campa[ñn]a",
        r"precampa[ñn]a",
    ],
    "Fiscalización / Recursos públicos": [
        r"fiscalizaci[oó]n",
        r"recursos?\s+p[uú]blicos?",
        r"rebase\s+de\s+gastos?",
        r"topes?\s+de\s+gastos?",
    ],
    "Propaganda gubernamental": [
        r"propaganda\s+gubernamental",
        r"promoci[oó]n\s+personalizada",
        r"informe\s+de\s+labores",
    ],
    "Vida interna de partidos políticos": [
        r"vida interna del?\s+partido",
        r"estatutos del partido",
        r"militante",
        r"afiliaci[oó]n partidista",
        r"comisi[oó]n\s+(nacional\s+)?de\s+honestidad\s+y\s+justicia",
    ],
    "Financiamiento y prerrogativas": [
        r"prerrogativas",
        r"financiamiento p[uú]blico",
        r"financiamiento privado",
    ],
    "Registro y sustitución de candidaturas": [
        r"registro de candidatura",
        r"sustituci[oó]n de candidat",
        r"cancelaci[oó]n del registro de",
        r"candidatura com[uú]n",
    ],
    "Paridad de género": [
        r"paridad\s+de\s+g[eé]nero",
        r"paridad\s+en\s+la\s+postulaci[oó]n",
    ],
    "Nulidad de elección": [
        r"nulidad\s+de\s+la\s+elecci[oó]n",
        r"c[oó]mputo\s+(municipal|distrital)",
    ],
    "Usos y costumbres indígenas": [
        r"usos y costumbres",
        r"sistema normativo ind[ií]gena",
        r"consulta ind[ií]gena",
        r"pueblos?\s+ind[ií]genas?",
    ],
}

TEMAS = list(PATRONES_TEMA.keys())

# "Otro / no identificado" : Engloba los documentos que no coinciden con ninguno del los 17 temas 
# (PATRONES_TEMA Y PATRONES_TEMA_ADICIONALES).
# "Otros temas identificados" : Almacena temas que si identifica algunos de los 7 temas adicionales
# ó es un tema libre identificado por recomendación de la API de Gemini pero son muchos para fines 
# de visualización en el panel de KPIs. 
# Todos los temas detectados se agregan a la lista de temas detectables, pero no tienen su propia barra en el gráfico.
OTRO = "Otro / no identificado"
TEMAS_CON_OTRO = TEMAS + [OTRO]

OTROS_ESPECIFICOS = "Otros temas identificados"
TEMAS_PANEL_CON_BUCKETS = TEMAS + [OTROS_ESPECIFICOS, OTRO]


# Cubre temas que tienen un grado de frecuencia suficiente para ser detectables, pero su frecuencia
# se queda fuera de las 10 más altas.
PATRONES_TEMA_ADICIONALES = {
    "Desistimiento": [
        r"desistimiento",
    ],
    "Integración de órganos electorales": [
        r"integraci[oó]n del?\s+consejo",
        r"integraci[oó]n del?\s+ayuntamiento",
        r"toma\s+de\s+protesta",
        r"rendir\s+(la\s+)?protesta",
        r"rinda\s+(la\s+)?protesta",
        r"tomar\s+protesta",
    ],
    "Acción afirmativa": [
        r"acci[oó]n afirmativa",
    ],
    "Debates y encuestas electorales": [
        r"debate entre candidat",
        r"encuesta electoral",
        r"encuesta de preferencia",
    ],
    "Reelección": [
        r"reelecci[oó]n",
    ],
    "Credencial para votar": [
        r"credencial para votar",
    ],
    "Recuento y nulidad de casilla": [
        r"recuento de votos",
        r"nulidad de la? votaci[oó]n recibida en casilla",
        r"paquete electoral",
    ],
    "Coalición electoral": [
        r"convenio\s+de\s+coalici[oó]n",
        r"coalici[oó]n\s+electoral",
    ],
    "Remuneraciones y dietas de cargos públicos": [
        r"remuneraci[oó]n(es)?",
        r"dietas?\b",
    ],
    "Derecho a ser votado / ejercicio del cargo": [
        r"derecho\s+(de|a)\s+ser\s+votad[oa]",
        r"voto\s+pasivo",
        r"derecho\s+pol[ií]tico[- ]electoral\s+de\s+ser\s+votad",
    ],
}

# Todos los temas detectables para la etiqueta de cada sentencia individual, en una sola lista. 
PATRONES_TEMA_TODOS = {**PATRONES_TEMA, **PATRONES_TEMA_ADICIONALES}

# Frases suficientemente específicas como para que una sola mención sea suficiente para dar por 
# detectado el tema (no debe cubrir la condición de UMBRAL_MENCIONES).
PATRONES_ALTA_PRECISION = {
    r"actos?\s+anticipados?\s+de\s+campa[ñn]a",
    r"convenio\s+de\s+coalici[oó]n",
    r"comisi[oó]n\s+(nacional\s+)?de\s+honestidad\s+y\s+justicia",
}


def _tema_detectado(
    patrones: list[str], low: str, umbral: int, low_resumen: str = ""
) -> bool:
    """
    Un tema se considera detectado si:


      (a) Si el documento cuenta con un resumen de hechos/agravios/resolución ya generado, 
          al menos una coincidencia es suficiente.
    
    ** re.findall cuenta cuantas veces se detecta una expresión regular.
      (b) Sí pertenece a PATRONES_ALTA_PRECISION y aparece al menos una vez.
      (c) Sí la suma de menciones en el texto completo es mayor o igual que UMBRAL_MENCIONES.
    
    Args:
        patrones: Lista de expresiones regulares asociadas al tema a evaluar.
        low: Cuerpo completo del texto judicial normalizado en minúsculas.
        umbral: UMBRAL_MENCIONES
        low_resumen: Texto del resumen (hechos/agravios/resolución) estructurado normalizado en minúsculas.

    Returns:
        bool: True si el tema es detectado según los criterios; False en caso contrario.
    """

    # (a)
    if low_resumen:
        for patron in patrones:
            if re.search(patron, low_resumen): # Detiene la búsqueda al encontrar la primera coincidencia.
                return True
    total = 0 # Contador
    for patron in patrones: # Itera el diccionario de expresiones regulares
        n = len(re.findall(patron, low))
        if patron in PATRONES_ALTA_PRECISION and n >= 1: # (b) Excepción de PATRONES_ALTA_PRECISION
            return True
        total += n # Acumula el conteo
    return total >= umbral # (c)


def detectar_temas_extendido(
    texto: str, umbral: int = UMBRAL_MENCIONES, texto_resumen: str = ""
) -> list[str]:
    """
    Ejecuta la detección temática del diccionario completo de categorias (PATRONES_TEMA_TODOS).

    A diferencia de la función base (_tema_detectado), este método evalúa el catálogo completo 
    de temas, permitiendo identificar todos los temas existentes en una sentencia individual.

    Args:
        texto: Texto completo de la sentencia a analizar.
        umbral: UMBRAL_MENCIONES
        texto_resumen: Resumen (hechos/agravios/resolución) de sentencia.

    Returns:
        Una lista con los nombres de todos los temas detectados en el documento.
    """

    # Sí el texto viene vacío, regresa lista vacía.
    if not texto:
        return []
    low = texto.lower() # Convierte el texto a minúsculas
    low_resumen = texto_resumen.lower() if texto_resumen else ""
    encontrados = [] # Lista que guarda los temas
    for tema, patrones in PATRONES_TEMA_TODOS.items():
        if _tema_detectado(patrones, low, umbral, low_resumen): # Llama a _tema_detectado pasándole los patrones del "tema actual"
            encontrados.append(tema) # Si da True, agrega el nombre del tema a la lista de encontrados
    return encontrados


def detectar_temas_extendido_con_otro(
    texto: str, umbral: int = UMBRAL_MENCIONES, texto_resumen: str = ""
) -> list[str]:
    """
    Asegura que ningún documento se quede sin etiqueta.

    Sí hay tema, tema; caso contrario asigna la etiqueta [OTRO]
    """

    temas = detectar_temas_extendido(texto, umbral=umbral, texto_resumen=texto_resumen) # Detección del tema. 
    return temas if temas else [OTRO] 


def detectar_temas(
    texto: str, umbral: int = UMBRAL_MENCIONES, texto_resumen: str = ""
) -> list[str]:
    """
    Ejecuta la detección temática del diccionario PATRONES_TEMA de categorias.

    Evalúa el catálogo completo de temas, permitiendo identificar todos los temas
    existentes en una sentencia individual.

    Args:
        texto: Texto completo de la sentencia a analizar.
        umbral: UMBRAL_MENCIONES
        texto_resumen: Resumen (hechos/agravios/resolución) de sentencia.

    Returns:
        Una lista con los nombres de todos los temas detectados en el documento.
    """

    # Sí el texto viene vacío, regresa lista vacía.
    if not texto:
        return []
    low = texto.lower() # Convierte el texto a minúsculas
    low_resumen = texto_resumen.lower() if texto_resumen else ""
    encontrados = []
    for tema, patrones in PATRONES_TEMA.items():
        if _tema_detectado(patrones, low, umbral, low_resumen):
            encontrados.append(tema)
    return encontrados


def detectar_temas_con_otro(
    texto: str, umbral: int = UMBRAL_MENCIONES, texto_resumen: str = ""
) -> list[str]:
    """
    Asegura que ningún documento se quede sin etiqueta.

    Sí hay tema, tema; caso contrario asigna la etiqueta [OTRO]
    """

    temas = detectar_temas(texto, umbral=umbral, texto_resumen=texto_resumen) # Detección del tema.
    return temas if temas else [OTRO]


def calcular_nucleo_dinamico(
    textos: list[str], top_n: int = 10, umbral: int = UMBRAL_MENCIONES
) -> list[tuple[str, int]]:
    """
    Calcula dinámicamente los temas más frecuentes del corpus.

    Funciona como parche a las definiciones previos de un núcleo estático, permite que el panel de KPIs refleje
    la frecuencia histórica real a medida que el corpus evoluciona.

    Args:
        textos: Lista con el contenido textual de todos los documentos del corpus.
        top_n: Cantidad máxima de temas principales a retornar (por defecto 10).
        umbral: UMBRAL_MENCIONES

    Returns:
        list[tuple[str, int]]: Lista de tuplas ordenadas (nombre_tema, num_documentos),
        organizadas por frecuencia descendente y nombre ascendente en caso de empate.
    """

    from collections import Counter

    conteo: Counter[str] = Counter() # Contador
    for texto in textos:
        for tema in detectar_temas_extendido(texto, umbral=umbral): # Detecta temas presentes en el documento.
            conteo[tema] += 1

    # Ordena los resultados: 
    # 1) Frecuencia de mayor a menor (-par[1]) 
    # 2) Orden alfabético del tema en caso de empate (par[0]).
    ordenados = sorted(conteo.items(), key=lambda par: (-par[1], par[0]))
    return ordenados[:top_n]


def detectar_temas_panel_con_bucket(
    texto: str,
    umbral: int = UMBRAL_MENCIONES,
    nucleo: list[str] | None = None,
    texto_resumen: str = "",
) -> list[str]:

    """
    Pensada específicamente para el gráfico "Sentencias por tema" del panel de KPIs:


    Lógica para definir los temas estandarizados del gráfico:
        (a) Si el documento tiene algún tema de núcleo principal -> se devuelven esos (puede ser más de uno).
        (b) Si no hay tema de núcleo principal pero sí de PATRONES_TEMA_TODOS o temas propuestos por API de Gemini,
            agrupa en [OTROS_ESPECIFICOS] (= "Otros temas identificados").
        (c) Si no se detecta ninguna coincidencia, asigna [OTRO] (= "Otro / no identificado").

    Args:
        texto: Cuerpo completo del texto.
        umbral: UMBRAL_MENCIONES
        nucleo: Lista de temas a tratar como prioritarios (Por defecto PATRONES_TEMA).
        texto_resumen: Resumen (hechos/agravios/resolución) de sentencia.

    Returns:
        list[str]: Lista con los temas principales, el bucket de otros específicos, 
        o el bucket de no identificados.
    """

    nucleo_set = set(nucleo) if nucleo is not None else set(PATRONES_TEMA.keys()) #Identifica que usará como nucleo, en caso de no definirse, usa PATRONES_TEMA por defecto.
    temas_todos = detectar_temas_extendido(texto, umbral=umbral, texto_resumen=texto_resumen)
    temas_core = [t for t in temas_todos if t in nucleo_set]
    if temas_core: # (a)
        return temas_core
    if temas_todos: 
        return [OTROS_ESPECIFICOS] # (b)
    return [OTRO] # (c)


def contar_menciones(texto: str) -> dict[str, int]:
    """
    Cuantifica las frecuencias absolutas de mención para cada tema en una sentencia.

    Actúa como un mecanismo de auditoría y depuración. Devuelve un diccionario con el tema y el total de menciones 
    detectadas en el texto.
    """

    # Sí el texto viene vacío, devuelve un diccionario donde todos los temas tienen conteo cero.
    if not texto:
        return {t: 0 for t in TEMAS}
    low = texto.lower() # Convierte el texto a minúsculas
    return {
        tema: sum(len(re.findall(p, low)) for p in patrones) # Suma las coincidencias
        for tema, patrones in PATRONES_TEMA.items()
    }
