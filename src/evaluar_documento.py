"""Evalúa la viabilidad de un resolutivo PROPUESTO en un proyecto de sentencia.

Este es el entregable PRINCIPAL del TFM (los resúmenes son un producto
secundario). La persona sube un documento (PDF o DOCX) que redactó ella
misma como PROYECTO de sentencia -- con la estructura habitual (Sumario de
la decisión / Glosario / Antecedentes / Consideraciones / RESUELVE) y un
resolutivo YA PROPUESTO que quiere DEFENDER -- y esta herramienta:

  1. Extrae el texto del documento, reutilizando el mismo pipeline de
     extracción/OCR/limpieza que se usó para construir el corpus de
     entrenamiento (pdf_extractor / docx_extractor / ocr_processor /
     cleaner), para que el texto tenga el mismo formato que vio el modelo.
  2. Detecta qué sentido está PROPONIENDO el resolutivo
     (sentido_extractor.detect_sentido sobre la sección del RESUELVE).
  3. Recorta el texto ANTES del resolutivo (Antecedentes + Consideraciones,
     igual que en el entrenamiento -- sin esto el modelo estaría leyendo
     la respuesta en vez de predecirla) y se lo da al modelo entrenado
     (explicar_prediccion.predecir_y_explicar) para obtener el sentido que
     el corpus histórico sugeriría, con su explicación LIME.
  4. Compara predicho vs. propuesto: "coincide" / "no coincide". Esto es
     la base para que la persona defienda o reconsidere su resolutivo --
     la herramienta NO corrige el resolutivo, solo evalúa su viabilidad.
  5. Reporta precedentes A FAVOR (mismo sentido -- SUSTENTAN la
     predicción) y OTROS (mismo asunto/tema pero sentido distinto -- NO
     sustentan, se muestran como contexto), usando como criterio de
     similitud un umbral estadístico calculado sobre la propia
     distribución de similitudes del corpus (ver PERCENTIL_SIMILITUD en
     explicar_prediccion.py), no un número fijo elegido a ojo.

Uso:
    python src/evaluar_documento.py ruta/al/proyecto_de_sentencia.pdf
"""

import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from cleaner import strip_scanned_letterhead
from docx_extractor import extract_docx
from explicar_prediccion import predecir_y_explicar
from ocr_processor import run_ocr
from pdf_extractor import extract_pdf
from precedentes_datos import enriquecer_precedentes
from sentido_extractor import (
    FUNDADO_TRANSITION_PATTERN,
    RESOLUTIVO_HEADER_PATTERN,
    detect_sentido,
    infer_procedure_category,
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def pre_resolutivo_text(full_text: str) -> tuple[str, str]:
    """Devuelve (texto_previo_al_resolutivo, metodo_de_corte).

    Misma lógica que `pre_resolutivo_text()` en
    scripts/build_dataset_modelo.py (se duplica aquí porque scripts/ no
    forma parte del paquete src/ y este módulo necesita poder ejecutarse
    de forma independiente). Cualquier cambio en el criterio de corte
    debe replicarse en ambos lugares.
    """
    matches = list(RESOLUTIVO_HEADER_PATTERN.finditer(full_text))
    if matches:
        return full_text[: matches[-1].start()], "resuelve_header"

    fallback = list(FUNDADO_TRANSITION_PATTERN.finditer(full_text))
    if fallback:
        return full_text[: fallback[-1].start()], "fundado_transition"

    return full_text, "sin_corte_texto_completo"


def extraer_texto_documento(file_path: Path) -> str:
    """Extrae y limpia el texto completo de un PDF o DOCX arbitrario,
    reutilizando el mismo pipeline (extracción directa + OCR de respaldo
    + limpieza de membrete institucional) que se usó para construir el
    corpus de entrenamiento.
    """
    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Formato no compatible: '{extension}'. Se aceptan: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )

    if extension == ".pdf":
        result = extract_pdf(file_path)
        if result["needs_ocr"]:
            result = run_ocr(file_path)
    else:
        result = extract_docx(file_path)

    full_text = "\n\n".join(
        unit["text"] for unit in result["units"] if unit.get("text")
    )

    letterhead_result = strip_scanned_letterhead(full_text)
    return letterhead_result["text"]


def evaluar_documento(
    file_path: Path,
    excluir_json_filename: str | None = None,
    num_samples: int = 150,
    nombre_para_categoria: str | None = None,
    categoria_procedimiento: str | None = None,
    usar_temas: bool = True,
) -> dict[str, Any]:
    """Ejecuta la evaluación completa sobre un documento y devuelve un
    dict estructurado con la predicción, el sentido propuesto, la
    comparación entre ambos y los precedentes a favor / otros.

    Args:
        file_path: ruta al PDF o DOCX del proyecto de sentencia. Puede ser
            una ruta temporal (ej. la que genera Streamlit al subir un
            archivo) -- para esos casos usa `nombre_para_categoria` en vez
            de depender de este nombre para inferir la categoría.
        excluir_json_filename: si el documento evaluado ya forma parte
            del corpus de referencia (ej. pruebas con documentos ya
            conocidos), su nombre de archivo json para que no se cite a
            sí mismo como precedente.
        num_samples: cuántas variantes perturbadas evalúa LIME para la
            explicación (ver predecir_y_explicar). Expuesto aquí para
            que la interfaz de demo pueda intercambiar velocidad por
            estabilidad de la explicación sin tocar este módulo.
        nombre_para_categoria: nombre de archivo a usar SOLO para inferir
            la categoría de procedimiento (AP/JIN/JDCN/MII/PES/RV), en vez
            de `file_path.name`. Necesario porque un archivo subido por
            Streamlit se guarda primero en un temporal con nombre genérico
            (ej. "tmpxyz123.pdf") que pierde el nombre original -- y ese
            nombre original suele ser la única pista disponible para saber
            si es un PES ("TEE-PES-11-2024..."), un incidente de
            incumplimiento, etc. Si no se especifica, se usa
            `file_path.name` (comportamiento anterior, correcto para
            archivos que ya están en su ruta final).
        categoria_procedimiento: si se especifica ('general',
            'pes_sancionador' o 'incumplimiento'), se usa directamente y
            NO se intenta inferir del nombre de archivo. Útil como
            selector manual en la interfaz de demo para los casos en que
            el nombre del archivo no sigue la convención "TEE-XXX-NN-
            AAAA" (documentos de prueba, o un caso nuevo que aún no tiene
            número de expediente asignado en el nombre).
        usar_temas: reenvía a explicar_prediccion.predecir_y_explicar() --
            ver su docstring. Default True (decisión de la autora: el tema
            es el segundo criterio de relevancia después del sentido). Pasa
            False si scripts/validar_reranking_temas.py muestra que no
            conviene para tu corpus.
    """
    file_path = Path(file_path)
    texto_completo = extraer_texto_documento(file_path)

    if not texto_completo.strip():
        raise ValueError(
            "No se pudo extraer texto legible del documento. Revisa que "
            "no esté vacío, dañado o protegido."
        )

    procedure_category = categoria_procedimiento or infer_procedure_category(
        nombre_para_categoria or file_path.name
    )

    propuesto = detect_sentido(texto_completo, procedure_category)
    sentido_propuesto = propuesto["sentido"]

    texto_previo, metodo_corte = pre_resolutivo_text(texto_completo)
    texto_previo = texto_previo.strip()

    if not texto_previo:
        raise ValueError(
            "No se pudo separar el texto previo al resolutivo (no se "
            "encontró un encabezado 'RESUELVE' reconocible). Revisa la "
            "estructura del documento -- debe incluir Antecedentes y "
            "Consideraciones antes del resolutivo."
        )

    prediccion = predecir_y_explicar(
        texto_previo,
        excluir_json_filename=excluir_json_filename,
        num_samples=num_samples,
        usar_temas=usar_temas,
    )
    sentido_predicho = prediccion["sentido_predicho"]

    # Agrega link al documento original en trieen.mx y resumen
    # automático (hechos/agravios/resolución) a cada precedente, sin
    # tocar la lógica de similitud de explicar_prediccion.py.
    precedentes_a_favor = enriquecer_precedentes(prediccion["precedentes_a_favor"])
    precedentes_otros = enriquecer_precedentes(prediccion["precedentes_otros"])

    if sentido_propuesto == "no_detectado":
        coincide = None
        mensaje_comparacion = (
            "No se pudo detectar automáticamente qué sentido propone el "
            "resolutivo de este documento -- revísalo manualmente y "
            f"compáralo con la predicción del modelo ('{sentido_predicho}')."
        )
    elif sentido_predicho == sentido_propuesto:
        coincide = True
        mensaje_comparacion = (
            f"El resolutivo propuesto ('{sentido_propuesto}') COINCIDE con "
            "lo que el modelo predice a partir de los antecedentes y "
            f"consideraciones ('{sentido_predicho}')."
        )
    else:
        coincide = False
        mensaje_comparacion = (
            f"El resolutivo propuesto ('{sentido_propuesto}') NO COINCIDE "
            "con lo que el modelo predice a partir de los antecedentes y "
            f"consideraciones ('{sentido_predicho}'). Esto no significa que "
            "el resolutivo propuesto sea incorrecto -- revisa la "
            "explicación y los precedentes para decidir si conviene "
            "reforzar la argumentación."
        )

    return {
        # nombre_para_categoria es el nombre ORIGINAL del archivo subido por
        # el usuario; file_path puede ser un temporal con nombre genérico
        # (ej. "tmpxyz123.pdf" creado por Streamlit) -- usar ese nombre
        # genérico aquí haría que la interfaz muestre un resultado sin
        # identificar de qué documento se trata.
        "archivo": nombre_para_categoria or file_path.name,
        "categoria_procedimiento": procedure_category,
        "metodo_corte_resolutivo": metodo_corte,
        "vista_previa_texto": generar_vista_previa(texto_previo),
        "sentido_propuesto": sentido_propuesto,
        "sentido_propuesto_evidencia": propuesto.get("sentido_evidencia"),
        "sentido_predicho": sentido_predicho,
        "confianza_prediccion": prediccion["confianza"],
        "coincide": coincide,
        "mensaje_comparacion": mensaje_comparacion,
        "explicacion_texto": prediccion["explicacion_texto"],
        "palabras_influyentes": prediccion["palabras_influyentes"],
        "precedentes_a_favor": precedentes_a_favor,
        "precedentes_otros": precedentes_otros,
        "temas_documento": prediccion.get("temas_documento", []),
    }


def generar_vista_previa(texto_previo: str, max_palabras: int = 180) -> str:
    """Extrae una vista previa legible del documento cargado, sin usar
    ningún modelo ni API externa -- puramente extractiva (solo recorta
    texto ya presente en el documento).

    Busca un encabezado de "HECHOS" o "ANTECEDENTES" para empezar ahí
    (evita que la vista previa arranque a mitad de un encabezado de
    tribunal o de carátula); si no encuentra ninguno usa el inicio del
    texto previo al resolutivo. Se corta a `max_palabras` palabras.
    """
    inicio_match = re.search(
        r"\b(HECHOS|ANTECEDENTES)\b",
        texto_previo,
        re.IGNORECASE,
    )
    texto_recorte = texto_previo[inicio_match.start():] if inicio_match else texto_previo
    texto_recorte = texto_recorte.strip()

    palabras = texto_recorte.split()
    if len(palabras) <= max_palabras:
        return texto_recorte
    return " ".join(palabras[:max_palabras]) + " [...]"


def imprimir_resultado(resultado: dict[str, Any]) -> None:
    """Imprime el resultado de la evaluación en un formato legible por consola."""

    print("=" * 80)
    print(f"Documento evaluado: {resultado['archivo']}")
    print(f"Categoría de procedimiento inferida: {resultado['categoria_procedimiento']}")
    print("-" * 80)
    print(f"Sentido PROPUESTO en el resolutivo: {resultado['sentido_propuesto']}")
    if resultado["sentido_propuesto_evidencia"]:
        print(f"  Evidencia: \"{resultado['sentido_propuesto_evidencia']}\"")
    print(
        "Sentido PREDICHO por el modelo (según antecedentes/consideraciones): "
        f"{resultado['sentido_predicho']} "
        f"(confianza {resultado['confianza_prediccion']:.0%})"
    )
    print("-" * 80)
    print(resultado["mensaje_comparacion"])
    print("-" * 80)
    print(resultado["explicacion_texto"])
    print("-" * 80)
    if resultado["precedentes_a_favor"]:
        print("Precedentes A FAVOR (mismo sentido, similitud alta para este corpus):")
        for precedente in resultado["precedentes_a_favor"]:
            print(f"  - {precedente['expediente']} ({precedente['similitud']:.0%})")
    else:
        print("Precedentes A FAVOR: ninguno con similitud suficientemente alta.")
    if resultado["precedentes_otros"]:
        print("Precedentes OTROS (mismo asunto, sentido distinto -- NO sustentan, es contexto):")
        for precedente in resultado["precedentes_otros"]:
            print(
                f"  - {precedente['expediente']} -> {precedente['sentido']} "
                f"({precedente['similitud']:.0%})"
            )
    else:
        print("Precedentes OTROS: ninguno con similitud suficientemente alta.")


def main() -> None:
    if len(sys.argv) != 2:
        print("Uso: python src/evaluar_documento.py ruta/al/documento.pdf")
        sys.exit(1)

    file_path = Path(sys.argv[1])
    if not file_path.exists():
        print(f"ERROR: no existe el archivo '{file_path}'.")
        sys.exit(1)

    try:
        resultado = evaluar_documento(file_path)
    except ValueError as error:
        print(f"ERROR: {error}")
        sys.exit(1)

    imprimir_resultado(resultado)

    salida_json = file_path.with_suffix(".evaluacion.json")
    salida_json.write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultado guardado en: {salida_json}")


if __name__ == "__main__":
    main()
