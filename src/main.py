"""Punto de entrada del pipeline de extracción documental."""

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

from cleaner import strip_scanned_letterhead
from document_loader import find_documents
from docx_extractor import extract_docx
from pdf_extractor import extract_pdf
from ocr_processor import run_ocr
from metadata_extractor import extract_metadata


INPUT_FOLDER = Path("documents/input")
OUTPUT_FOLDER = Path("documents/output")

# Número máximo de procesos en paralelo para evitar saturar el servidor debido al alto consumo de 
# capacidad de CPU del OCR.
MAX_WORKERS = 4


def output_paths_for(file_path: Path) -> tuple[Path, Path]:
    """
    Calcula las rutas de salida (json, txt) para un documento de entrada.

    Args:
        file_path: Ruta del archivo original de entrada.

    Returns:
        tuple[Path, Path]: Tupla que contiene la ruta de salida para el JSON y para el texto plano.
    """

    relative_path = file_path.relative_to(INPUT_FOLDER)

    # relative_path quita la ruta raiz
    # .parts desglosa la ruta en una tupla (ej. folder1/folder2/doc.pdf --> ('folder1', 'folder2', 'doc.pdf'))
    # .with_suffix("") quita la extensión, ej. (.pdf)
    # "_*_".join(...) reemplaza con guiones bajos (ej. folder1_folder2_doc)
    safe_relative_name = "_".join(
        relative_path.with_suffix("").parts 
    )

    json_file = OUTPUT_FOLDER / f"{safe_relative_name}.json"
    text_file = OUTPUT_FOLDER / f"{safe_relative_name}.txt"

    return json_file, text_file


def already_processed(file_path: Path) -> bool:
    """
    
    Verifica si un documento cuenta con un resultado de extracción válido y previo. 
    
    Permite interrumpir el pipeline (Ctrl+C) y reanudar sin tener que repetir el trabajo ya hecho.
    Revisa que los archivos ya existan, que no estén dañados y que realmente se hayan terminado de
    procesar (es decir, que no les falte OCR).

    Args:
        file_path: Ruta del archivo de entrada a verificar.

    Returns:
        bool: True si el documento ya fue procesado con éxito; False en caso contrario.
    """

    json_file, text_file = output_paths_for(file_path) # Busca las rutas de salida.

    # Sí el archivo de salida no existe, no está procesado.
    if not (json_file.exists() and text_file.exists()):
        return False

    try:
        saved_result = json.loads(json_file.read_text(encoding="utf-8")) # Intenta abrir el documento
    except (json.JSONDecodeError, OSError): # JSON corrupto o ilegible: mejor reprocesar que confiar en el.
        return False # Se envía para reproceso

    # Si el resultado guardado todavia indica que le faltaba OCR, no es un resultado valido. Es 
    # posible que viene de corridas antiguas con fallas.
    if saved_result.get("needs_ocr") is True:
        return False

    return True


def extract_document(
    file_path: Path,
) -> dict[str, Any]:
    """
    Selecciona y ejecuta el extractor adecuado según la extensión del archivo.

    Si el PDF no tiene texto embebido suficiente (needs_ocr=True), se reprocesa automáticamente con
    OCR antes de devolver el resultado.

    Args:
        file_path: Ruta del documento a procesar.

    Returns:
        dict[str, Any]: Diccionario con los resultados estructurados de la extracción.

    Raises:
        ValueError: Si el formato del archivo no es compatible con el pipeline.
    """

    extension = file_path.suffix.lower() # Extrae extension y convierte a minúsculas.

    # Identifica documentos PDF y extrae contenido
    if extension == ".pdf":
        result = extract_pdf(file_path)

        # Sí el PDF no es legible, ejecuta OCR.
        if result["needs_ocr"]:
            result = run_ocr(file_path)

        return result

    # Identifica documentos Word (.docx) y extrae el contenido.
    if extension == ".docx":
        return extract_docx(file_path)

    # Error si la extensión no es ".docx"
    raise ValueError(
        f"Formato no compatible: {extension}"
    )


def save_result(
    file_path: Path,
    result: dict[str, Any],
) -> None:
    """Guarda el resultado de extracción en JSON y TXT."""

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    relative_path = file_path.relative_to(
        INPUT_FOLDER
    )

    json_file, text_file = output_paths_for(file_path)

    metadata = extract_metadata(
        file_path=file_path,
        input_root=INPUT_FOLDER,
    )

    complete_result = {
        "source_file": file_path.name,
        "relative_path": str(relative_path),
        "extension": file_path.suffix.lower(),
        "processed_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        **metadata,
        **result,
    }

    extracted_text = "\n\n".join(
        unit["text"]
        for unit in result["units"]
        if unit.get("text")
    )

    # El membrete institucional (escudo + "TRIBUNAL ESTATAL ELECTORAL" +
    # "NAYARIT") solo aparece una vez, al principio del documento completo
    # -- por eso se limpia aqui, sobre el texto ya unido, y no por pagina.
    letterhead_result = strip_scanned_letterhead(extracted_text)
    extracted_text = letterhead_result["text"]

    complete_result["letterhead_removed"] = letterhead_result["letterhead_removed"]
    complete_result["extracted_text_clean"] = extracted_text

    json_file.write_text(
        json.dumps(
            complete_result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    text_file.write_text(
        extracted_text,
        encoding="utf-8",
    )


def process_one(file_path: Path) -> tuple[Path, dict[str, Any] | None, str | None]:
    """
    Procesa un solo documento de forma aislada (para usarse en un worker
    de otro proceso). Devuelve (archivo, resultado, error) — exactamente
    uno de (resultado, error) será distinto de None.
    """

    try:
        result = extract_document(file_path)
        return file_path, result, None
    except Exception as error:
        return file_path, None, str(error)


def main() -> None:
    """Ejecuta el pipeline documental sobre todos los documentos pendientes."""

    all_documents = find_documents(
        INPUT_FOLDER
    )

    pending_documents = [
        document
        for document in all_documents
        if not already_processed(document)
    ]

    already_done = len(all_documents) - len(pending_documents)

    print(
        f"\nDocumentos encontrados: {len(all_documents)}"
    )
    print(
        f"Ya procesados en una corrida anterior (se omiten): {already_done}"
    )
    print(
        f"Por procesar en esta corrida: {len(pending_documents)}\n"
    )

    completed_count = 0
    ocr_applied_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one, document): document
            for document in pending_documents
        }

        for future in as_completed(futures):
            document = futures[future]
            relative_path = document.relative_to(INPUT_FOLDER)
            file_path, result, error = future.result()

            if error is not None:
                error_count += 1
                print(f"  ERROR en {relative_path}: {error}")
                continue

            save_result(
                file_path=file_path,
                result=result,
            )

            if result["extraction_method"].startswith("ocrmypdf"):
                ocr_applied_count += 1
                print(f"  COMPLETADO (con OCR): {relative_path}")
            else:
                completed_count += 1
                print(f"  COMPLETADO: {relative_path}")

    print("\nRESUMEN")
    print(f"Ya procesados antes (omitidos): {already_done}")
    print(f"Extracciones directas completadas: {completed_count}")
    print(f"Documentos procesados con OCR: {ocr_applied_count}")
    print(f"Errores: {error_count}")


if __name__ == "__main__":
    main()
