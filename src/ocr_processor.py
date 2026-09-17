"""Procesamiento OCR (Reconocimiento Óptico de Caracteres) de documentos PDF escaneados."""

import subprocess
from pathlib import Path
from typing import Any

from pdf_extractor import extract_pdf


OCR_OUTPUT_FOLDER = Path("documents/output/ocr_pdf")

# Límite de tiempo por documento para evitar bloqueos del pipeline. Si un PDF presenta anomalías 
# estructurales o corrupción de datos que congelena Tesseract/OCRmyPDF, la tarea se aborta de forma
# controlada tras 300 segundos, permitiendo continuar con el resto del corpus.
OCR_TIMEOUT_SECONDS = 300


def run_ocr(file_path: Path) -> dict[str, Any]:
    """
    Genera un PDF con capa de texto mediante OCRmyPDF y después extrae su texto con PyMuPDF.

    Si OCRmyPDF no termina dentro de OCR_TIMEOUT_SECONDS, se aborta ese documento en particular
    (se marca como error) y el pipeline continúa con el resto — nunca se queda trabado esperando 
    a un solo archivo.
    """

    OCR_OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Define el nombre del archivo de salida para el PDF con OCR, basado en el nombre del archivo original.
    output_file = (
        OCR_OUTPUT_FOLDER
        / f"{file_path.stem}_ocr.pdf"
    )

    # Comando para ejecutar OCRmyPDF con Tesseract, configurado para español, corrección de inclinación y rotación de páginas.
    command = [
        "ocrmypdf",
        "--language",
        "spa",
        "--deskew",
        "--rotate-pages",
        "--skip-text",
        str(file_path),
        str(output_file),
    ]

    try:
        process = subprocess.run( # Ejecuta el comando OCRmyPDF en un subproceso, capturando la salida y errores, con un tiempo límite definido.
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=OCR_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        # Limpia el archivo parcial que OCRmyPDF pudo haber dejado a medias.
        if output_file.exists():
            output_file.unlink()

        raise RuntimeError(
            f"OCRmyPDF excedió el límite de {OCR_TIMEOUT_SECONDS} segundos "
            f"y fue interrumpido para no bloquear el resto del corpus. "
            f"Revisa este documento por separado: {file_path.name}"
        ) from error

    if process.returncode != 0: # Si OCRmyPDF devuelve un código de error distinto de cero, se lanza una excepción con la salida y errores capturados.
        # Limpia el archivo parcial que OCRmyPDF pudo haber dejado a medias.
        raise RuntimeError(
            "OCRmyPDF no pudo procesar el documento.\n"
            f"Salida: {process.stdout}\n"
            f"Error: {process.stderr}"
        )

    result = extract_pdf(output_file) # Extrae el texto del PDF generado por OCRmyPDF usando la función extract_pdf definida en pdf_extractor.py.
    result["extraction_method"] = (
        "ocrmypdf+tesseract+pymupdf"
    )
    result["ocr_pdf"] = str(output_file)
    result["status"] = "completed"
    result["needs_ocr"] = False # Quita la marca de "necesita OCR" para indicar que el documento ya ha sido procesado.

    return result
