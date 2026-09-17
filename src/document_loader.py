"""Localización de documentos dentro del repositorio."""

from pathlib import Path


SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def find_documents(folder: Path) -> list[Path]:
    """
    Busca documentos PDF y DOCX dentro de una carpeta
    y en todas sus subcarpetas.
    """

    # Valida si la carpeta existe, si no lanza un error.
    if not folder.exists(): 
        raise FileNotFoundError(
            f"No existe la carpeta de entrada: {folder.resolve()}"
        ) 

    # Recorre la estrucutra de carpetas filtrando archivos .pfd y .docx 
    documents = [
        file
        for file in folder.rglob("*") # Escanea todos los archivos y carpetas dentro del directorio.
        if file.is_file() # Verifica que sea un archivo y su extensión.
        and file.suffix.lower() in SUPPORTED_EXTENSIONS
    ] 

    return sorted(documents)