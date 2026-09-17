"""
Revisa si hay sentencias nuevas publicadas en trieen.mx comparando el
catálogo actual contra un "manifiesto" (lista de nombres de archivo ya
conocidos) que se guarda como texto plano dentro del repositorio.

A diferencia de download_trieen.py, este script NO descarga los PDFs
-- eso implicaría subir archivos binarios pesados al repositorio de
código, algo que no conviene. Solo detecta si hay documentos nuevos y
actualiza el manifiesto, para que un flujo de GitHub Actions pueda avisar
(por ejemplo, abriendo un Issue) sin necesidad de bajar los PDFs al propio
repositorio.

La descarga y el procesamiento (OCR, limpieza, etc.) de los documentos
nuevos se sigue haciendo donde corras el pipeline completo (tu Mac u
otro entorno con más recursos).

Uso:
    python scripts/check_new_documents.py

Salida:
    - Actualiza data/known_documents.json
    - Imprime un resumen legible
    - Si se ejecuta dentro de GitHub Actions, escribe variables en
      $GITHUB_OUTPUT (has_new_documents, new_documents_count,
      new_documents_list) para que el workflow decida si abre un Issue.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_trieen import CatalogFetchError, fetch_catalog  # noqa: E402


MANIFEST_FILE = Path("data/known_documents.json")


def load_known_documents() -> set[str]:
    if not MANIFEST_FILE.exists():
        return set()

    data = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    known = set()

    for filenames in data.values():
        known.update(filenames)

    return known


def save_manifest(catalog: dict[str, list[str]]) -> None:
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_github_output(new_documents: list[str]) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")

    if not output_path:
        return

    with open(output_path, "a", encoding="utf-8") as handle:
        handle.write(f"has_new_documents={'true' if new_documents else 'false'}\n")
        handle.write(f"new_documents_count={len(new_documents)}\n")

        # Formato multilinea seguro para GITHUB_OUTPUT
        handle.write("new_documents_list<<EOF\n")
        for name in new_documents:
            handle.write(f"- {name}\n")
        handle.write("EOF\n")


def main() -> None:
    print("Consultando el catálogo público de trieen.mx...")

    try:
        catalog = fetch_catalog()
    except CatalogFetchError as exc:
        # fetch_catalog() ya reintentó varias veces internamente para
        # absorber bloqueos anti-bot de un solo instante. Si llegamos
        # aquí es porque el bloqueo persistió tras esos reintentos --
        # eso sí debe marcarse como falla para que sigas recibiendo el
        # aviso de GitHub Actions como hasta ahora, solo que con un
        # mensaje legible en vez de un traceback crudo de Python.
        print(f"AVISO: {exc}")
        print("No se actualiza el manifiesto en esta corrida. Revisa manualmente si trieen.mx sigue bloqueando las solicitudes.")
        sys.exit(1)

    known = load_known_documents()

    all_current = {
        name
        for filenames in catalog.values()
        for name in filenames
    }

    new_documents = sorted(all_current - known)

    print(f"Documentos en el catálogo actual: {len(all_current)}")
    print(f"Documentos ya conocidos (manifiesto anterior): {len(known)}")
    print(f"Documentos nuevos detectados: {len(new_documents)}")

    for name in new_documents:
        print(f"  NUEVO: {name}")

    save_manifest(catalog)
    write_github_output(new_documents)


if __name__ == "__main__":
    main()
