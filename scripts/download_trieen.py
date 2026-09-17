"""
Descarga el corpus público de sentencias del Tribunal Estatal Electoral
de Nayarit (trieen.mx) para alimentar el pipeline de documents/input.

El sitio publica el listado completo en una página WordPress cuyo
contenido se puede leer de forma fiable a través de su API REST
(wp-json), evitando depender del renderizado en el navegador.

Uso:
    python scripts/download_trieen.py

Requiere: pip install requests
"""

import re
import time
import unicodedata
from html import unescape
from pathlib import Path
from urllib.parse import quote

import requests


WP_API_URL = "https://trieen.mx/wp-json/wp/v2/pages/160"
BASE_FILES_URL = "https://trieen.mx/wp-content/uploads/filestrieen"
OUTPUT_FOLDER = Path("documents/input")
REQUEST_TIMEOUT = 30
DELAY_BETWEEN_DOWNLOADS = 0.5  # segundos, para no saturar el servidor del tribunal

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; TFM-IEEN-research/1.0; "
        "uso académico, Trabajo Fin de Máster)"
    )
}


class CatalogFetchError(Exception):
    """Se lanza cuando trieen.mx no devuelve el catálogo esperado tras varios
    intentos -- típicamente un bloqueo anti-bot momentáneo del hosting
    compartido (WAF/CDN) que responde HTTP 200 con una página de
    verificación en vez del JSON de WordPress, o mantenimiento del sitio.
    No es un error del repositorio ni de las credenciales."""


def _parse_catalog_html(content: str) -> dict[str, list[str]]:
    """Parsea el HTML de WordPress ya extraído hacia el catálogo por categoría."""

    # El HTML de la página agrupa los enlaces bajo encabezados
    # <h1>SENTENCIAS <CATEGORIA></h1>.
    blocks = re.split(
        r"<center><b><h1>(.*?)</h1></b></center>",
        content,
    )

    catalog: dict[str, list[str]] = {}

    for index in range(1, len(blocks), 2):
        category_label = (
            blocks[index]
            .replace("SENTENCIAS", "")
            .strip()
        )

        block = blocks[index + 1] if index + 1 < len(blocks) else ""

        hrefs = re.findall(
            r"<a href='([^']+)'>",
            block,
        )

        filenames = [
            unescape(href).split("/")[-1]
            for href in hrefs
        ]

        catalog[category_label] = filenames

    return catalog


def fetch_catalog(max_retries: int = 3, retry_delay: float = 5.0) -> dict[str, list[str]]:
    """Obtiene el catálogo de sentencias agrupado por categoría (AP, JIN, JDCN, MII, PES, RV).

    Reintenta ante fallas transitorias (bloqueo anti-bot momentáneo,
    hiccup del hosting) antes de darse por vencido. Si tras los
    reintentos sigue sin poder leer un catálogo válido, lanza
    `CatalogFetchError` con un mensaje claro en vez de un traceback crudo
    de JSON.
    """

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(
                WP_API_URL,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            content = response.json()["content"]["rendered"]
            return _parse_catalog_html(content)
        except (requests.exceptions.RequestException, ValueError, KeyError, TypeError) as exc:
            last_error = exc
            if attempt < max_retries:
                print(
                    f"Intento {attempt}/{max_retries} fallido al leer el catálogo "
                    f"({exc!r}); reintentando en {retry_delay:.0f}s..."
                )
                time.sleep(retry_delay)

    raise CatalogFetchError(
        "No se pudo obtener un catálogo válido de trieen.mx tras "
        f"{max_retries} intentos. Posible bloqueo anti-bot temporal del "
        f"hosting o mantenimiento del sitio. Último error: {last_error!r}"
    ) from last_error


def build_download_url(
    category: str,
    filename: str,
) -> str:
    """Construye la URL absoluta y correctamente codificada del archivo."""

    return f"{BASE_FILES_URL}/{category}/{quote(filename)}"


def sanitize_filename(filename: str) -> str:
    """Normaliza el nombre de archivo para evitar problemas de codificación."""

    return unicodedata.normalize("NFC", filename)


def download_catalog(catalog: dict[str, list[str]]) -> None:
    """Descarga cada archivo del catálogo, organizándolo por categoría."""

    total = sum(len(filenames) for filenames in catalog.values())
    downloaded = 0
    skipped = 0
    errors: list[str] = []

    print(f"\nTotal de documentos en el catálogo: {total}\n")

    for category, filenames in catalog.items():
        category_folder = OUTPUT_FOLDER / category
        category_folder.mkdir(parents=True, exist_ok=True)

        for filename in filenames:
            destination = category_folder / sanitize_filename(filename)

            if destination.exists() and destination.stat().st_size > 0:
                skipped += 1
                continue

            url = build_download_url(category, filename)

            try:
                response = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()

                destination.write_bytes(response.content)
                downloaded += 1

                print(f"  OK    [{category}] {filename}")

            except requests.RequestException as error:
                errors.append(f"{category}/{filename}: {error}")
                print(f"  ERROR [{category}] {filename}: {error}")

            time.sleep(DELAY_BETWEEN_DOWNLOADS)

    print("\nRESUMEN DE DESCARGA")
    print(f"Descargados: {downloaded}")
    print(f"Ya existentes (omitidos): {skipped}")
    print(f"Errores: {len(errors)}")

    if errors:
        error_log = OUTPUT_FOLDER / "_errores_descarga.txt"
        error_log.write_text("\n".join(errors), encoding="utf-8")
        print(f"Detalle de errores guardado en: {error_log}")


def main() -> None:
    catalog = fetch_catalog()
    download_catalog(catalog)


if __name__ == "__main__":
    main()
