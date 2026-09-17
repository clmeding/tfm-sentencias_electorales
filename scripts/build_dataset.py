"""
Consolida todos los documentos ya procesados (documents/output/*.json) en un único dataset tabular (CSV), 
listo para el análisis exploratorio y, más adelante, el modelo predictivo.

Requiere que ya hayas ejecutado:
    1. src/main.py                     (extracción + OCR + limpieza)
    2. scripts/reclean_existing.py     (si tenías documentos de antes de
                                         que existiera la limpieza de
                                         membrete)

Este script NO vuelve a hacer OCR ni a descargar nada -- solo lee los JSON ya generados, 
aplica la detección de sentido o resolutivo, y junta todo en una sola tabla.

Uso:
    python scripts/build_dataset.py

Salida:
    data/dataset_sentencias.csv
"""

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentido_extractor import (  # noqa: E402
    detect_sentido,
    infer_procedure_category,
    is_acuerdo_plenario,
    resolve_category,
)


OUTPUT_FOLDER = Path("documents/output")
DATASET_FILE = Path("data/dataset_sentencias.csv")

# Documentos excluidos manualmente por OCR ilegible (páginas escaneadas
# al revés que la corrección automática de orientación no logró arreglar
# -- ver scripts/diagnostico_calidad_ocr.py, corrida del 25-ago-2026:
# 7 de 886 documentos, 0.8%, casi todos del lote 2017). El arreglo de
# rotación en ocr_processor.py ya evita este problema en documentos
# NUEVOS; estos quedan excluidos porque ya están procesados con el
# texto malo y no vale la pena volver a correr OCR sobre todo el corpus
# por 7 casos.
EXCLUIDOS_OCR_ILEGIBLE = {
    "SENTENCIA TEE-JDCN-25-2022 Y ACUMULADOS 19 DE ENERO 2023.pdf",
    "TEE-PES-25-2017.pdf",
    "TEE-RV-11-2017.pdf",
    "TEE-PES-22-2017.pdf",
    "TEE-PES-21-2017.pdf",
    "TEE-RV-08-2017.pdf",
    "TEE-PES-20-2017.pdf",
}

COLUMNS = [
    "source_file",
    "json_filename",
    "category",
    "expediente",
    "year",
    "document_date",
    "document_type",
    "has_acumulados",
    "needs_ocr",
    "letterhead_removed",
    "text_length",
    "sentido",
    "sentido_metodo",
    "sentido_evidencia",
    "es_acuerdo_plenario",
]


def load_text(data: dict) -> str:
    """Usa el texto ya limpio si existe; si no, cae al texto crudo."""

    if "extracted_text_clean" in data:
        return data["extracted_text_clean"]

    units = data.get("units", [])
    return "\n\n".join(unit.get("text", "") for unit in units if unit.get("text"))


def main() -> None:
    json_files = sorted(OUTPUT_FOLDER.glob("*.json"))

    if not json_files:
        print(
            "No se encontraron documentos en documents/output/. "
            "Corre primero src/main.py."
        )
        return

    rows = []
    sin_texto = 0
    conteo_sentido: dict[str, int] = {}
    excluidos_encontrados = 0

    for json_file in json_files:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        text = load_text(data)

        filename = data.get("source_file", json_file.stem)

        if filename in EXCLUIDOS_OCR_ILEGIBLE:
            excluidos_encontrados += 1
            continue

        if not text.strip():
            sin_texto += 1

        category = resolve_category(filename, data.get("category"))
        acuerdo_plenario = is_acuerdo_plenario(filename, text)

        procedure_category = infer_procedure_category(filename, category)
        sentido_result = detect_sentido(text, procedure_category)

        sentido = sentido_result["sentido"]
        conteo_sentido[sentido] = conteo_sentido.get(sentido, 0) + 1

        rows.append({
            "source_file": filename,
            "json_filename": json_file.name,
            "category": category,
            "expediente": data.get("expediente"),
            "year": data.get("year"),
            "document_date": data.get("document_date"),
            "document_type": data.get("document_type"),
            "has_acumulados": data.get("has_acumulados"),
            "needs_ocr": data.get("needs_ocr"),
            "letterhead_removed": data.get("letterhead_removed"),
            "text_length": len(text),
            "sentido": sentido,
            "sentido_metodo": sentido_result.get("sentido_metodo"),
            "sentido_evidencia": sentido_result.get("sentido_evidencia"),
            "es_acuerdo_plenario": acuerdo_plenario,
        })

    DATASET_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(DATASET_FILE, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    acuerdos_plenarios = sum(1 for row in rows if row["es_acuerdo_plenario"])

    print(f"Dataset guardado en: {DATASET_FILE}")
    print(f"Documentos incluidos: {len(rows)}")
    print(f"Documentos excluidos por OCR ilegible: {excluidos_encontrados}")
    print(f"Documentos sin texto extraído: {sin_texto}")
    print(
        f"Documentos marcados como Acuerdo Plenario (no sentencia, "
        f"es_acuerdo_plenario=True): {acuerdos_plenarios}"
    )
    print("\nDistribución de 'sentido' detectado:")
    for label, count in sorted(conteo_sentido.items(), key=lambda x: -x[1]):
        pct = (count / len(rows)) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
