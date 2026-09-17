"""
Mide, sobre el corpus YA PROCESADO (documents/output/*.json), qué tan
probable es que el texto extraído de cada documento sea basura de OCR
(por ejemplo, páginas escaneadas al revés que la corrección automática
de orientación no logró arreglar).

NO vuelve a hacer OCR ni descarga nada -- solo lee los JSON que ya
tienes generados y calcula un puntaje rápido por documento. Corre en
segundos, incluso con cientos de documentos.

Criterio: cuenta cuántas palabras muy comunes del español ("de", "la",
"el", "que", "en", "los", "las", "por", "para", "con", "del", "se",
"al") aparecen por cada 1000 caracteres de texto. Un texto real en
español (sentencias, acuerdos, etc.) tiene muchísimas de estas palabras
-- calibrado contra el propio caso real que encontramos
(TEE-RV-11-2017.pdf): el texto bueno midió ~65 y el texto basura (hoja
al revés) midió ~14 en esta escala. El umbral por defecto (25) queda
cómodamente entre los dos.

Uso:
    python scripts/diagnostico_calidad_ocr.py
    python scripts/diagnostico_calidad_ocr.py --umbral 30
    python scripts/diagnostico_calidad_ocr.py --detalle   # muestra un
                                                             extracto de
                                                             cada uno
                                                             marcado
"""

import argparse
import json
import re
from pathlib import Path

OUTPUT_FOLDER = Path("documents/output")
REPORTE_FILE = Path("data/diagnostico_calidad_ocr.csv")

STOPWORDS = [
    " de ", " la ", " el ", " que ", " en ", " los ", " las ",
    " por ", " para ", " con ", " del ", " se ", " al ",
]

LONGITUD_MINIMA = 200  # documentos más cortos que esto no se evalúan
                        # (podrían ser genuinamente breves, no basura)


def load_text(data: dict) -> str:
    """Igual que en build_dataset.py: usa el texto limpio si existe."""

    if "extracted_text_clean" in data:
        return data["extracted_text_clean"]

    units = data.get("units", [])
    return "\n\n".join(
        unit.get("text", "") for unit in units if unit.get("text")
    )


def puntaje_legibilidad(text: str) -> float:
    """Palabras clave comunes por cada 1000 caracteres de texto."""

    normalizado = " " + re.sub(r"\s+", " ", text.lower()) + " "
    conteo = sum(normalizado.count(palabra) for palabra in STOPWORDS)
    caracteres = max(len(text), 1)
    return conteo / caracteres * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--umbral",
        type=float,
        default=25.0,
        help="Puntaje por debajo del cual se marca como sospechoso (default 25).",
    )
    parser.add_argument(
        "--detalle",
        action="store_true",
        help="Muestra un extracto del texto de cada documento sospechoso.",
    )
    args = parser.parse_args()

    json_files = sorted(OUTPUT_FOLDER.glob("*.json"))

    if not json_files:
        print(
            "No se encontraron documentos en documents/output/. "
            "Corre esto desde la raíz del repo."
        )
        return

    filas = []
    sospechosos = []
    muy_cortos = 0

    for json_file in json_files:
        data = json.loads(json_file.read_text(encoding="utf-8"))
        text = load_text(data)
        filename = data.get("source_file", json_file.stem)

        if len(text.strip()) < LONGITUD_MINIMA:
            muy_cortos += 1
            filas.append(
                {
                    "archivo": filename,
                    "json": json_file.name,
                    "longitud_texto": len(text),
                    "puntaje": None,
                    "sospechoso": "muy_corto",
                }
            )
            continue

        puntaje = puntaje_legibilidad(text)
        es_sospechoso = puntaje < args.umbral

        filas.append(
            {
                "archivo": filename,
                "json": json_file.name,
                "longitud_texto": len(text),
                "puntaje": round(puntaje, 2),
                "sospechoso": es_sospechoso,
            }
        )

        if es_sospechoso:
            sospechosos.append((filename, json_file.name, puntaje, text))

    total = len(json_files)
    print(f"Documentos evaluados: {total}")
    print(f"Muy cortos (< {LONGITUD_MINIMA} caracteres, no evaluados): {muy_cortos}")
    print(f"Sospechosos de OCR ilegible (puntaje < {args.umbral}): {len(sospechosos)}")
    print(
        f"Porcentaje sospechoso sobre el total evaluado: "
        f"{len(sospechosos) / max(total - muy_cortos, 1) * 100:.1f}%"
    )
    print()

    if sospechosos:
        print("Lista de documentos sospechosos (archivo, puntaje):")
        for filename, json_name, puntaje, text in sorted(
            sospechosos, key=lambda x: x[2]
        ):
            print(f"  {filename:40s} puntaje={puntaje:.1f}  ({json_name})")
            if args.detalle:
                extracto = re.sub(r"\s+", " ", text[:200])
                print(f"      -> {extracto}")

    REPORTE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with REPORTE_FILE.open("w", encoding="utf-8") as f:
        f.write("archivo,json,longitud_texto,puntaje,sospechoso\n")
        for fila in filas:
            f.write(
                f'"{fila["archivo"]}","{fila["json"]}",'
                f'{fila["longitud_texto"]},{fila["puntaje"]},'
                f'{fila["sospechoso"]}\n'
            )

    print(f"\nGuardado reporte completo en {REPORTE_FILE}")


if __name__ == "__main__":
    main()
