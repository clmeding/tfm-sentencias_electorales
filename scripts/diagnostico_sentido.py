"""Diagnóstico: por qué detect_sentido() no encontró ningún sentido.

Muestra el texto COMPLETO de la sección que el sistema identificó como
"resolutivo" (desde el encabezado RESUELVE hasta el final del documento),
para revisar a simple vista si:

  a) se extrajo la sección correcta (o el ancla "RESUELVE" encontrada no
     era la real, sino una mención en el índice o en una cita de otro
     expediente),
  b) el verbo clave (confirma/revoca/modifica/desecha/sobresee/
     fundado/infundado) está presente pero con ruido de OCR que el
     patrón actual no reconoce (ej. letras espaciadas, letras perdidas,
     caracteres irreconocibles en medio de la palabra).

Uso:
    python scripts/diagnostico_sentido.py "ruta/al/documento.pdf" [categoria]

`categoria` es opcional -- una de 'general', 'incumplimiento' o
'pes_sancionador'. Si no se especifica, se infiere del nombre de archivo
igual que en evaluar_documento().
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evaluar_documento import extraer_texto_documento  # noqa: E402
from sentido_extractor import (  # noqa: E402
    RESOLUTIVO_HEADER_PATTERN,
    FUNDADO_TRANSITION_PATTERN,
    detect_sentido,
    fix_common_ocr_artifacts,
    infer_procedure_category,
)


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python scripts/diagnostico_sentido.py ruta/al/documento.pdf [categoria]")
        sys.exit(1)

    ruta = Path(sys.argv[1])
    categoria_forzada = sys.argv[2] if len(sys.argv) > 2 else None

    if not ruta.exists():
        print(f"ERROR: no existe el archivo: {ruta}")
        sys.exit(1)

    texto_completo = extraer_texto_documento(ruta)
    categoria = categoria_forzada or infer_procedure_category(ruta.name)
    cleaned_text = fix_common_ocr_artifacts(texto_completo)

    matches_resuelve = list(RESOLUTIVO_HEADER_PATTERN.finditer(cleaned_text))
    matches_fundado = list(FUNDADO_TRANSITION_PATTERN.finditer(cleaned_text))

    print("=" * 80)
    print(f"Archivo: {ruta.name}")
    print(f"Categoría de procedimiento usada: {categoria}"
          f"{' (forzada manualmente)' if categoria_forzada else ' (inferida del nombre de archivo)'}")
    print(f"Longitud del texto completo extraído: {len(texto_completo)} caracteres")
    print(f"Coincidencias de encabezado RESUELVE/UELVE encontradas: {len(matches_resuelve)}")
    print(f"Coincidencias del respaldo 'por lo expuesto y fundado, se:': {len(matches_fundado)}")

    if matches_resuelve:
        resolutivo_section = cleaned_text[matches_resuelve[-1].end():]
        print("Se usó la ÚLTIMA coincidencia de RESUELVE/UELVE como ancla.")
    elif matches_fundado:
        resolutivo_section = cleaned_text[matches_fundado[-1].end():]
        print("No se encontró RESUELVE -- se usó el respaldo 'por lo expuesto y fundado, se:'.")
    else:
        resolutivo_section = None
        print("NO se encontró ningún ancla (ni RESUELVE ni el respaldo). "
              "Este documento caería en 'sin_seccion_resuelve'.")

    print("=" * 80)
    print("\nTEXTO COMPLETO DE LA SECCIÓN IDENTIFICADA COMO RESOLUTIVO:\n")
    print(resolutivo_section if resolutivo_section else "(no aplica)")
    print("\n" + "=" * 80)

    resultado = detect_sentido(texto_completo, categoria)
    print(f"\nResultado de detect_sentido():")
    for clave, valor in resultado.items():
        print(f"  {clave}: {valor}")


if __name__ == "__main__":
    main()
