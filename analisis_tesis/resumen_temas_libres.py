"""Cuenta la frecuencia de TODOS los temas que Gemini identificó libremente
en el corpus (data/temas_gemini.json, generado por
scripts/detectar_temas_gemini.py), para que puedas revisar el universo
completo de temas descubiertos ANTES de decidir cuáles curar/agrupar para
los gráficos del panel de KPIs.

Este script NO decide nada por ti -- solo organiza la información para que
la curación (agrupar variantes de redacción parecidas, descartar temas de
un solo documento, fusionar sinónimos) la hagas tú con criterio jurídico,
revisando también los expedientes de ejemplo de cada tema.

Uso (desde la raíz del repo, después de correr detectar_temas_gemini.py):
    python scripts/resumen_temas_libres.py
    python scripts/resumen_temas_libres.py --top 100
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMAS_GEMINI_JSON = REPO_ROOT / "data" / "temas_gemini.json"
DATASET_CSV = REPO_ROOT / "data" / "dataset_sentencias.csv"
SALIDA_CSV = REPO_ROOT / "data" / "temas_libres_frecuencia.csv"

CONFIANZA_ALTA_MEDIA = {"alta", "media"}


def normalizar(tema: str) -> str:
    """Clave de agrupación: minúsculas, espacios colapsados, sin punto final.
    Deliberadamente simple (no quita acentos ni hace stemming) -- el modelo
    corre con temperature=0.1 y ejemplos consistentes, así que la mayoría de
    las variantes reales serán mayúsculas/minúsculas o un punto de más, no
    sinónimos distintos. Los sinónimos genuinos (ej. "Paridad de género" vs
    "Equidad de género en candidaturas") hay que agruparlos a mano -- por
    eso este reporte incluye ejemplos de expedientes por tema.
    """
    return re.sub(r"\s+", " ", tema.strip().lower()).rstrip(".")


def cargar_expediente_por_json(jf: str, df: pd.DataFrame) -> str:
    fila = df[df["json_filename"] == jf]
    if fila.empty:
        return jf
    expediente = fila.iloc[0].get("expediente")
    if isinstance(expediente, str) and expediente.strip():
        return expediente
    return Path(jf).stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top", type=int, default=60, help="Cuántos temas mostrar en consola (default 60, todos van al CSV).")
    args = parser.parse_args()

    if not TEMAS_GEMINI_JSON.exists():
        sys.exit(
            f"ERROR: no existe '{TEMAS_GEMINI_JSON}'. Corre primero:\n"
            "    python scripts/detectar_temas_gemini.py"
        )

    registros = json.loads(TEMAS_GEMINI_JSON.read_text(encoding="utf-8"))
    registros = [r for r in registros if "error" not in r]

    df_inventario = pd.read_csv(DATASET_CSV) if DATASET_CSV.exists() else pd.DataFrame(columns=["json_filename", "expediente"])

    # clave_normalizada -> {"variantes": Counter(texto_original), "docs": set(json_filename), "docs_alta_media": set, "expedientes_ejemplo": []}
    grupos: dict[str, dict] = defaultdict(lambda: {"variantes": defaultdict(int), "docs": set(), "docs_alta_media": set()})

    total_docs_con_tema = 0
    total_docs_otro = 0
    total_docs_baja = 0

    for r in registros:
        jf = r["json_filename"]
        temas = r.get("temas") or []
        confianza = r.get("confianza", "baja")

        if not temas:
            total_docs_otro += 1
            continue
        if confianza not in CONFIANZA_ALTA_MEDIA:
            total_docs_baja += 1
            # Igual se cuenta para el descubrimiento (ver docstring), pero
            # no se marca como alta/media.
        total_docs_con_tema += 1

        for tema in temas:
            clave = normalizar(tema)
            grupos[clave]["variantes"][tema] += 1
            grupos[clave]["docs"].add(jf)
            if confianza in CONFIANZA_ALTA_MEDIA:
                grupos[clave]["docs_alta_media"].add(jf)

    filas = []
    for clave, info in grupos.items():
        etiqueta_representativa = max(info["variantes"].items(), key=lambda kv: kv[1])[0]
        variantes_distintas = len(info["variantes"])
        docs = sorted(info["docs"])
        ejemplos = [cargar_expediente_por_json(jf, df_inventario) for jf in docs[:3]]
        filas.append(
            {
                "tema": etiqueta_representativa,
                "n_docs": len(info["docs"]),
                "n_docs_alta_media": len(info["docs_alta_media"]),
                "variantes_de_redaccion": variantes_distintas,
                "expedientes_ejemplo": "; ".join(ejemplos),
            }
        )

    tabla = pd.DataFrame(filas).sort_values("n_docs", ascending=False).reset_index(drop=True)
    tabla["pct_docs"] = (100.0 * tabla["n_docs"] / max(total_docs_con_tema, 1)).round(1)

    SALIDA_CSV.parent.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA_CSV, index=False)

    print(f"Documentos procesados por Gemini (sin error): {len(registros)}")
    print(f"  - con al menos un tema identificado: {total_docs_con_tema}")
    print(f"  - de esos, con confianza baja (revisar con cuidado): {total_docs_baja}")
    print(f"  - sin tema identificado ('Otro / no identificado'): {total_docs_otro}")
    print(f"\nTemas distintos descubiertos (tras normalizar mayúsc./espacios): {len(tabla)}")
    print(f"Guardado en: {SALIDA_CSV}\n")

    print(f"{'TEMA':45s} {'#docs':>6s} {'%docs':>6s} {'variantes':>10s}  expedientes de ejemplo")
    for _, fila in tabla.head(args.top).iterrows():
        print(
            f"{fila['tema']:45.45s} {fila['n_docs']:6d} {fila['pct_docs']:5.1f}% "
            f"{fila['variantes_de_redaccion']:10d}  {fila['expedientes_ejemplo']}"
        )

    if len(tabla) > args.top:
        print(f"\n... y {len(tabla) - args.top} temas más en el CSV completo.")

    n_una_variante_mas = int((tabla["variantes_de_redaccion"] > 1).sum())
    if n_una_variante_mas:
        print(
            f"\nAVISO: {n_una_variante_mas} de estos temas tienen más de una "
            "redacción distinta que se agruparon como el mismo (variación de "
            "mayúsculas/espacios). Revisa la columna 'variantes_de_redaccion' "
            "si quieres confirmar que la normalización no juntó cosas "
            "distintas por error."
        )

    print(
        "\nSiguiente paso (curación, la haces tú con criterio jurídico): "
        "revisa este CSV, decide qué temas son lo bastante frecuentes y "
        "específicos para los gráficos del panel de KPIs (mismo criterio "
        "que ya usaste para los 6 'núcleo' originales), y cuáles conviene "
        "fusionar por ser sinónimos con redacción distinta. Cuando tengas "
        "la lista curada, dime cuáles son y actualizamos PATRONES_TEMA / "
        "TEMAS_CON_OTRO en src/detectar_temas.py (o el equivalente para el "
        "flujo basado en Gemini) para que el panel y el reordenamiento de "
        "precedentes por tema usen esa lista final."
    )


if __name__ == "__main__":
    main()
