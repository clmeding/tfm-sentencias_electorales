"""Construye el dataset de MODELADO (distinto del dataset de inventario que
genera build_dataset.py): para cada documento con sentido detectado (y que
no sea Acuerdo Plenario), extrae el texto PREVIO al resolutivo (hechos,
agravios, considerandos) como insumo del modelo predictivo, y guarda el
sentido ya detectado como etiqueta objetivo.

Por qué el texto previo al resolutivo y no el texto completo: el sentido
se detecta precisamente leyendo el resolutivo (la palabra "RESUELVE" y lo
que sigue). Si el modelo usara el texto completo como insumo, en realidad
no estaría "prediciendo" nada -- solo leería la respuesta que ya está en
el propio texto. Para que sea un modelo predictivo real, se entrena con
lo que se argumenta ANTES de la resolución.

Uso:
    python scripts/build_dataset_modelo.py

Documentos etiquetados por Gemini (data/sentido_gemini.json, generado por
detectar_sentido_gemini.py): se incorporan al corpus SOLO si confianza es
"alta" o "media". Como estos documentos no tienen un encabezado "RESUELVE"
detectable por regex (por eso el método de reglas los marcó
no_detectado), se usa el "marcador_resolutivo" verbatim que Gemini extrajo
para ubicar el mismo punto de corte y evitar que el resolutivo (la
respuesta) se filtre como insumo del modelo. Si ese marcador no se
encuentra como subcadena exacta del texto (p. ej. Gemini no lo copió
literal), se aplica un corte aproximado conservador (se descarta el
último 15% del texto) y la fila queda marcada con metodo_corte =
"aproximado_gemini" para poder auditarla o excluirla fácilmente.
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sentido_extractor import RESOLUTIVO_HEADER_PATTERN, FUNDADO_TRANSITION_PATTERN

OUTPUT_FOLDER = Path("documents/output")
CSV_INVENTARIO = Path("data/dataset_sentencias.csv")
SENTIDO_GEMINI = Path("data/sentido_gemini.json")
DATASET_MODELO = Path("data/dataset_modelo.parquet")

CONFIANZA_ACEPTABLE = {"alta", "media"}

# Las 13 clases oficiales del modelo. Cualquier valor fuera de esta lista
# (typo, variante en plural, alucinacion del modelo, etc.) se descarta,
# aunque venga con confianza alta -- preferimos perder el documento a
# introducir una 14a clase espuria.
SENTIDOS_VALIDOS = {
    "confirma", "revoca", "modifica",
    "existencia_infraccion", "inexistencia_infraccion",
    "declara_fundado", "declara_infundado", "declara_improcedente",
    "sobresee",
    "cumplida", "incumplida", "parcialmente_cumplida",
    "desecha",
}
RECORTE_APROXIMADO_FRACCION = 0.15  # se descarta el último 15% del texto


def pre_resolutivo_text(full_text: str) -> tuple[str, str]:
    """Devuelve (texto_previo_al_resolutivo, metodo_de_corte)."""
    matches = list(RESOLUTIVO_HEADER_PATTERN.finditer(full_text))
    if matches:
        return full_text[: matches[-1].start()], "resuelve_header"

    fallback = list(FUNDADO_TRANSITION_PATTERN.finditer(full_text))
    if fallback:
        return full_text[: fallback[-1].start()], "fundado_transition"

    return full_text, "sin_corte_texto_completo"


def cargar_sentido_gemini() -> dict[str, dict]:
    """Devuelve {json_filename: {sentido, marcador_resolutivo}} solo para
    los casos con confianza aceptable (alta o media)."""
    if not SENTIDO_GEMINI.exists():
        return {}
    registros = json.loads(SENTIDO_GEMINI.read_text(encoding="utf-8"))
    usable = {}
    descartados_por_etiqueta_invalida = []
    for r in registros:
        if "error" in r:
            continue
        if r.get("sentido") == "no_aplica" or r.get("confianza") not in CONFIANZA_ACEPTABLE:
            continue
        if r.get("sentido") not in SENTIDOS_VALIDOS:
            descartados_por_etiqueta_invalida.append((r["json_filename"], r.get("sentido")))
            continue
        usable[r["json_filename"]] = r
    if descartados_por_etiqueta_invalida:
        print(f"AVISO: se descartaron {len(descartados_por_etiqueta_invalida)} etiqueta(s) de Gemini fuera de las 13 clases validas:")
        for nombre, etiqueta in descartados_por_etiqueta_invalida:
            print(f"   {nombre} -> '{etiqueta}'")
    return usable


def pre_resolutivo_text_con_marcador(full_text: str, marcador: str) -> tuple[str, str]:
    """Igual que pre_resolutivo_text(), pero para documentos sin ancla
    regex: intenta cortar en el marcador verbatim que dio Gemini; si no
    aparece tal cual en el texto, corta de forma aproximada (último 15%)."""
    marcador = (marcador or "").strip()
    if marcador:
        idx = full_text.find(marcador)
        if idx != -1:
            return full_text[:idx], "marcador_gemini"

    corte = int(len(full_text) * (1 - RECORTE_APROXIMADO_FRACCION))
    return full_text[:corte], "aproximado_gemini"


def main() -> None:
    df = pd.read_csv(CSV_INVENTARIO)
    print(f"Total filas en el inventario: {len(df)}")

    df = df[df["es_acuerdo_plenario"] == False]

    sentido_gemini = cargar_sentido_gemini()
    print(f"Etiquetas de Gemini disponibles (confianza alta/media): {len(sentido_gemini)}")

    con_sentido = df["sentido"] != "no_detectado"
    con_gemini = df["json_filename"].isin(sentido_gemini.keys())
    df = df[con_sentido | con_gemini]
    print(f"Filas tras excluir Acuerdos Plenarios y no_detectado sin etiqueta de Gemini: {len(df)}")

    rows = []
    missing = 0
    for _, row in df.iterrows():
        json_path = OUTPUT_FOLDER / row["json_filename"]
        if not json_path.exists():
            missing += 1
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        full_text = data.get("extracted_text_clean", "") or ""

        gemini_info = sentido_gemini.get(row["json_filename"])
        if gemini_info is not None:
            # sentido regex == no_detectado pero Gemini sí lo clasificó
            sentido = gemini_info["sentido"]
            texto_insumo, metodo_corte = pre_resolutivo_text_con_marcador(
                full_text, gemini_info.get("marcador_resolutivo", "")
            )
        else:
            sentido = row["sentido"]
            texto_insumo, metodo_corte = pre_resolutivo_text(full_text)

        rows.append(
            {
                "json_filename": row["json_filename"],
                "category": row["category"],
                "expediente": row["expediente"],
                "sentido": sentido,
                "texto_insumo": texto_insumo.strip(),
                "metodo_corte": metodo_corte,
                "n_chars_insumo": len(texto_insumo.strip()),
            }
        )

    if missing:
        print(f"AVISO: {missing} archivos JSON referenciados en el CSV no se encontraron.")

    out = pd.DataFrame(rows)
    out = out[out["n_chars_insumo"] > 0]

    antes = len(out)
    out = out.drop_duplicates(subset="texto_insumo", keep="first")
    duplicados = antes - len(out)
    if duplicados:
        print(
            f"\nAVISO: se descartaron {duplicados} documento(s) duplicado(s) "
            "(texto idéntico al de otro archivo ya incluido, probablemente "
            "el mismo expediente subido dos veces con nombres distintos)."
        )

    print(f"\nFilas finales con texto insumo no vacío (sin duplicados): {len(out)}")
    print("\nMétodo de corte usado:")
    print(out["metodo_corte"].value_counts())
    print("\nDistribución de clases (sentido) en el dataset de modelado:")
    print(out["sentido"].value_counts())

    DATASET_MODELO.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(DATASET_MODELO, index=False)
    print(f"\nGuardado en {DATASET_MODELO}")


if __name__ == "__main__":
    main()
