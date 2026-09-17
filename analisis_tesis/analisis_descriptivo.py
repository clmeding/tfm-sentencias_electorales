"""Analisis descriptivo del corpus para el escrito del TFM.

Este script vive FUERA de scripts/ a proposito: no es parte del pipeline
de produccion (build_dataset.py, entrenar_modelo.py, etc.), es una capa de
reporte que solo lee los CSV/JSON que el pipeline ya genera y produce
graficas + un resumen en texto listos para pegar/citar en el documento del
TFM. No modifica ni recalcula nada del pipeline real.

Cuenta la historia completa "de lo general a lo particular" que pide la
guia del TFM para la fase de analisis descriptivo:

    1. Universo de partida: inventario completo ya extraido y limpio
       (data/dataset_sentencias.csv) -- este archivo ya excluye, desde
       build_dataset.py, los documentos con OCR ilegible.
    2. Exclusion de Acuerdos Plenarios (no tienen un resolutivo de fondo
       clasificable).
    3. De los elegibles, cuantos ya tienen sentido detectado por regla
       (regex sobre RESUELVE) y cuantos se resolvieron via Gemini
       (confianza alta/media) para llegar al corpus final de modelado.
    4. Composicion del corpus final: por categoria de procedimiento, por
       ano, por sentido (con el desbalance de clases) y por tema.
    5. Estadisticos descriptivos de longitud de texto del corpus completo.

Uso:
    python analisis_tesis/analisis_descriptivo.py [--repo-root RUTA]

Por defecto asume que este archivo vive en <repo>/analisis_tesis/ y que
<repo>/data/ y <repo>/scripts/ existen como hermanos. Si lo mueves a otro
lugar, pasa --repo-root apuntando a la raiz del repo del pipeline.

Entradas esperadas (todas ya generadas por el pipeline real):
    data/dataset_sentencias.csv     (build_dataset.py)
    data/temas_por_documento.csv    (scripts/calcular_temas.py)
    data/sentido_gemini.json        (scripts/detectar_sentido_gemini.py)
        -- si no existe, se usa data/sentido_gemini_v1_backup.json como
           respaldo y se avisa en el resumen (ver seccion de limitaciones
           mas abajo).

Salida (todo dentro de analisis_tesis/output/):
    01_embudo_corpus.png
    02_categoria_procedimiento.png
    03_documentos_por_anio.png
    04_sentido_resolucion.png
    05_origen_etiqueta_sentido.png
    06_temas_nucleo.png
    07_longitud_texto.png
    resumen_descriptivo.md   -- todas las cifras + tablas en texto, con
                                 las mismas notas de limitaciones que se
                                 imprimen en pantalla.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

# ---------------------------------------------------------------------------
# Paleta y estilo
# ---------------------------------------------------------------------------
COLOR_TEAL = "#20808D"
COLOR_TERRA = "#A84B2F"
COLOR_DARK_TEAL = "#1B474D"
COLOR_LIGHT_CYAN = "#BCE2E7"
COLOR_MAUVE = "#944454"
COLOR_GOLD = "#FFC553"
COLOR_OLIVE = "#848456"
COLOR_BROWN = "#6E522B"
COLOR_TEXT = "#28251D"
COLOR_MUTED = "#7A7974"
PALETA_CATEGORICA = [
    COLOR_TEAL, COLOR_TERRA, COLOR_DARK_TEAL, COLOR_LIGHT_CYAN,
    COLOR_MAUVE, COLOR_GOLD, COLOR_OLIVE, COLOR_BROWN,
]

plt.rcParams.update({
    "font.size": 11,
    "text.color": COLOR_TEXT,
    "axes.edgecolor": COLOR_MUTED,
    "axes.labelcolor": COLOR_TEXT,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

SENTIDOS_VALIDOS = {
    "confirma", "revoca", "modifica",
    "existencia_infraccion", "inexistencia_infraccion",
    "declara_fundado", "declara_infundado", "declara_improcedente",
    "sobresee",
    "cumplida", "incumplida", "parcialmente_cumplida",
    "desecha",
}
CONFIANZA_ACEPTABLE = {"alta", "media"}


def cargar_gemini_usable(repo_root: Path) -> tuple[dict, str]:
    """Devuelve ({json_filename: registro}, nombre_del_archivo_usado).

    Prefiere data/sentido_gemini.json (el que espera build_dataset_modelo.py
    en produccion); si no existe, cae de vuelta a
    data/sentido_gemini_v1_backup.json y lo reporta como limitacion.
    """
    principal = repo_root / "data" / "sentido_gemini.json"
    respaldo = repo_root / "data" / "sentido_gemini_v1_backup.json"

    ruta = principal if principal.exists() else respaldo
    if not ruta.exists():
        return {}, "(ninguno encontrado)"

    registros = json.loads(ruta.read_text(encoding="utf-8"))
    usable = {}
    for r in registros:
        if "error" in r:
            continue
        if r.get("sentido") == "no_aplica" or r.get("confianza") not in CONFIANZA_ACEPTABLE:
            continue
        if r.get("sentido") not in SENTIDOS_VALIDOS:
            continue
        usable[r["json_filename"]] = r
    return usable, ruta.name


def construir_embudo(df: pd.DataFrame, gemini_usable: dict) -> dict:
    total_inventario = len(df)
    df_elegible = df[df["es_acuerdo_plenario"] == False]  # noqa: E712
    n_elegible = len(df_elegible)
    n_plenarios = total_inventario - n_elegible

    con_regla = df_elegible["sentido"] != "no_detectado"
    con_gemini = df_elegible["json_filename"].isin(gemini_usable.keys())

    n_regla = int(con_regla.sum())
    n_solo_gemini = int((~con_regla & con_gemini).sum())
    n_sin_etiqueta = int((~con_regla & ~con_gemini).sum())
    n_final = n_regla + n_solo_gemini

    return {
        "total_inventario": total_inventario,
        "n_plenarios": n_plenarios,
        "n_elegible": n_elegible,
        "n_regla": n_regla,
        "n_solo_gemini": n_solo_gemini,
        "n_sin_etiqueta": n_sin_etiqueta,
        "n_final": n_final,
    }


def graf_embudo(embudo: dict, out: Path) -> None:
    etapas = [
        ("Inventario\ncompleto", embudo["total_inventario"], COLOR_DARK_TEAL),
        ("Sin Acuerdos\nPlenarios", embudo["n_elegible"], COLOR_TEAL),
        ("Corpus final\nde modelado", embudo["n_final"], COLOR_TERRA),
    ]
    techo = embudo["total_inventario"] * 1.30
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    etiquetas = [e[0] for e in etapas]
    valores = [e[1] for e in etapas]
    colores = [e[2] for e in etapas]
    barras = ax.bar(etiquetas, valores, color=colores, width=0.55, zorder=3)
    for barra, valor in zip(barras, valores):
        ax.text(barra.get_x() + barra.get_width() / 2, valor + techo * 0.015, str(valor),
                 ha="center", va="bottom", fontsize=9, fontweight="bold", color=COLOR_TEXT, zorder=4)
    ax.set_ylim(0, techo)
    ax.set_yticks([])
    ax.set_title("De inventario a corpus de modelado", fontsize=11, fontweight="bold", loc="left")
    y_anotacion = embudo["total_inventario"] * 1.14
    ax.text(
        0.5, y_anotacion, f"-{embudo['n_plenarios']} Acuerdos Plenarios",
        ha="center", va="bottom", fontsize=9, color=COLOR_MUTED,
    )
    ax.text(
        1.5, y_anotacion,
        f"-{embudo['n_sin_etiqueta']} sin sentido detectable", #\n(ni regla ni Gemini alta/media)",
        ha="center", va="bottom", fontsize=9, color=COLOR_MUTED,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def graf_barra_horizontal(serie: pd.Series, titulo: str, out: Path, color=COLOR_TEAL) -> None:
    """Genera una gráfica de barras horizontales a partir de una serie de conteos."""
    serie = serie.dropna().sort_values(ascending=True)

    if serie.empty:
        return

    etiqueta_max = max(len(str(x)) for x in serie.index)
    ancho_fig = max(7.5, 6.3 + etiqueta_max * 0.085)
    fig, ax = plt.subplots(figsize=(ancho_fig, max(2.4, 0.42 * len(serie))))

    barras = ax.barh(serie.index.astype(str), serie.values, color=color)

    max_valor = max(serie.values)
    desplazamiento = max_valor * 0.01 if max_valor else 0.1

    for barra, valor in zip(barras, serie.values):
        ax.text(
            valor + desplazamiento,
            barra.get_y() + barra.get_height() / 2,
            str(int(valor)),
            va="center",
            fontsize=9,
            color=COLOR_TEXT,
        )

    ax.set_xticks([])
    ax.set_title(titulo, fontsize=11, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def graf_barra_vertical(serie: pd.Series, titulo: str, out: Path, color=COLOR_TEAL) -> None:
    """Genera una gráfica de barras verticales a partir de una serie de conteos."""
    serie = serie.dropna()

    if serie.empty:
        return

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    barras = ax.bar(serie.index.astype(str), serie.values, color=color)

    max_valor = max(serie.values)
    desplazamiento = max_valor * 0.015 if max_valor else 0.1

    for barra, valor in zip(barras, serie.values):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            valor + desplazamiento,
            str(int(valor)),
            ha="center",
            fontsize=9,
            color=COLOR_TEXT,
        )

    ax.set_yticks([])
    ax.set_title(titulo, fontsize=11, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def graf_origen_etiqueta(embudo: dict, out: Path) -> None:
    datos = pd.Series({
        "Detectado por regla\n(RESUELVE)": embudo["n_regla"],
        "Asistido por API": embudo["n_solo_gemini"],
        "Sin etiqueta\n(excluido del modelo)": embudo["n_sin_etiqueta"],
    })
    fig, ax = plt.subplots(figsize=(6.5, 3))
    colores = [COLOR_TEAL, COLOR_GOLD, "#D4D1CA"]
    barras = ax.bar(datos.index, datos.values, color=colores)
    for barra, valor in zip(barras, datos.values):
        pct = 100 * valor / datos.sum() if datos.sum() else 0
        ax.text(barra.get_x() + barra.get_width() / 2, valor + max(datos.values) * 0.02,
                 f"{valor} ({pct:.0f}%)", ha="center", fontsize=9, color=COLOR_TEXT)
    ax.set_yticks([])
    ax.set_title("De donde viene la etiqueta de sentido", fontsize=11, fontweight="bold", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=300)
    plt.close(fig)


def graf_histograma_longitud(serie: pd.Series, out: Path) -> None:
    """Genera el histograma de longitud de los documentos."""
    serie = pd.to_numeric(serie, errors="coerce").dropna()

    if serie.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.hist(serie, bins=30, color=COLOR_TEAL, edgecolor="white")

    mediana = serie.median()
    ax.axvline(
        mediana,
        color=COLOR_TERRA,
        linestyle="--",
        linewidth=1.5,
        label=f"Mediana: {mediana:,.0f} caracteres",
    )

    ax.set_xlabel("Longitud del texto extraído (caracteres)")
    ax.set_yticks([])
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(
        "Distribución de longitud de documento (n={})".format(len(serie)),
        fontsize=11,
        fontweight="bold",
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parent.parent,
        help="Raiz del repo del pipeline (donde estan data/, scripts/, documents/).",
    )
    args = parser.parse_args()
    repo_root: Path = args.repo_root

    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    inventario_path = repo_root / "data" / "dataset_sentencias.csv"
    temas_path = repo_root / "data" / "temas_por_documento.csv"

    if not inventario_path.exists():
        raise SystemExit(
            f"No se encontro {inventario_path}. Corre esto desde el repo del "
            "pipeline (o pasa --repo-root)."
        )

    df = pd.read_csv(inventario_path)

    columnas_requeridas = {"es_acuerdo_plenario", "sentido", "json_filename"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise SystemExit(
            "Faltan columnas obligatorias en dataset_sentencias.csv: "
            + ", ".join(sorted(faltantes))
        )

    gemini_usable, archivo_gemini_usado = cargar_gemini_usable(repo_root)
    embudo = construir_embudo(df, gemini_usable)

    limitaciones = []
    if archivo_gemini_usado != "sentido_gemini.json":
        limitaciones.append(
            f"Se uso '{archivo_gemini_usado}' como respaldo porque "
            "data/sentido_gemini.json no esta en el repo. Si tienes el "
            "archivo original localmente, colocalo en data/ y vuelve a "
            "correr este script para reproducir el numero oficial "
            "documentado en CAMBIOS.md (856)."
        )

    # --- Graficas del embudo y origen de etiqueta -------------------------
    graf_embudo(embudo, out_dir / "01_embudo_corpus.png")
    graf_origen_etiqueta(embudo, out_dir / "05_origen_etiqueta_sentido.png")

    # --- Graficas de composicion (usa temas_por_documento.csv si existe) --
    if temas_path.exists():
        temas_df = pd.read_csv(temas_path, dtype={"year": str})
        graf_barra_vertical(
            temas_df["category"].value_counts().sort_index(),
            "Documentos por categoria de procedimiento",
            out_dir / "02_categoria_procedimiento.png",
            color=COLOR_DARK_TEAL
        )
        graf_barra_vertical(
            temas_df["year"].value_counts().sort_index(),
            "Documentos por año",
            out_dir / "03_documentos_por_anio.png",
        )
        sentido_regla = temas_df["sentido"].fillna("no_detectado")
        graf_barra_horizontal(
            sentido_regla.value_counts(),
            "Sentido de la resolucion (deteccion por regla)",
            out_dir / "04_sentido_resolucion.png",
            color=COLOR_MAUVE
        )
        if "temas_panel" in temas_df.columns or "temas" in temas_df.columns:
            col_tema = "temas_panel" if "temas_panel" in temas_df.columns else "temas"
            etiquetas_tema = sorted({
                t.strip()
                for cadena in temas_df[col_tema].dropna()
                for t in str(cadena).split(",")
                if t.strip()
            })
            if etiquetas_tema:
                conteo_temas = pd.Series({
                    tema: temas_df[col_tema].fillna("").str.contains(tema, regex=False).sum()
                    for tema in etiquetas_tema
                })
                graf_barra_horizontal(
                    conteo_temas,
                    "Documentos por tema (nucleo dinamico)",
                    out_dir / "06_temas_nucleo.png",
                    color=COLOR_OLIVE
                )
    else:
        limitaciones.append(
            f"No se encontro {temas_path}: no se generaron las graficas de "
            "categoria/ano/sentido/tema. Corre scripts/calcular_temas.py "
            "primero."
        )

    # --- Longitud de texto (sobre el inventario completo) ------------------
    if "text_length" in df.columns:
        text_length = pd.to_numeric(df["text_length"], errors="coerce").dropna()
        if text_length.empty:
            stats_longitud = None
            limitaciones.append(
                "La columna 'text_length' existe, pero no contiene valores numéricos utilizables."
            )
        else:
            graf_histograma_longitud(
                text_length,
                out_dir / "07_longitud_texto.png",
            )
            stats_longitud = text_length.describe()
    else:
        stats_longitud = None
        limitaciones.append("El inventario no tiene columna 'text_length'.")

    # --- Resumen en texto ---------------------------------------------------
    lineas = []
    lineas.append("# Analisis descriptivo del corpus -- resumen para el escrito\n")
    lineas.append("## 1. Embudo de construccion del corpus\n")
    lineas.append(f"- Inventario completo (ya sin OCR ilegible): **{embudo['total_inventario']} documentos**")
    lineas.append(f"- Acuerdos Plenarios excluidos: **{embudo['n_plenarios']}**")
    lineas.append(f"- Elegibles para clasificacion de sentido: **{embudo['n_elegible']}**")
    lineas.append(f"- Con sentido detectado por regla (RESUELVE): **{embudo['n_regla']}**")
    lineas.append(f"- Resueltos via Gemini (confianza alta/media), no detectados por regla: **{embudo['n_solo_gemini']}**")
    lineas.append(f"- Sin etiqueta utilizable (excluidos del modelo): **{embudo['n_sin_etiqueta']}**")
    lineas.append(f"- **Corpus final de modelado: {embudo['n_final']} documentos**\n")

    if stats_longitud is not None:
        lineas.append("## 2. Longitud de texto (inventario completo)\n")
        lineas.append(f"- Media: {stats_longitud['mean']:,.0f} caracteres")
        lineas.append(f"- Mediana: {stats_longitud['50%']:,.0f} caracteres")
        lineas.append(f"- Minimo: {stats_longitud['min']:,.0f} / Maximo: {stats_longitud['max']:,.0f}\n")

    if limitaciones:
        lineas.append("## 3. Limitaciones / pendientes a revisar\n")
        for lim in limitaciones:
            lineas.append(f"- {lim}")
        lineas.append("")

    (out_dir / "resumen_descriptivo.md").write_text("\n".join(lineas), encoding="utf-8")

    print("\n".join(lineas))
    print(f"\nGraficas y resumen guardados en: {out_dir}")


if __name__ == "__main__":
    main()
