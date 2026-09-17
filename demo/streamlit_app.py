"""Interfaz local (Streamlit) para la demo de tesis: evaluación de
viabilidad de un resolutivo propuesto.

Envuelve `src/evaluar_documento.py` -- no reimplementa nada del motor,
solo agrega una capa visual para presentar el resultado en la defensa
sin depender de la terminal ni de internet (una vez que el modelo de
embeddings ya está descargado localmente, ver LEEME.md de esta carpeta).

Uso:
    streamlit run demo/streamlit_app.py
"""

import json
import sys
import tempfile
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from evaluar_documento import evaluar_documento  # noqa: E402
from detectar_temas import OTRO, OTROS_ESPECIFICOS, TEMAS_PANEL_CON_BUCKETS  # noqa: E402

TEMAS_CSV = Path(__file__).resolve().parent.parent / "data" / "temas_por_documento.csv"
NUCLEO_JSON = Path(__file__).resolve().parent.parent / "data" / "nucleo_temas_actual.json"


@st.cache_data
def _cargar_etiquetas_panel() -> list[str]:
    """Lee el núcleo dinámico que guardó scripts/calcular_temas.py (los N
    temas más frecuentes del corpus real en el momento en que se corrió),
    para que el panel grafique EXACTAMENTE ese mismo núcleo -- no una
    lista fija en el código, que podría quedar desactualizada. Si el
    archivo todavía no existe (repo recién clonado, antes de correr
    calcular_temas.py por primera vez), usa la lista fija de
    detectar_temas.py como respaldo para que la app no truene."""
    if NUCLEO_JSON.exists():
        try:
            data = json.loads(NUCLEO_JSON.read_text(encoding="utf-8"))
            nucleo = [item["tema"] for item in data["nucleo"]]
            return nucleo + [OTROS_ESPECIFICOS, OTRO]
        except Exception:
            pass
    return list(TEMAS_PANEL_CON_BUCKETS)


@st.cache_data
def _cargar_temas_por_documento() -> pd.DataFrame | None:
    """Carga el CSV precalculado por scripts/calcular_temas.py. Si todavía
    no se generó (o el repo no lo tiene), el panel de temas y las
    etiquetas de tema en precedentes simplemente no se muestran --
    el resto de la app sigue funcionando igual."""
    if not TEMAS_CSV.exists():
        return None
    df = pd.read_csv(TEMAS_CSV, dtype={"year": str})
    df["temas"] = df["temas"].fillna("")
    if "temas_extendidos" not in df.columns:
        # CSV generado antes de agregar temas_extendidos (versión anterior de
        # calcular_temas.py) -- usa los mismos 6 temas como respaldo, en vez
        # de fallar. Recomendado: volver a correr calcular_temas.py.
        df["temas_extendidos"] = df["temas"]
    df["temas_extendidos"] = df["temas_extendidos"].fillna("")
    if "sentido" in df.columns:
        df["sentido"] = df["sentido"].fillna("no_detectado")
    return df


_TEMAS_DF = _cargar_temas_por_documento()
_TEMAS_POR_EXPEDIENTE = (
    dict(zip(_TEMAS_DF["expediente"], _TEMAS_DF["temas_extendidos"]))
    if _TEMAS_DF is not None
    else {}
)

st.set_page_config(
    page_title="Evaluación de viabilidad de resolutivos — TFM",
    layout="wide",
)

st.title("Evaluación de viabilidad de un resolutivo propuesto")
st.caption(
    "Sube un proyecto de sentencia (PDF o DOCX) con un resolutivo ya "
    "redactado. La herramienta NO lo corrige: compara lo que propones "
    "contra lo que el modelo, entrenado con el corpus histórico del "
    "TEEN, predice a partir de los antecedentes y consideraciones "
    "-- y cita los precedentes más relevantes a favor y en contra, con "
    "su resumen y un link al documento original cuando están disponibles."
)

with st.sidebar:
    st.header("Sobre el modelo")
    st.write(
        "Clasificador entrenado sobre **856 sentencias históricas** del "
        "TEEN, con un modelo de Regresión Logística L2. Exactitud de "
        "referencia: **59.6%** · F1 macro: **0.402** · F1 ponderado: "
        "**0.594** (validación cruzada, 5 particiones)."
    )
    st.write(
        "Los precedentes se buscan por similitud de **asunto**; en "
        "ausencia de asunto, por similitud de **sentido**. Cada "
        "precedente muestra un resumen y un enlace a la página web "
        "donde se localiza el documento."
    )


# Número fijo de muestras para el cálculo interno de LIME (antes expuesto
# como control en la barra lateral). Se usa el valor más bajo probado
# (más rápido) ya que la explicación no se muestra en la interfaz.
_NUM_SAMPLES_LIME = 20

# Catálogo de sentidos que puede detectar sentido_extractor.py / predecir el
# modelo. Se usa para el glosario que se muestra junto al sentido propuesto
# y al predicho -- solo se listan ahí los que de verdad aparecen en el
# documento evaluado, no el catálogo completo, para no saturar la pantalla.
GLOSARIO_SENTIDOS = {
    "confirma": "Se confirma la resolución impugnada; queda sin cambios.",
    "revoca": "Se revoca la resolución impugnada; queda sin efectos.",
    "modifica": "Se modifica parcialmente la resolución impugnada.",
    "desecha": "No se admite el recurso, sin resolver el fondo del asunto.",
    "sobresee": "Se da por terminado el procedimiento sin resolver el fondo.",
    "declara_improcedente": "Se declara improcedente el recurso; no cumple los requisitos para resolverse en el fondo.",
    "declara_fundado": "Los agravios planteados sí acreditaron la violación alegada.",
    "declara_infundado": "Los agravios planteados no acreditaron la violación alegada.",
    "existencia_infraccion": "Se determina que sí existió la infracción denunciada.",
    "inexistencia_infraccion": "Se determina que no existió la infracción denunciada.",
    "cumplida": "La sentencia o resolución previa se cumplió en sus términos.",
    "incumplida": "La sentencia o resolución previa no se cumplió.",
    "parcialmente_cumplida": "La sentencia o resolución previa se cumplió solo en parte.",
    "queda_sin_materia": "El procedimiento perdió su objeto antes de resolverse el fondo.",
    "no_detectado": "No se pudo determinar el sentido automáticamente.",
}


def _formatear_sentido(valor: str) -> str:
    """Solo estética para mostrar al usuario: 'declara_infundado' ->
    'Declara infundado'. El valor interno (con guion bajo, minúsculas)
    no cambia -- esto es puramente de presentación."""
    return valor.replace("_", " ").capitalize()


def _mostrar_precedente(precedente: dict, mostrar_sentido: bool = False) -> None:
    """Renderiza un precedente como un expander: etiqueta corta con
    expediente/similitud siempre visible, y al abrirlo, el link al
    documento original (si se pudo reconstruir) y el resumen
    hechos/agravios/resolución (si ya fue generado)."""
    etiqueta = precedente["expediente"]
    if mostrar_sentido:
        etiqueta += f" → resolutivo: {precedente['sentido']}"
    etiqueta += f" ({precedente['similitud']:.0%} similitud)"

    with st.expander(etiqueta):
        if precedente.get("url_original"):
            st.markdown(f"[Ver documento original en trieen.mx]({precedente['url_original']})")
        else:
            st.caption("No se encontró el link al documento original.")

        resumen = precedente.get("resumen")
        if resumen:
            st.markdown(f"**Hechos:** {resumen.get('hechos', '—')}")
            st.markdown(f"**Agravios:** {resumen.get('agravios', '—')}")
            st.markdown(f"**Resolución:** {resumen.get('resolucion', '—')}")
        else:
            st.caption("Este documento todavía no tiene resumen generado.")

        temas = _TEMAS_POR_EXPEDIENTE.get(precedente["expediente"], "")
        if temas:
            st.markdown(f"**Temas:** {temas}")


tab_evaluar, tab_panel = st.tabs(["Evaluar documento", "Generales"])

with tab_panel:
    if _TEMAS_DF is None:
        st.info(
            "Todavía no se generó data/temas_por_documento.csv -- corre "
            "`python scripts/calcular_temas.py` desde la raíz del repo y "
            "vuelve a abrir esta pestaña."
        )
    else:
        st.caption(
            "Corpus histórico extraído (se excluyen Acuerdos Plenarios, "
            "igual que en el resto del proyecto)."
        )

        _PALETA_TEEN = [
            "#3A3A3A", "#806104", "#B99525", "#D4B96A", "#E8DDC7",
        ]
        _COLOR_TEEN_BARRAS = "#806104"  # dorado oscuro del TEEN, para las barras de una sola serie

        def _barras_horizontales(serie: pd.Series, color: str, alto: int | None = None):
            datos = serie.reset_index()
            datos.columns = ["categoria", "valor"]
            if alto is None:
                alto = max(28 * len(datos), 120)
            chart = (
                alt.Chart(datos)
                .mark_bar(color=color)
                .encode(
                    x=alt.X("valor:Q", title=None, axis=alt.Axis(labelFontSize=13)),
                    y=alt.Y(
                        "categoria:N", sort=None, title=None,
                        axis=alt.Axis(labelFontSize=14, labelLimit=280),
                    ),
                )
                .properties(height=alto)
            )
            st.altair_chart(chart, width="stretch")

        def _barras_verticales(serie: pd.Series, color: str):
            datos = serie.reset_index()
            datos.columns = ["categoria", "valor"]
            v_min, v_max = datos["valor"].min(), datos["valor"].max()
            datos["etiqueta"] = datos["valor"].apply(
                lambda v: str(v) if v in (v_min, v_max) else ""
            )
            barras = alt.Chart(datos).mark_bar(color=color).encode(
                x=alt.X("categoria:N", sort=None, title=None, axis=alt.Axis(labelFontSize=13)),
                y=alt.Y("valor:Q", title=None, axis=alt.Axis(labelFontSize=13)),
            )
            texto = alt.Chart(datos).mark_text(
                dy=-8, fontSize=13, color="#808495",
            ).encode(
                x=alt.X("categoria:N", sort=None),
                y=alt.Y("valor:Q"),
                text="etiqueta:N",
            )
            st.altair_chart(barras + texto, width="stretch")

        col_categoria, col_anio = st.columns([2, 3])

        def _es_oscuro(color_hex: str) -> bool:
            color_hex = color_hex.lstrip("#")
            r, g, b = (int(color_hex[i : i + 2], 16) for i in (0, 2, 4))
            luminancia = 0.299 * r + 0.587 * g + 0.114 * b
            return luminancia < 150

        with col_categoria:
            st.markdown("#### Sentencias por categoría de procedimiento")
            conteo_categoria = _TEMAS_DF["category"].value_counts()
            colores_dona = _PALETA_TEEN[: len(conteo_categoria)]
            fig_categoria, ax_categoria = plt.subplots(figsize=(4, 4))
            _, etiquetas_dona, porcentajes_dona = ax_categoria.pie(
                conteo_categoria.values,
                labels=conteo_categoria.index,
                autopct="%1.0f%%",
                startangle=90,
                colors=colores_dona,
                wedgeprops=dict(width=0.45, edgecolor="white", linewidth=0.8),
                pctdistance=0.78,
                textprops={"fontsize": 7, "color": "#808495"},
            )
            for texto_pct, color_cuna in zip(porcentajes_dona, colores_dona):
                texto_pct.set_color("white" if _es_oscuro(color_cuna) else "#3a3a3a")
                texto_pct.set_fontweight("bold")
            ax_categoria.text(
                0, 0, f"Total\n{conteo_categoria.sum()}",
                ha="center", va="center", fontsize=11, fontweight="bold",
            )
            fig_categoria.patch.set_alpha(0)
            st.pyplot(fig_categoria)

        with col_anio:
            st.markdown("#### Sentencias por año")
            conteo_anio = _TEMAS_DF["year"].value_counts().sort_index()
            _barras_verticales(conteo_anio, _COLOR_TEEN_BARRAS)

        col_tema, col_sentido = st.columns(2)

        columna_tema = "temas_panel" if "temas_panel" in _TEMAS_DF.columns else "temas"
        etiquetas_panel = _cargar_etiquetas_panel()
        n_nucleo = len(etiquetas_panel) - 2  # sin contar los 2 buckets finales
        conteo_temas = pd.Series(
            {
                tema: _TEMAS_DF[columna_tema].str.contains(tema, regex=False).sum()
                for tema in etiquetas_panel
            }
        ).sort_values(ascending=True)
        conteo_sentido = _TEMAS_DF["sentido"].value_counts().sort_values(ascending=True)
        conteo_sentido.index = conteo_sentido.index.str.replace("_", " ").str.capitalize()
        alto_comun = max(28 * len(conteo_temas), 28 * len(conteo_sentido), 120)

        with col_tema:
            st.markdown("#### Sentencias por tema")
            _barras_horizontales(conteo_temas, _COLOR_TEEN_BARRAS, alto=alto_comun)
            st.caption(f"Los {n_nucleo} temas más frecuentes del corpus.")

        with col_sentido:
            st.markdown("#### Sentido de la resolución")
            _barras_horizontales(conteo_sentido, _COLOR_TEEN_BARRAS, alto=alto_comun)
            st.caption("Las categorías varían según el tipo de procedimiento.")

        with st.expander("Ver detalle por documento"):
            _detalle = _TEMAS_DF[
                ["expediente", "category", "year", "sentido", "temas_extendidos"]
            ].rename(columns={"temas_extendidos": "tema"})
            st.dataframe(
                _detalle,
                width="stretch",
                hide_index=True,
            )

with tab_evaluar:
    archivo = st.file_uploader(
        "Proyecto de sentencia",
        type=["pdf", "docx"],
    )

    evaluar = st.button("Evaluar documento", type="primary", disabled=archivo is None)

    if evaluar and archivo is not None:
        suffix = Path(archivo.name).suffix.lower()
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(archivo.getvalue())
            tmp_path = Path(tmp.name)

        try:
            with st.spinner(
                "Extrayendo texto, corriendo el modelo y buscando "
                "precedentes... puede tardar hasta un minuto."
            ):
                resultado = evaluar_documento(
                    tmp_path,
                    num_samples=_NUM_SAMPLES_LIME,
                    # tmp_path tiene un nombre genérico (tempfile.NamedTemporaryFile
                    # lo renombra) -- se pasa el nombre ORIGINAL del archivo subido
                    # para que la detección automática de categoría de
                    # procedimiento tenga la mejor pista posible.
                    nombre_para_categoria=archivo.name,
                )
        except ValueError as error:
            st.error(str(error))
            resultado = None
        finally:
            tmp_path.unlink(missing_ok=True)

        if resultado is not None:
            st.subheader(f"Resultado: {resultado['archivo']}")

            with st.expander("Vista previa", expanded=True):
                st.write(resultado["vista_previa_texto"])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Sentido PROPUESTO (resolutivo)", _formatear_sentido(resultado["sentido_propuesto"]))
                if resultado["sentido_propuesto_evidencia"]:
                    st.caption(f"Evidencia: \"{resultado['sentido_propuesto_evidencia']}\"")
            with col2:
                st.metric(
                    "Sentido PREDICHO por el modelo",
                    _formatear_sentido(resultado["sentido_predicho"]),
                    f"{resultado['confianza_prediccion']:.0%} confianza",
                )

            coincide = resultado["coincide"]
            if coincide is True:
                st.success(resultado["mensaje_comparacion"])
            elif coincide is False:
                st.warning(resultado["mensaje_comparacion"])
            else:
                st.info(resultado["mensaje_comparacion"])

            sentidos_del_documento = {
                resultado["sentido_propuesto"],
                resultado["sentido_predicho"],
            }
            glosario_aplicable = {
                s: GLOSARIO_SENTIDOS[s]
                for s in sentidos_del_documento
                if s in GLOSARIO_SENTIDOS
            }
            if glosario_aplicable:
                st.markdown("**¿Qué significa cada sentido?**")
                for sentido, significado in glosario_aplicable.items():
                    st.write(f"- **{_formatear_sentido(sentido)}**: {significado}")

            st.markdown("### Precedentes relevantes")
            if resultado.get("temas_documento"):
                st.caption("Temas detectados en este documento: " + ", ".join(resultado["temas_documento"]))
            else:
                st.caption(
                    "No se detectó ningún tema del catálogo en este documento -- "
                    "los precedentes se ordenan solo por sentido y similitud, sin "
                    "reordenar por tema."
                )
            col_favor, col_otros = st.columns(2)
            with col_favor:
                st.markdown("**A favor** (mismo resolutivo -- sustentan la predicción)")
                if resultado["precedentes_a_favor"]:
                    for p in resultado["precedentes_a_favor"]:
                        _mostrar_precedente(p)
                else:
                    st.write("Ninguno con similitud suficientemente alta.")
            with col_otros:
                st.markdown("**Otros** (mismo asunto, resolutivo distinto -- no sustentan)")
                if resultado["precedentes_otros"]:
                    for p in resultado["precedentes_otros"]:
                        _mostrar_precedente(p, mostrar_sentido=True)
                else:
                    st.write("Ninguno con similitud suficientemente alta.")

            st.divider()
            st.caption(
                "Esta herramienta evalúa la viabilidad estadística del "
                "resolutivo propuesto frente al corpus histórico -- no "
                "sustituye el criterio jurídico ni corrige el resolutivo."
            )
