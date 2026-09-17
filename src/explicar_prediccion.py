"""Predice el sentido de un documento nuevo y explica los motivos de la
predicción en lenguaje natural, señalando qué palabras/frases del texto
más influyeron (usando LIME sobre el pipeline embeddings + clasificador,
tratado como caja negra).

Requiere que ya se hayan corrido, en este orden:
    python scripts/build_dataset_modelo.py
    python scripts/generar_embeddings.py          (modelo local, gratuito)
    python scripts/comparar_escenarios_corte.py   (o entrenar_modelo.py)

El embedding de un documento NUEVO se genera con el mismo modelo local
(sentence-transformers) que scripts/generar_embeddings.py usa para el
corpus, para que ambos queden en el mismo espacio vectorial (768
dimensiones). No requiere API key ni conexión a internet: corre
completamente offline, lo cual es importante porque LIME llama a esta
función una vez por cada muestra perturbada (num_samples, 20 a 150 según
el slider del demo) -- si dependiera de una API de pago, cada evaluación
dispararía decenas o cientos de llamadas. Con el modelo local no hay
costo ni riesgo de que falte conexión el día de la defensa.

Uso como script (ejemplo con 3 documentos al azar del propio dataset):
    python src/explicar_prediccion.py

Uso como módulo:
    from explicar_prediccion import predecir_y_explicar
    resultado = predecir_y_explicar(texto_del_documento)
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lime.lime_text import LimeTextExplainer
from scipy import sparse
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from detectar_temas import OTRO, detectar_temas_extendido

# Mismos parámetros que scripts/generar_embeddings.py -- si algo cambia
# ahí (modelo, tamaño de fragmento), hay que replicar el cambio aquí
# para que el documento nuevo caiga en el mismo espacio vectorial que
# el corpus de referencia.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
WORDS_PER_CHUNK = 90

MODELO_DIR = Path(__file__).resolve().parent.parent / "modelo"

# Cuántos precedentes mostrar como máximo por lado (a favor / otros).
# Esto es una decisión de despliegue (para no saturar la pantalla en una
# demo), NO el criterio de qué cuenta como "realmente similar" -- ese
# criterio sale de PERCENTIL_SIMILITUD, calculado sobre la propia
# distribución de similitudes del corpus (ver _umbral_similitud()).
MAX_PRECEDENTES_POR_LADO = 4

# Percentil de la distribución de similitudes coseno entre TODOS los pares
# de documentos del corpus de entrenamiento. Solo se consideran
# "precedentes" los documentos cuya similitud con el documento evaluado
# caiga en este percentil o más alto -- es decir, similitud inusualmente
# alta para este corpus específico (que comparte mucho lenguaje repetido
# de formato/boilerplate entre todas las sentencias), no un umbral fijo
# elegido a ojo.
PERCENTIL_SIMILITUD = 90

_embedder = None
_clf = None
_scaler = None
_label_encoder = None
_corpus_labels_df = None

# Espacio vectorial SEPARADO para precedentes (ver scripts/generar_precedentes_tfidf.py
# para la justificación completa). El embedding de documento completo (arriba) sigue
# siendo el único usado para CLASIFICAR el sentido -- esto solo cambia cómo se buscan
# "sentencias parecidas" para mostrar en la explicación.
_tfidf_vectorizer = None
_tfidf_matriz_corpus = None
_similarity_threshold_cache_tfidf: float | None = None

# Tabla de temas (materia) por documento del corpus, precalculada por
# scripts/calcular_temas.py (regex, ver src/detectar_temas.py). Se usa SOLO
# como señal adicional para ordenar precedentes ya seleccionados por TF-IDF
# (ver `usar_temas` en precedentes_relevantes) -- opcional y con reserva
# explícita si el archivo no existe, para no romper el flujo actual.
TEMAS_CSV = Path(__file__).resolve().parent.parent / "data" / "temas_por_documento.csv"
_temas_por_json: dict[str, set[str]] | None = None


def _cargar_modelo() -> None:
    global _clf, _scaler, _label_encoder
    if _clf is None:
        _clf = joblib.load(MODELO_DIR / "clasificador.joblib")
        _scaler = joblib.load(MODELO_DIR / "scaler.joblib")
        _label_encoder = joblib.load(MODELO_DIR / "label_encoder.joblib")


def _cargar_corpus_referencia() -> None:
    """Carga las etiquetas (sentido/categoría/expediente) de los
    documentos de entrenamiento, usadas como base para buscar precedentes
    similares (documentos ya resueltos con el mismo sentido predicho).

    Nota: ya NO carga embeddings_documentos.npy aquí -- ese archivo sigue
    siendo necesario para ENTRENAR el clasificador (scripts/comparar_
    escenarios_corte.py / entrenar_modelo.py), pero la búsqueda de
    precedentes usa el espacio TF-IDF separado (ver
    _cargar_tfidf_precedentes()) desde el cambio que corrige que los
    precedentes compartieran solo lenguaje de formato en vez del asunto
    real del caso.
    """
    global _corpus_labels_df
    if _corpus_labels_df is None:
        _corpus_labels_df = pd.read_parquet(MODELO_DIR / "dataset_modelo_labels.parquet")


def _cargar_tfidf_precedentes() -> None:
    """Carga el vectorizador TF-IDF y la matriz del corpus usados SOLO para
    buscar precedentes (ver scripts/generar_precedentes_tfidf.py). Requiere
    haber corrido ese script una vez -- si no existen los archivos, se avisa
    con un error claro en vez de fallar de forma confusa más adelante."""
    global _tfidf_vectorizer, _tfidf_matriz_corpus
    if _tfidf_vectorizer is None:
        ruta_vectorizador = MODELO_DIR / "tfidf_vectorizer.joblib"
        ruta_matriz = MODELO_DIR / "tfidf_matriz_corpus.npz"
        if not ruta_vectorizador.exists() or not ruta_matriz.exists():
            raise FileNotFoundError(
                "Faltan los archivos de precedentes TF-IDF "
                f"('{ruta_vectorizador.name}', '{ruta_matriz.name}') en "
                f"'{MODELO_DIR}'. Corre primero: "
                "python scripts/generar_precedentes_tfidf.py"
            )
        _tfidf_vectorizer = joblib.load(ruta_vectorizador)
        _tfidf_matriz_corpus = sparse.load_npz(ruta_matriz)


def _umbral_similitud_tfidf(percentil: float = PERCENTIL_SIMILITUD) -> float:
    """Igual que _umbral_similitud(), pero calculado sobre la distribución
    de similitudes coseno TF-IDF del corpus (el espacio vectorial usado para
    precedentes). Se cachea por separado porque es un espacio distinto."""
    global _similarity_threshold_cache_tfidf

    if _similarity_threshold_cache_tfidf is not None:
        return _similarity_threshold_cache_tfidf

    _cargar_tfidf_precedentes()

    matriz_similitud = cosine_similarity(_tfidf_matriz_corpus)
    n = matriz_similitud.shape[0]
    pares_sin_diagonal = matriz_similitud[~np.eye(n, dtype=bool)]

    _similarity_threshold_cache_tfidf = float(np.percentile(pares_sin_diagonal, percentil))
    return _similarity_threshold_cache_tfidf


def _vector_tfidf_documento(texto: str) -> sparse.csr_matrix:
    """Vectoriza un documento NUEVO con el mismo vectorizador TF-IDF
    ajustado sobre el corpus -- garantiza que caiga en el mismo espacio,
    igual sea un documento del corpus o uno recién evaluado."""
    _cargar_tfidf_precedentes()
    return _tfidf_vectorizer.transform([texto])


def _cargar_temas_corpus() -> dict[str, set[str]]:
    """Carga data/temas_por_documento.csv (generado por scripts/calcular_temas.py)
    y lo indexa por json_filename -> conjunto de temas detectados (sin
    contar la etiqueta 'Otro / no identificado', que no cuenta como
    coincidencia real de tema).

    Si el archivo no existe todavia (repo sin correr calcular_temas.py),
    devuelve un dict vacio -- precedentes_relevantes() simplemente se
    comporta como antes (sin senal de tema) en vez de fallar.
    """
    global _temas_por_json
    if _temas_por_json is not None:
        return _temas_por_json

    if not TEMAS_CSV.exists():
        _temas_por_json = {}
        return _temas_por_json

    tabla = pd.read_csv(TEMAS_CSV)
    _temas_por_json = {}
    for _, fila in tabla.iterrows():
        crudo = str(fila.get("temas_extendidos", "") or "")
        temas = {t.strip() for t in crudo.split(",") if t.strip() and t.strip() != OTRO}
        _temas_por_json[fila["json_filename"]] = temas
    return _temas_por_json


def _etiqueta_expediente(fila) -> str:
    """Devuelve el número de expediente para mostrar en un precedente.

    24 de los 566 documentos del corpus tienen el campo 'expediente' vacío
    en los metadatos (dato faltante en el origen, no un error de este
    módulo). En esos casos se usa el nombre de archivo (sin extensión) como
    respaldo, para no mostrar 'nan' en pantalla.
    """
    expediente = fila.get("expediente")
    if isinstance(expediente, str) and expediente.strip():
        return expediente
    return Path(fila["json_filename"]).stem


def precedentes_relevantes(
    texto: str,
    sentido_predicho: str,
    max_por_lado: int = MAX_PRECEDENTES_POR_LADO,
    excluir_json_filename: str | None = None,
    usar_temas: bool = True,
) -> dict[str, list[dict]]:
    """Busca, entre los documentos ya etiquetados del corpus, los casos
    cuyo ASUNTO es más parecido al del documento evaluado -- usando
    similitud coseno en el espacio TF-IDF (ver
    scripts/generar_precedentes_tfidf.py), NO el embedding semántico que
    usa el clasificador. Este cambio de espacio vectorial es intencional:
    el embedding de documento completo agrupa por lenguaje de
    formato/procedimiento (compartido por todas las sentencias de este
    tribunal) en vez de por el asunto de fondo del caso; TF-IDF, al
    bajarle peso automáticamente a las palabras comunes entre documentos
    y subírselo a las distintivas de cada caso, es mucho más discriminativo
    para este propósito (validado con muestra real: 16/40 vecinos con
    categoría distinta vs 18/40 del embedding semántico).

    Solo se consideran "precedentes" los documentos cuya similitud TF-IDF
    con el documento evaluado supere el umbral estadístico de
    _umbral_similitud_tfidf() -- calculado sobre la propia distribución de
    similitudes del corpus, igual que el umbral del clasificador.

    Devuelve un dict con dos listas (cada una limitada a `max_por_lado`,
    ordenadas de mayor a menor similitud) más la clave "temas_documento":
        - "a_favor": precedentes similares que comparten el sentido
          predicho -- SUSTENTAN la predicción/el resolutivo propuesto.
        - "otros": precedentes de asunto/tema similar que resolvieron con
          un sentido DISTINTO -- NO sustentan la predicción, se muestran
          como contexto (casos parecidos en argumentación que, sin
          embargo, no llegaron a la misma conclusión).
        - "temas_documento": temas (materia) detectados en el documento
          evaluado, vía src/detectar_temas.py (regex, mismo criterio que
          el panel de KPIs). Lista vacía si no se detectó ninguno.

    Si el documento evaluado ya forma parte del corpus de referencia
    (ej. al probar el módulo con documentos del propio dataset de
    entrenamiento), pasar su `json_filename` en `excluir_json_filename`
    para que no aparezca como su propio precedente.

    `usar_temas` (default True -- decisión explicita de la autora: el tema
    es el segundo criterio en importancia después del sentido, especialmente
    del lado "a_favor"): dentro de cada lado (a_favor / otros) los
    candidatos que YA superan el umbral TF-IDF se reordenan para priorizar
    los que comparten al menos un tema (materia, detectado por
    src/detectar_temas.py) con el documento evaluado -- NO cambia el umbral
    ni el conjunto de candidatos que califican como "precedente" (eso sigue
    decidiéndolo solo sentido + similitud TF-IDF), solo el ORDEN dentro de
    ese conjunto ya filtrado (y por tanto cuáles de ellos entran en el top
    `max_por_lado` que se muestra). Requiere que exista
    data/temas_por_documento.csv (scripts/calcular_temas.py o, con mejor
    calidad de tema, scripts/calcular_temas_gemini.py); si no existe, se
    ignora `usar_temas` en silencio y se ordena solo por similitud (mismo
    comportamiento que antes de este cambio) -- no rompe nada si aún no has
    generado ese CSV. Pasar `usar_temas=False` explicitamente para volver al
    comportamiento anterior (solo similitud, sin priorizar por tema). Ver
    scripts/validar_reranking_temas.py para medir el efecto real sobre tu
    corpus antes/después de activarlo.
    """
    _cargar_corpus_referencia()
    _cargar_tfidf_precedentes()

    candidatos_df = _corpus_labels_df
    candidatos_tfidf = _tfidf_matriz_corpus

    if excluir_json_filename is not None:
        mascara = candidatos_df["json_filename"] != excluir_json_filename
        candidatos_df = candidatos_df[mascara].reset_index(drop=True)
        candidatos_tfidf = candidatos_tfidf[mascara.values]

    vector_doc = _vector_tfidf_documento(texto)
    similitudes = cosine_similarity(vector_doc, candidatos_tfidf).reshape(-1)

    umbral = _umbral_similitud_tfidf()
    supera_umbral = similitudes >= umbral

    temas_documento: set[str] = set()
    comparte_tema_arr = None
    if usar_temas:
        temas_por_json = _cargar_temas_corpus()
        if temas_por_json:
            temas_documento = set(detectar_temas_extendido(texto))
            comparte_tema_arr = np.array(
                [
                    bool(temas_documento & temas_por_json.get(jf, set()))
                    for jf in candidatos_df["json_filename"]
                ]
            )
        # Si no hay CSV de temas, usar_temas se ignora en silencio (ver
        # docstring) y comparte_tema_arr se queda en None.

    resultados: dict[str, list[dict]] = {
        "a_favor": [],
        "otros": [],
        "temas_documento": sorted(temas_documento),
    }

    for lado, mascara_lado in (
        ("a_favor", supera_umbral & (candidatos_df["sentido"] == sentido_predicho).values),
        ("otros", supera_umbral & (candidatos_df["sentido"] != sentido_predicho).values),
    ):
        indices = np.where(mascara_lado)[0]

        if comparte_tema_arr is not None:
            # Orden compuesto: primero los que comparten tema (booleano,
            # descendente), y dentro de cada grupo por similitud
            # descendente -- no toca qué candidatos pasaron el umbral,
            # solo cuáles de ellos quedan en el top `max_por_lado`.
            clave = comparte_tema_arr[indices].astype(float) * 10.0 + similitudes[indices]
            indices_ordenados = indices[np.argsort(-clave)][:max_por_lado]
        else:
            indices_ordenados = indices[np.argsort(-similitudes[indices])][:max_por_lado]

        for idx in indices_ordenados:
            fila = candidatos_df.iloc[idx]
            temas_candidato = (
                sorted(_cargar_temas_corpus().get(fila["json_filename"], set()))
                if usar_temas
                else []
            )
            resultados[lado].append(
                {
                    "expediente": _etiqueta_expediente(fila),
                    "json_filename": fila["json_filename"],
                    "category": fila.get("category"),
                    "sentido": fila["sentido"],
                    "similitud": float(similitudes[idx]),
                    "temas": temas_candidato,
                    "comparte_tema": bool(temas_documento & set(temas_candidato)) if usar_temas else False,
                }
            )

    return resultados


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(MODEL_NAME)
    return _embedder


def _chunk_text(text: str, words_per_chunk: int = WORDS_PER_CHUNK) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, len(words), words_per_chunk)
    ]


def _embed_document(text: str) -> np.ndarray:
    """Genera el embedding de un documento NUEVO con el modelo local,
    replicando EXACTAMENTE el proceso de scripts/generar_embeddings.py:
    parte el texto en fragmentos de ~90 palabras y promedia (mean
    pooling) -- así cae en el mismo espacio vectorial que el corpus de
    referencia (768 dimensiones). Cualquier cambio en ese script debe
    replicarse aquí.
    """
    chunks = _chunk_text(text)
    embedder = _get_embedder()
    chunk_embeddings = embedder.encode(chunks, batch_size=64, show_progress_bar=False)
    return chunk_embeddings.mean(axis=0)


def predict_proba_batch(texts: list[str]) -> np.ndarray:
    """Función de predicción compatible con LIME."""
    _cargar_modelo()
    embeddings = np.array([_embed_document(t) for t in texts])
    embeddings_scaled = _scaler.transform(embeddings)
    return _clf.predict_proba(embeddings_scaled)


def predecir_y_explicar(
    texto: str,
    num_features: int = 8,
    num_samples: int = 150,
    excluir_json_filename: str | None = None,
    usar_temas: bool = True,
) -> dict:
    """Predice el sentido de un documento (texto previo al resolutivo) y
    genera una explicación de motivos en lenguaje natural.

    Args:
        texto: texto del documento (idealmente ya recortado antes del
            resolutivo, igual que en el entrenamiento).
        num_features: cuántas palabras influyentes reportar.
        num_samples: cuántas variantes perturbadas evalúa LIME (más
            muestras = explicación más estable, pero más lenta).
        excluir_json_filename: si el documento evaluado ya está en el
            corpus de referencia (ej. pruebas con el propio dataset de
            entrenamiento), su nombre de archivo para que no se cite a
            sí mismo como precedente.
        usar_temas: reenvía a precedentes_relevantes() -- ver su docstring.
            Default True (decisión de la autora, ver docstring de
            precedentes_relevantes). Pasar False para volver al orden
            anterior (solo similitud) si scripts/validar_reranking_temas.py
            muestra que no conviene para tu corpus.
    """
    _cargar_modelo()

    embedding = _embed_document(texto).reshape(1, -1)
    embedding_scaled = _scaler.transform(embedding)
    probs = _clf.predict_proba(embedding_scaled)[0]
    pred_idx = probs.argmax()
    pred_label = _label_encoder.inverse_transform([pred_idx])[0]
    confianza = probs[pred_idx]

    explainer = LimeTextExplainer(class_names=list(_label_encoder.classes_))
    explanation = explainer.explain_instance(
        texto,
        predict_proba_batch,
        num_features=num_features,
        num_samples=num_samples,
        labels=[pred_idx],
    )

    palabras_influyentes = explanation.as_list(label=pred_idx)
    palabras_a_favor = [w for w, peso in palabras_influyentes if peso > 0]
    palabras_en_contra = [w for w, peso in palabras_influyentes if peso < 0]

    if palabras_a_favor:
        texto_explicacion = (
            f"Se predice '{pred_label}' (confianza {confianza:.0%}) principalmente "
            f"por la presencia de términos como: {', '.join(palabras_a_favor)}."
        )
    else:
        texto_explicacion = (
            f"Se predice '{pred_label}' (confianza {confianza:.0%}), aunque ninguna "
            f"palabra individual tuvo un peso claramente positivo."
        )
    if palabras_en_contra:
        texto_explicacion += (
            f" Palabras que jugaron en contra de esta clase: {', '.join(palabras_en_contra)}."
        )

    precedentes = precedentes_relevantes(
        texto,
        pred_label,
        excluir_json_filename=excluir_json_filename,
        usar_temas=usar_temas,
    )

    if precedentes["a_favor"]:
        expedientes_favor = ", ".join(
            f"{p['expediente']} ({p['similitud']:.0%} similitud)" for p in precedentes["a_favor"]
        )
        texto_explicacion += (
            f" Este caso es semánticamente similar a otros expedientes ya "
            f"resueltos como '{pred_label}': {expedientes_favor}."
        )
    else:
        texto_explicacion += (
            " No se encontraron precedentes con similitud inusualmente alta "
            "que respalden directamente esta predicción."
        )

    if precedentes["otros"]:
        expedientes_otros = ", ".join(
            f"{p['expediente']} ({p['sentido']}, {p['similitud']:.0%} similitud)"
            for p in precedentes["otros"]
        )
        texto_explicacion += (
            f" Ojo: también hay casos de asunto similar que NO sustentan esta "
            f"predicción porque resolvieron distinto: {expedientes_otros}."
        )

    return {
        "sentido_predicho": pred_label,
        "confianza": float(confianza),
        "palabras_influyentes": palabras_influyentes,
        "precedentes_a_favor": precedentes["a_favor"],
        "precedentes_otros": precedentes["otros"],
        "temas_documento": precedentes.get("temas_documento", []),
        "explicacion_texto": texto_explicacion,
    }


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_parquet(MODELO_DIR.parent / "data" / "dataset_modelo.parquet")
    muestra = df.sample(min(3, len(df)), random_state=7)

    for _, row in muestra.iterrows():
        print("=" * 80)
        print(f"Documento: {row['json_filename']}")
        print(f"Sentido real: {row['sentido']}")
        resultado = predecir_y_explicar(
            row["texto_insumo"], excluir_json_filename=row["json_filename"]
        )
        print(f"Sentido predicho: {resultado['sentido_predicho']} (confianza {resultado['confianza']:.0%})")
        print(f"Explicación: {resultado['explicacion_texto']}")
        print()
