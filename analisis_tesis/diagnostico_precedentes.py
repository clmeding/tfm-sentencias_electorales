"""Diagnóstico rápido: por qué src/evaluar_documento.py no encontró
precedentes para un documento en particular.

No cambia nada del umbral en producción -- solo IMPRIME, para el documento
que le pases, la similitud TF-IDF real contra los documentos más parecidos
del corpus (ignorando el umbral del percentil 90 que usa
precedentes_relevantes() para decidir qué mostrar). Así puedes distinguir
dos escenarios muy distintos:

  A) El documento más parecido tiene una similitud cercana al umbral (por
     ejemplo 20% cuando el umbral es 28%) -- comportamiento ESPERADO,
     sobre todo con un documento sintético/de prueba cuyos hechos, nombres
     y fechas son inventados y por lo tanto no comparten vocabulario
     distintivo con ningún caso real del corpus.
  B) La similitud con TODO el corpus es prácticamente cero (por ejemplo
     <1%) -- esto sí sería señal de un problema real (vectorizador vacío,
     texto mal extraído, o el corpus cargado no es el que se espera).

Uso:
    python scripts/diagnostico_precedentes.py ruta/al/documento.pdf
    python scripts/diagnostico_precedentes.py ruta/al/documento.docx
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

import numpy as np  # noqa: E402
from sklearn.metrics.pairwise import cosine_similarity  # noqa: E402

import explicar_prediccion as ep  # noqa: E402
from evaluar_documento import extraer_texto_documento, pre_resolutivo_text  # noqa: E402


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Uso: python scripts/diagnostico_precedentes.py ruta/al/documento.(pdf|docx)")

    ruta = Path(sys.argv[1])
    if not ruta.exists():
        raise SystemExit(f"ERROR: no existe el archivo '{ruta}'.")

    texto_limpio = extraer_texto_documento(ruta)
    texto_previo, metodo_corte = pre_resolutivo_text(texto_limpio)

    print(f"Documento: {ruta.name}")
    print(f"Método de corte del resolutivo: {metodo_corte}")
    print(f"Palabras en el texto usado para precedentes (antes del resolutivo): {len(texto_previo.split())}\n")

    ep._cargar_corpus_referencia()
    ep._cargar_tfidf_precedentes()

    vector_doc = ep._vector_tfidf_documento(texto_previo)
    n_terminos_encontrados = int((vector_doc != 0).sum())
    print(f"Términos del documento reconocidos por el vocabulario TF-IDF del corpus: {n_terminos_encontrados}")
    if n_terminos_encontrados == 0:
        print(
            "AVISO: el documento no comparte NINGÚN término con el vocabulario TF-IDF "
            "del corpus -- esto sí es anómalo, revisa que el texto se haya extraído bien "
            "(¿el .docx se leyó vacío? ¿el OCR falló?).\n"
        )

    similitudes = cosine_similarity(vector_doc, ep._tfidf_matriz_corpus).reshape(-1)
    umbral = ep._umbral_similitud_tfidf()

    print(f"\nUmbral actual para contar como 'precedente' (percentil {ep.PERCENTIL_SIMILITUD} del corpus): {umbral:.3f} ({umbral:.1%})\n")

    top_n = 10
    top_idx = np.argsort(-similitudes)[:top_n]
    print(f"Los {top_n} documentos más parecidos del corpus (sin aplicar el umbral):")
    labels_df = ep._corpus_labels_df
    for rango, idx in enumerate(top_idx, start=1):
        fila = labels_df.iloc[idx]
        marca = "  <-- SÍ supera el umbral" if similitudes[idx] >= umbral else ""
        print(
            f"  {rango:2d}. {similitudes[idx]:.3f} ({similitudes[idx]:.1%})  "
            f"sentido={fila['sentido']:28s} archivo={fila['json_filename']}{marca}"
        )

    max_similitud = float(similitudes.max())
    print(f"\nSimilitud máxima encontrada: {max_similitud:.3f} ({max_similitud:.1%})")
    if max_similitud >= umbral:
        print("-> Al menos un documento SÍ debería haber aparecido como precedente. Si la app mostró 'ninguno', revisa el filtro de categoría/sentido en precedentes_relevantes().")
    elif max_similitud >= umbral * 0.7:
        print("-> Cerca del umbral pero no lo cruza -- comportamiento esperado si el documento es sintético/atípico frente al corpus real.")
    else:
        print("-> Muy por debajo del umbral -- con un documento sintético de prueba esto puede ser normal (vocabulario/hechos inventados), pero si esperabas que fuera parecido a un caso real conocido, vale la pena revisar el texto extraído.")


if __name__ == "__main__":
    main()
