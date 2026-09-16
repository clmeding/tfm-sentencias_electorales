# tfm-sentencias_electorales
# Pipeline de análisis, Recuperación y Predicción de Resoluciones Electorales para el Tribunal Estatal Electoral de Nayarit

Trabajo de Fin de Máster. Autora: Clarissa Anahi Medina Garcia.

**Repositorio privado.** Ver [LICENSE.md](LICENSE.md) — todos los derechos reservados. 
El acceso de colaboradores está limitado a fines de revisión académica.

## Qué hace este proyecto

Extrae, limpia y estructura el texto de las sentencias públicas del Tribunal Estatal Electoral de Nayarit (fuente: [trieen.mx](https://trieen.mx/sentencias/)) para construir un corpus analizable.

En una etapa posterior, el pipeline aplica técnicas de procesamiento de lenguaje natural y aprendizaje automático supervisado para crear una interfaz que permite cargar un borrador de sentencia. Como resultado, el usuario recibe una muestra acotada de resoluciones afines al tema de fondo, una clasificación automática del sentido resolutivo junto con el nivel de confianza de la estimación, y acceso a resúmenes de las sentencias relacionadas.

Los resúmenes contienen los hechos, agravios, resolución, tema y enlace a la página web del documento, permitiendo obtener una idea previa de su contenido sin necesidad de consultar el documento completo.

Solo se consideran **sentencias** (se excluyen los Acuerdos Plenarios del corpus de modelado).

## Estructura

```
src/
  document_loader.py                          Localiza los documentos a procesar
  pdf_extractor.py                            Extracción directa de texto de PDFs
  docx_extractor.py                           Extracción de texto de archivos .docx
  ocr_processor.py                            OCR para PDFs escaneados (ocrmypdf +
                                              tesseract)
  cleaner.py                                  Limpieza de texto (incluye eliminación del
                                              ruido de OCR en el membrete institucional)
  metadata_extractor.py                       Extracción de metadatos desde el nombre de
                                              archivo
  sentido_extractor.py                        Detección del sentido de la resolución
                                              (confirma / revoca / modifica / etc.)
  detectar_temas.py                           Detector de temas por palabras clave
                                              (regex, sin API) para el panel de KPIs del
                                              corpus
  precedentes_datos.py                        Recuperación de precedentes (TF-IDF +
                                              coincidencia de sentido y tema)
  explicar_prediccion.py                      Predicción del clasificador de sentido +
                                              explicación LIME
  evaluar_documento.py                        Extracción y corte previo al resolutivo
                                              para un documento nuevo (usado por el demo
                                              Streamlit)
  main.py                                     Orquesta el pipeline completo

scripts/  (pipeline productivo)
  download_trieen.py                          Descarga el corpus completo desde trieen.mx
  check_new_documents.py                      Revisa si hay documentos nuevos (usado por
                                              el workflow programado de GitHub Actions)
  build_dataset.py                            Genera el inventario data/dataset_sentencias.csv
                                              (una fila por documento, con su sentido)
  build_dataset_modelo.py                     Construye el dataset de MODELADO: texto
                                              previo al resolutivo (insumo) + sentido
                                              (etiqueta). Valida que cada etiqueta
                                              pertenezca al catálogo oficial de 13
                                              clases y descarta documentos duplicados
                                              (mismo texto, nombre de archivo distinto).
  detectar_sentido_gemini.py                  Detecta el sentido vía API de Gemini para
                                              los documentos que sentido_extractor.py
                                              (basado en reglas) no pudo clasificar.
                                              Solo se aceptan etiquetas con confianza
                                              alta o media.
  calcular_temas.py                           Recorre documents/output/*.json y genera
                                              data/temas_por_documento.csv (detección de
                                              temas por palabras clave, sin IA) para el
                                              panel de KPIs del corpus histórico.
  generar_embeddings.py                       Genera los embeddings semánticos de cada
                                              documento del dataset de modelado con un
                                              modelo local (sentence-transformers,
                                              paraphrase-multilingual-mpnet-base-v2, 768
                                              dimensiones). Corre offline, sin costo por
                                              evaluación.
  entrenar_modelo.py                          Entrena el clasificador final (Regresión
                                              Logística L2) sobre el 100% de los datos y
                                              deja modelo/*.joblib listo para producción.
  generar_precedentes_tfidf.py                Construye el índice TF-IDF usado para
                                              recuperar precedentes similares.
  generar_resumenes.py                        Genera resúmenes estructurados (hechos,
                                              agravios, resolución) de cada sentencia
                                              vía Gemini, usados en el demo de evaluación.
  cargar_sqlite.py                            Carga ETL a base de datos SQLite (etapa
                                              "Load"), integrando
                                              dataset_sentencias.csv, resumenes.json y
                                              temas_por_documento.csv.

scripts/  (análisis, evaluación y diagnóstico)
  comparar_modelos_clasificacion.py           Compara Regresión Logística L2, SVM Lineal
                                              y Random Forest (k=5) -- sustenta la
                                              elección del modelo final.
  evaluar_kfold.py                            Genera las métricas oficiales reportadas
                                              (accuracy, F1 macro, F1 ponderado) por
                                              validación cruzada estratificada k=5.
  diagnostico_confusion.py                    Genera la matriz de confusión y el F1 por
                                              clase (out-of-fold) del modelo oficial.
  comparar_escenarios_corte.py                Prueba de sensibilidad: corte aproximado
                                              vs. corpus completo -- superada por
                                              evaluar_kfold.py, se conserva como
                                              referencia histórica de la decisión (ver
                                              CAMBIOS.md).
  comparar_precedentes_tfidf_vs_semantico.py  Reconstruye, sobre el corpus completo, la
                                              comparación que justificó usar TF-IDF en
                                              vez de embeddings semánticos para
                                              precedentes.
  validar_reranking_temas.py                  Validó que reordenar precedentes por tema
                                              compartido mejora los resultados --
                                              decisión ya adoptada por defecto en
                                              producción (usar_temas=True).
  diagnostico_sentido.py                      Por qué un documento no obtuvo sentido por
                                              regex.
  diagnostico_precedentes.py                  Por qué un documento no obtuvo precedentes.
  mejorar_features_negacion.py                Prueba (no adoptada) de agregar features
                                              léxicas de negación a los embeddings.
  detectar_temas_gemini.py                    Retiquetado del corpus completo por tema
                                              vía Gemini -- usado solo para la
                                              comparación documentada en
                                              detectar_temas.py (revisión 30-ago-2026),
                                              no forma parte del detector regex que
                                              corre en producción.
  calcular_temas_gemini.py                    Combina esa detección Gemini con la regex
                                              de respaldo, para la misma comparación.
  resumen_temas_libres.py                     Resume los temas de vocabulario libre que
                                              Gemini identificó en ese retiquetado.

demo/
  streamlit_app.py                            Aplicación demostrativa: evaluación de un
                                              documento nuevo (sentido + LIME +
                                              precedentes) y panel de KPIs del corpus
                                              histórico.

analisis_tesis/
  analisis_descriptivo.py                     Análisis descriptivo completo del corpus
                                              para el TFM (embudo de construcción,
                                              categoría, año, sentido, temas, longitud
                                              de texto).
  matriz_confusion_heatmap.py                 Genera las figuras de matriz de confusión
                                              de los modelos comparados.
  generar_diagrama_sentido.py                 Genera el diagrama de flujo de detección
                                              del sentido.

.github/workflows/
  check_new_sentencias.yml                    Revisión diaria automática de documentos
                                              nuevos en trieen.mx (abre un Issue si
                                              encuentra alguno).

data/
  dataset_sentencias.csv                      Inventario de todos los documentos
                                              (metadatos + sentido detectado), generado
                                              por build_dataset.py
  dataset_modelo.parquet                      Dataset de modelado (texto insumo +
                                              etiqueta), generado por
                                              build_dataset_modelo.py -- no se versiona
                                              en git, se regenera localmente
  known_documents.json                        Manifiesto de documentos ya detectados
                                              (usado por el workflow programado)
  temas_por_documento.csv                     Tema(s) detectado(s) por documento,
                                              generado por calcular_temas.py
  nucleo_temas_actual.json                    Núcleo dinámico de temas más frecuentes
                                              del corpus real, recalculado cada vez que
                                              corre calcular_temas.py
  resumenes.json                              Resúmenes estructurados (hechos, agravios,
                                              resolución) de cada sentencia, generados
                                              por generar_resumenes.py
  diagnostico_calidad_ocr.csv                 Puntaje de calidad de OCR por documento,
                                              generado por diagnostico_calidad_ocr.py
  temas_gemini.json                           Retiquetado del corpus completo por tema
                                              vía Gemini (comparación histórica)
  temas_gemini_backup_09sep.json              Respaldo del retiquetado anterior a la
                                              incorporación del respaldo Gemini
  temas_libres_frecuencia.csv                 Frecuencia de los temas de vocabulario
                                              libre identificados por Gemini
  validacion_reranking_temas.csv              Resultado de la validación del
                                              reordenamiento de precedentes por tema
  sentido_gemini_v1_backup.json               Respaldo de una versión anterior de las
                                              etiquetas de sentido detectadas por Gemini

modelo/
  clasificador.joblib                         Modelo entrenado (regresión logística
                                              sobre embeddings)
  scaler.joblib                               Escalador usado antes de clasificar
  label_encoder.joblib                        Codificador de las clases de sentido
```

## Análisis descriptivo del catálogo de sentencias
El corpus se construye del total de 886 documentos publicados en el sitio oficial del Tribunal Estatal del Estado de Nayarit. Durante la extracción del texto, 7 documentos resultaron ilegibles por fallas de digitalización que la corrección automatica de orientación no logró resolver por lo que partimos de un inventario base de 879 documentos. A partir del inventario base se excluyeron 13 Acuerdos Plenarios, lo que deja 866 documentos elegibles. De estos, se excluyeron 10 adicionales: 4 porque no fue posible obtener una etiqueta de sentido útil y 6 por ser duplicados. El corpus final de modelado queda en 856 documentos.

## Embeddings semánticos

Cada documento del dataset de modelado se convierte en un vector semántico antes de entrenar el clasificador (scripts/generar_embeddings.py). Se compararon dos modelos locales sobre una muestra de 673 documentos (partición 80/20 estratificada):

- **MiniLM** (384 dimensiones): exactitud 63.7%, F1 macro 0.443, F1 ponderado 0.625.
- **mpnet** (paraphrase-multilingual-mpnet-base-v2, 768 dimensiones): exactitud 67.4%, F1 macro 0.519, F1 ponderado 0.663.

Se adoptó mpnet como modelo oficial: la mejora en F1 macro indica que generaliza mejor específicamente sobre las clases minoritarias, no solo sobre las clases con más ejemplos. Corre localmente y sin costo por evaluación (scripts/generar_embeddings.py).

## Modelo de clasificación del sentido resolutivo

Se compararon tres algoritmos sobre el corpus final (856 documentos, embeddings mpnet-base-v2 de 768 dimensiones) con validación cruzada estratificada k=5 (scripts/comparar_modelos_clasificacion.py, scripts/evaluar_kfold.py):
- **Regresión Logística L2** (modelo final): exactitud 0.568 ± 0.033, F1 macro 0.386 ± 0.025, F1 ponderado 0.565 ± 0.031.
- **SVM Lineal**: descartado, no mejoró ninguna métrica frente a la Regresión Logística y presentó problemas de convergencia.
- **Random Forest** (ajustado con RandomizedSearchCV): mejores métricas globales, pero falla por completo en la clase minoritaria declara_improcedente (0/19 aciertos) y confunde con más frecuencia el par jurídicamente opuesto existencia_infraccion/inexistencia_infraccion.

Se eligió la Regresión Logística L2 como modelo final por su interpretabilidad y un comportamiento más controlado entre las 13 categorías, priorizando el comportamiento por clase sobre la métrica global. El detalle por clase y las matrices de confusión se generan con scripts/diagnostico_confusion.py y analisis_tesis/matriz_confusion_heatmap.py.

La explicación de cada predicción individual (LIME) se genera con src/explicar_prediccion.py, y la recuperación de precedentes similares (TF-IDF + coincidencia de sentido y tema) con src/precedentes_datos.py y scripts/generar_precedentes_tfidf.py. Ambas se integran en el demo de Streamlit (demo/streamlit_app.py).

## Validación cruzada k-fold

El modelo de clasificación del sentido se evalúa con validación cruzada estratificada (k=5, random_state=42) en vez de una partición única, para evitar conclusiones sensibles al muestreo aleatorio (scripts/evaluar_kfold.py, scripts/comparar_modelos_clasificacion.py).

Antes de fijar el corpus oficial, se comparó el escenario "corte aproximado" (documentos donde no se pudo ubicar el resolutivo exacto y se descarta el último 15% del texto, 159/856 documentos) contra el corpus completo. La diferencia entre ambos (0.006 en exactitud, 0.003 en F1 macro) está dentro del rango de variabilidad entre folds, así que se adoptó el corpus completo como oficial.

Resultado final (Regresión Logística L2, corpus completo, 856 documentos): exactitud 0.568 ± 0.033, F1 macro 0.386 ± 0.025 (0.4000 sobre el conjunto completo de predicciones out-of-fold), F1 ponderado 0.565 ± 0.031. El detalle por clase y las matrices de confusión se generan con scripts/diagnostico_confusion.py.

## Generación de resúmenes

Cada precedente que el sistema muestra incluye un resumen breve (hechos, agravios y resolución) para que la persona usuaria pueda decidir si vale la pena revisar el documento completo sin abrirlo. El resumen se genera una sola vez por sentencia mediante la API de Gemini y se guarda en data/resumenes.json (scripts/generar_resumenes.py).

A diferencia del dataset de modelado -- que solo incluye las sentencias con una etiqueta de sentido detectada -- el resumen se genera para todas las sentencias del corpus, sin importar si el clasificador pudo determinar su sentido, porque su propósito es distinto: ayudar a evaluar la utilidad de un precedente, no entrenar el modelo.

El proceso es incremental: cada vez que se agregan documentos nuevos (main.py), se vuelve a correr generar_resumenes.py y solo genera el resumen de los documentos pendientes, sin repetir ni volver a pagar por los que ya tienen resumen guardado.

## Recuperación de precedentes

Para identificar sentencias similares a un borrador, se evaluaron cuatro estrategias sobre una muestra de 40 documentos reales (similitud de coseno, candidatos por encima del percentil 90): embeddings semánticos (mpnet) reutilizados del clasificador, la sección de HECHOS extraída por expresiones regulares, resúmenes generados por LLM, y TF-IDF sobre el documento completo.

TF-IDF obtuvo el mejor resultado (13 vecinos de temática distinta, frente a 21-23 de las otras estrategias) y se confirmó al repetir la evaluación sobre el corpus completo (856 documentos). Se adoptó como base del proceso (scripts/generar_precedentes_tfidf.py).

El proceso final combina tres señales (src/precedentes_datos.py):
1. Similitud de coseno sobre representación TF-IDF -- filtra candidatos por umbral de similitud textual.
2. Coincidencia de sentido -- separa candidatos en "A favor" (mismo sentido que la predicción) y "Otros" (sentido distinto, útil como contraste).
3. Coincidencia de tema -- prioriza, dentro de cada grupo, los precedentes que comparten al menos un tema con el documento evaluado.

Diagnóstico y validación: scripts/comparar_precedentes_tfidf_vs_semantico.py, scripts/diagnostico_precedentes.py, scripts/validar_reranking_temas.py.





