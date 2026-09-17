# Pipeline de análisis de sentencias — TEE Nayarit

Trabajo de Fin de Máster. Autora: Clarissa Anahi Medina Garcia.

**Repositorio privado.** Ver [LICENSE.md](LICENSE.md) — todos los
derechos reservados. El acceso de colaboradores está limitado a fines de
revisión académica.

## Qué hace este proyecto

Extrae, limpia y estructura el texto de las sentencias públicas del
Tribunal Estatal Electoral de Nayarit (fuente: [trieen.mx](https://trieen.mx/sentencias/)),
para construir un corpus analizable y, en una etapa posterior, entrenar un
modelo de aprendizaje automático que estime la probabilidad de que una
resolución sea confirmada, modificada o revocada.

Solo se consideran **sentencias** (se excluyen los Acuerdos Plenarios del
corpus de modelado).

## Estructura

```
src/
  document_loader.py     Localiza los documentos a procesar
  pdf_extractor.py       Extracción directa de texto de PDFs
  docx_extractor.py      Extracción de texto de archivos .docx
  ocr_processor.py       OCR para PDFs escaneados (ocrmypdf + tesseract)
  cleaner.py             Limpieza de texto (incluye eliminación del
                          ruido de OCR en el membrete institucional)
  metadata_extractor.py  Extracción de metadatos desde el nombre de archivo
  sentido_extractor.py   Detección del sentido de la resolución
                          (confirma / revoca / modifica / etc.)
  detectar_temas.py      Detector de temas por palabras clave (regex,
                          sin IA) para el panel de KPIs del corpus
  main.py                Orquesta el pipeline completo

scripts/
  download_trieen.py       Descarga el corpus completo desde trieen.mx
  check_new_documents.py   Revisa si hay documentos nuevos (usado por
                            el workflow programado de GitHub Actions)
  build_dataset.py         Genera el inventario data/dataset_sentencias.csv
                           (una fila por documento, con su sentido)
  build_dataset_modelo.py  Construye el dataset de MODELADO: texto previo
                           al resolutivo (insumo) + sentido (etiqueta).
                           Valida que cada etiqueta pertenezca al catálogo
                           oficial de 13 clases y descarta documentos
                           duplicados (mismo texto, nombre de archivo
                           distinto).
  detectar_sentido_gemini.py     Detecta el sentido vía API de Gemini para
                            los documentos que sentido_extractor.py (basado
                            en reglas) no pudo clasificar. Solo se aceptan
                            etiquetas con confianza alta o media.
  generar_embeddings.py    Genera los embeddings semánticos de cada
                            documento del dataset de modelado con un
                            modelo local (sentence-transformers,
                            paraphrase-multilingual-mpnet-base-v2, 768
                            dimensiones). Es el script oficial del
                            pipeline -- corre offline, sin costo por
                            evaluación.
  generar_embeddings_gemini.py   Versión alterna que usa la API de
                            Gemini (gemini-embedding-001, 768
                            dimensiones) en vez del modelo local. Se
                            conserva solo como referencia histórica de
                            la comparación de embeddings (ver CAMBIOS.md,
                            sección 9) -- ya no forma parte del pipeline
                            oficial por motivos de reproducibilidad.
  comparar_escenarios_corte.py   Análisis de sensibilidad (partición
                            única 80/20, ver CAMBIOS.md sección 9) --
                            superado por evaluar_kfold.py, se conserva
                            solo como referencia histórica.
  evaluar_kfold.py         Valida los escenarios completo/sin_aproximado
                            con validación cruzada estratificada (k=5) --
                            método oficial de evaluación del proyecto (ver
                            CAMBIOS.md, sección 10).
  diagnostico_confusion.py Genera la matriz de confusión y el F1 por
                            clase (out-of-fold) del modelo oficial.
  diagnostico_calidad_ocr.py     Mide qué tan legible salió el OCR de
                            cada documento, para detectar casos con texto
                            ilegible que deban excluirse del corpus.
  calcular_temas.py        Recorre documents/output/*.json y genera
                            data/temas_por_documento.csv (detección de
                            temas por palabras clave, sin IA) para el
                            panel de KPIs del corpus histórico.
  generar_resumenes.py     Genera resúmenes estructurados (hechos,
                            agravios, resolución) de cada sentencia vía
                            Gemini, usados en el demo de evaluación.
  entrenar_modelo.py       Entrena y evalúa el clasificador de sentido
                            sobre un único escenario del dataset completo

analisis_tesis/
  analisis_descriptivo.py  Script aislado (fuera de scripts/, NO forma
                           parte del pipeline de producción) que genera
                           el análisis descriptivo completo del corpus
                           para el escrito de tesis: desde el inventario
                           crudo hasta el detalle final (embudo de
                           construcción del corpus, categoría, año,
                           sentido, temas, longitud de texto). Ver
                           sección "Análisis descriptivo para el TFM"
                           más abajo.

.github/workflows/
  check_new_sentencias.yml  Revisión diaria automática de documentos
                             nuevos en trieen.mx (abre un Issue si
                             encuentra alguno)

data/
  known_documents.json    Manifiesto de documentos ya detectados
                           (usado por el workflow programado)
  dataset_sentencias.csv  Inventario de todos los documentos (metadatos +
                           sentido detectado), generado por build_dataset.py
  temas_por_documento.csv Tema(s) detectado(s) por documento (detección
                           por palabras clave), generado por calcular_temas.py
  dataset_modelo.parquet  Dataset de modelado (texto insumo + etiqueta),
                           generado por build_dataset_modelo.py -- no se
                           versiona en git, se regenera localmente

modelo/
  clasificador.joblib      Modelo entrenado (regresión logística sobre
                           embeddings)
  scaler.joblib            Escalador usado antes de clasificar
  label_encoder.joblib     Codificador de las clases de sentido
  (no se versionan en git -- se regeneran corriendo los scripts de
  modelado, ver más abajo)
```

## Modelo predictivo de sentido

Además del pipeline de extracción, el proyecto incluye un modelo que
predice el sentido probable de una sentencia (confirma/revoca/modifica/
etc.) a partir del texto **previo al resolutivo** (hechos, agravios,
considerandos) -- deliberadamente se excluye el resolutivo mismo, porque
es justo ahí donde se anuncia la respuesta, y usarlo como insumo
convertiría al modelo en un simple buscador de palabras clave en vez de
un modelo predictivo real.

**Corpus de modelado:** el detector de sentido basado en reglas
(`sentido_extractor.py`) identifica 565 documentos utilizables sobre un
inventario total de 879 (886 originales menos 7 documentos con OCR
ilegible, ver más abajo). Para aprovechar parte de los ~300 restantes
que quedan como `no_detectado` (en su mayoría por variantes de OCR o
redacciones atípicas del resolutivo, no porque no tuvieran sentido), se
añadió una segunda pasada con la API de Gemini (`detectar_sentido_gemini.py`)
que clasifica esos documentos y solo acepta las etiquetas con confianza
alta o media. Tras excluir Acuerdos Plenarios (13), documentos sin
etiqueta de sentido utilizable (4) y duplicados (6), el corpus de
modelado final queda en **856 documentos**, cifra verificada de forma
reproducible contra el código y los datos versionados en git — ver la
nota de corrección de la auditoría en
[CAMBIOS.md, sección 11](CAMBIOS.md#11-auditoría-de-reproducibilidad-del-corpus-de-modelado-849-no-856-2-de-septiembre-de-2026).

> **Nota sobre la cifra del corpus:** una auditoría del 2 de septiembre
> de 2026 (sección 11 de CAMBIOS.md) había declarado 849 documentos como
> cifra reproducible, en lugar de los 856 de la sección 10. Una revisión
> posterior (5 de septiembre) confirmó que el código y los datos usados
> en esa auditoría son idénticos a los de hoy y que el pipeline completo
> reproduce 856 de forma consistente -- la auditoría de 849 no se
> había verificado ejecutando el pipeline completo en su momento. Se
> retracta esa cifra: **856 documentos es la cifra oficial y
> reproducible**, consistente con las métricas de exactitud/F1 de la
> tabla siguiente.

**Análisis de sensibilidad (con/sin corte aproximado):** en parte de esos
documentos no fue posible ubicar el punto de corte con un marcador
exacto (encabezado RESUELVE, transición "por lo fundado y motivado", o el
marcador verbatim que devolvió Gemini), así que se usó un corte
conservador del último ~15% del texto, dejando un escenario
`sin_aproximado` más chico. La comparación se hace con
validación cruzada k-fold (`scripts/evaluar_kfold.py`, k=5) en vez de
una sola partición 80/20, porque una partición única resultó ser
demasiado sensible al ruido con un corpus de este tamaño (ver CAMBIOS.md
sección 10 para el detalle de cómo esto revirtió una conclusión
anterior):

| Escenario | Documentos | Exactitud | F1 macro | F1 ponderado |
|---|---|---|---|---|
| Completo | 856 | 59.6% ± 1.5 pp | 0.402 ± 0.032 | 0.594 ± 0.014 |
| Sin corte aproximado | 697 | 59.0% ± 1.5 pp | 0.399 ± 0.048 | 0.589 ± 0.016 |

Con validación cruzada, ambos escenarios quedan prácticamente empatados
(la diferencia está dentro de una desviación estándar entre folds) — la
ventaja que parecía tener "sin_aproximado" en la evaluación de partición
única anterior no se sostiene con una medición más robusta. Dado que el
desempeño es equivalente, se declaró el escenario **completo** como
oficial del proyecto, en reemplazo de `sin_aproximado` (ver CAMBIOS.md
sección 10): aprovecha más datos sin sacrificar desempeño.

Enfoque: se generan embeddings semánticos con un modelo local
(`sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, 768
dimensiones) de cada documento, y sobre esos embeddings se entrena un
clasificador (regresión logística). Cada predicción se acompaña de una
explicación en lenguaje natural (vía LIME) que señala qué palabras del
texto más influyeron en el resultado.

Se evaluaron tres opciones de embeddings (Gemini, un modelo local más
pequeño -- MiniLM -- y el modelo local mpnet actual) antes de decidir;
ver la comparación completa y la justificación de reproducibilidad en
CAMBIOS.md, sección 9. Se optó por mpnet: cierra casi toda la brecha en
exactitud frente a Gemini y corre completamente offline, sin costo por
evaluación ni dependencia de una API externa.

Cómo reproducir el corpus de modelado y las métricas de k-fold en tu
máquina local (requiere `documents/output/*.json`, no versionado en
git):

```bash
python scripts/build_dataset.py        # regenera data/dataset_sentencias.csv (879 filas)
python scripts/detectar_sentido_gemini.py  # regenera data/sentido_gemini.json (si no lo tienes ya)
python scripts/build_dataset_modelo.py  # construye data/dataset_modelo.parquet (856 filas)
python scripts/generar_embeddings.py    # modelo local (mpnet) -- no requiere API key
python scripts/evaluar_kfold.py         # valida ambos escenarios con k-fold (k=5)
python scripts/diagnostico_confusion.py # matriz de confusión y F1 por clase, out-of-fold
python src/explicar_prediccion.py       # ejemplo de predicción + explicación sobre 3 documentos al azar
```

Desempeño del modelo oficial (escenario **completo**, validación
cruzada k-fold, k=5, embeddings mpnet locales, ver detalle completo en
CAMBIOS.md sección 10 y la nota de reproducibilidad arriba): exactitud
59.6% ± 1.5 pp, F1 macro 0.402 ± 0.032, F1 ponderado 0.594 ± 0.014. Las
clases con pocos ejemplos en el corpus (cumplida, declara_improcedente,
declara_infundado, parcialmente_cumplida, incumplida, modifica) tienen
métricas poco confiables por falta de datos -- además, la matriz de
confusión muestra que se confunden principalmente con clases legalmente
vecinas (p. ej. modifica con confirma/revoca), no al azar. Se documenta
como limitación honesta del tamaño del corpus en el TFM, no se oculta.

## Panel de KPIs por tema (corpus histórico)

`demo/streamlit_app.py` incluye una pestaña **"Panel del corpus
histórico"** con:

- Total de sentencias analizadas.
- Sentencias por tema (barra) -- incluye una barra "Otro / no
  identificado" con las sentencias que no calzan en ninguno de los
  temas del catálogo, para que el gráfico sume el 100% del corpus sin
  dar la impresión de que falta cobertura.
- Sentencias por año y por categoría de procedimiento (barras).

Los temas se detectan por palabras clave (`src/detectar_temas.py`,
regex, sin IA) con un umbral de 3+ menciones por documento para evitar
falsos positivos (ej. "coalición" apareciendo solo en el nombre de un
partido, no porque el caso trate de coaliciones). El panel excluye los
Acuerdos Plenarios usando la columna `es_acuerdo_plenario` de
`data/dataset_sentencias.csv` (si esa columna no existe, no filtra
nada). No clasifica con precisión jurídica "de qué trata" la sentencia
-- es detección por palabras clave, documentado como tal en la propia
pestaña del panel.

Para regenerar el CSV de temas: `python scripts/calcular_temas.py`
(segundos, sin GPU ni API).

## Análisis descriptivo para el TFM

`analisis_tesis/analisis_descriptivo.py` es un script aislado (fuera de
`scripts/`, no forma parte del pipeline de producción) que genera el
análisis descriptivo completo del corpus para la sección correspondiente
del escrito: el embudo completo desde el inventario crudo hasta el
detalle final, categoría de procedimiento, año, sentido de la
resolución, temas y distribución de longitud de texto.

```bash
python analisis_tesis/analisis_descriptivo.py --repo-root .
```

Genera 7 gráficas (paleta Nexus) y un resumen en Markdown en
`analisis_tesis/output/`. Ver la corrección de la auditoría de
reproducibilidad del corpus (856 documentos, confirmado) en CAMBIOS.md,
sección 11.

## Resúmenes automáticos

`scripts/generar_resumenes.py` genera, vía Gemini, un resumen estructurado
(hechos / agravios / resolución) de cada sentencia en alcance -- 873
documentos (todas las sentencias, sin Acuerdos Plenarios, sin importar si
se les detectó sentido o no, calculado antes de la exclusión de OCR
ilegible). Se guarda en `data/resumenes.json`, es resumible (si se
interrumpe, retoma donde se quedó) y se usa en el demo de evaluación de
documentos.

## Uso

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Descargar el corpus completo
python scripts/download_trieen.py

# Procesar (extracción de texto + OCR cuando es necesario + limpieza)
python src/main.py
```

Requiere `ocrmypdf` y `tesseract` (con el paquete de idioma español)
instalados en el sistema para el OCR de PDFs escaneados.

## Revisión automática de documentos nuevos

El workflow en `.github/workflows/check_new_sentencias.yml` corre todos
los días y avisa (abriendo un Issue en este repositorio) si aparecen
sentencias nuevas en trieen.mx que aún no se han procesado. No descarga
ni procesa los PDFs en la nube — solo detecta y notifica, para correr el
pipeline completo donde se tengan los recursos necesarios (OCR).

## Trabajo futuro: automatizar el pipeline completo en la nube

Hoy el workflow programado (`check_new_sentencias.yml`) solo detecta
documentos nuevos y abre un Issue, porque el texto extraído y los PDFs
(varios GB) viven solo en local y están en `.gitignore` -- no caben en
Git. Procesar un documento nuevo (descarga, OCR, limpieza, resumen,
predicción) sigue requiriendo correr el pipeline a mano.

Una vía para automatizarlo por completo sin costo adicional relevante,
evaluada pero no implementada por prioridad de tiempo antes de la
entrega del TFM:

- **Cómputo:** ampliar la misma GitHub Action para que, además de
  detectar, corra el pipeline completo (descarga -> OCR con
  `tesseract-ocr` vía `apt install` -> limpieza -> resumen con Gemini ->
  predicción del modelo). El repo privado incluye 2,000 minutos/mes
  gratis de Actions ([GitHub Docs](https://docs.github.com/billing/managing-billing-for-github-actions/about-billing-for-github-actions)),
  más que suficiente para procesar unos cuantos documentos nuevos por
  corrida.
- **Almacenamiento:** mover `documents/` (PDFs + texto extraído) a un
  bucket tipo Cloudflare R2 en vez de mantenerlo solo en local -- da
  10 GB gratis al mes y **no cobra por descargar datos** (egress
  gratis), así la Action puede leer y escribir ahí sin costo extra
  ([Cloudflare R2 pricing](https://developers.cloudflare.com/r2/pricing/)).
- La llamada a Gemini (`scripts/generar_resumenes.py`) y la predicción
  del modelo ya no dependerían de correr nada en una máquina local --
  solo cambiaría la API key de variable de entorno local a "secret" de
  GitHub Actions.

Se decidió **posponer esta migración** y documentarla aquí como línea de
trabajo futuro, priorizando cerrar la herramienta de evaluación de
documentos (`src/evaluar_documento.py`) y el demo para la defensa antes
de la fecha de entrega.

## Qué NO hace el panel de temas (para no generar expectativas de más)

- No clasifica con precisión jurídica "de qué trata" la sentencia -- es
  detección por palabras clave.
- No agrega los KPIs más interpretativos de un dashboard de referencia
  (votación, ponencia por sexo, criterio relevante, perspectiva de
  género) -- esos requieren datos que hoy no se extraen del texto y
  probablemente juicio manual o Gemini con validación, quedan fuera dado
  el tiempo disponible hasta la entrega.

## Ver también

- [CAMBIOS.md](CAMBIOS.md) — historial de cambios y decisiones técnicas.
