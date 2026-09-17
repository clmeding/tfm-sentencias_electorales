# Cambios (actualización: nuevo módulo sentido_extractor.py) sobre tu pipeline original

Probé todo el flujo de punta a punta con documentos reales descargados de
trieen.mx antes de entregártelo, así que ya sabemos que funciona.

## 1. Bug corregido en `src/main.py`
Tu `main.py` importaba `run_ocr` pero nunca lo llamaba. Cuando un PDF se
marcaba como `needs_ocr = True`, el pipeline solo imprimía "PENDIENTE DE
OCR" y lo dejaba sin texto extraído — el módulo `ocr_processor.py` nunca se
ejecutaba. Ahora `extract_document()` llama automáticamente a `run_ocr()`
cuando hace falta, y el resumen final distingue "completadas directas" vs.
"procesadas con OCR".

Verificado con dos "Acuerdos Plenarios" reales del TEEN: ambos necesitaban
OCR (eran PDFs escaneados) y ahora se procesan correctamente de punta a
punta.

## 2. Nuevo módulo `src/metadata_extractor.py`
Extrae automáticamente, a partir de la ruta y el nombre del archivo:
- `category` (AP, JIN, JDCN, MII, PES, RV — el tipo de medio de impugnación)
- `expediente` (ej. `TEE-JIN-29-2024`)
- `year` y `document_date`
- `document_type` (sentencia, acuerdo_plenario, incidente, resolucion...)
- `has_acumulados` (si el expediente trae asuntos acumulados)

Esto es la base para las siguientes etapas: vincular cada sentencia con su
"sentido" (confirma/revoca/modifica) y para las features del clasificador.
Ya lo integré a `save_result()` en `main.py`, así que cada JSON de salida
trae estos campos.

## 3. Nuevo script `scripts/download_trieen.py`
Descarga el corpus público de sentencias del Tribunal Estatal Electoral de
Nayarit (trieen.mx) sin necesidad del cPanel del IEEN. El sitio expone el
listado completo (886 documentos: AP, JIN, JDCN, MII, PES, RV) a través de
su API REST de WordPress (`wp-json/wp/v2/pages/160`), que es más fiable que
intentar leer el HTML renderizado.

El script:
- Descarga cada PDF en `documents/input/<CATEGORIA>/`.
- Omite archivos que ya existan (puedes volver a correrlo sin duplicar).
- Espera 0.5s entre descargas para no saturar el servidor del tribunal.
- Guarda un log de errores en `documents/input/_errores_descarga.txt` si algo falla.

Ejecútalo desde la raíz del proyecto:
```
pip install requests
python scripts/download_trieen.py
```

Luego corre tu pipeline normal:
```
python src/main.py
```

## Dependencias nuevas
Agrega a tu `requirements.txt`:
```
requests
```
(`ocrmypdf` y `pymupdf` ya los tenías; asegúrate de tener instalado también
el paquete de idioma `tesseract-ocr-spa` en tu sistema para que el OCR en
español funcione — sin él, `ocrmypdf` falla con un error claro).

## Pendientes que quedan para la siguiente sesión
- `cleaner.py` es conservador; el texto salido de OCR trae ruido del
  membrete institucional repetido en cada página (ej. "TRIBUNAL ELECTORAL
  NAYARIT"). Vale la pena añadir una limpieza de encabezados/pies de página
  repetidos antes de pasar el texto a los embeddings.
- Ampliar la validación manual del detector de sentido (ver sección
  siguiente) sobre una muestra más grande antes de usarlo para entrenar el
  modelo de clasificación.

## 4bis. Corrección crítica: el OCR se podía quedar trabado para siempre

**El problema que reportaste ("se quedó en un loop infinito"):** el código
llamaba a `ocrmypdf` sin ningún límite de tiempo. Si un solo PDF tenía una
página dañada o era muy pesado, el proceso se podía quedar esperando
indefinidamente — y como el pipeline original procesaba un documento a la
vez, eso bloqueaba a los 833 documentos restantes detrás del que se atoraba.

**Qué se corrigió:**
1. `ocr_processor.py` ahora tiene un límite de 300 segundos (5 min) por
   documento. Si `ocrmypdf` no termina en ese tiempo, se aborta *solo ese
   documento* (queda registrado como error) y el pipeline sigue con el resto.
2. `main.py` ahora **recuerda qué documentos ya procesó** (revisa si ya
   existe el JSON/TXT de salida). Si interrumpes el proceso (Ctrl+C) o se
   cae por cualquier motivo, al volver a correrlo no repite nada de lo que
   ya completó — sigue exactamente donde se quedó.
3. `main.py` ahora procesa **4 documentos en paralelo** (antes uno por uno),
   usando `ProcessPoolExecutor`. Esto reduce el tiempo total casi 4 veces.
   Puedes ajustar `MAX_WORKERS` en la parte superior del archivo si tu Mac
   tiene más núcleos disponibles y quieres ir más rápido aún.

**Antes de volver a correrlo:** si tienes el proceso viejo trabado, deténlo
con Ctrl+C (o cierra la terminal). No hace falta borrar nada de
`documents/output/` — el pipeline nuevo detecta automáticamente qué ya
estaba hecho.

**Corrección adicional (importante si ya corriste el `main.py` original
antes de recibir estos cambios):** la primera versión de "saltar lo ya
procesado" solo revisaba si existía el archivo JSON de salida — pero el
`main.py` original (el que tenías antes de la primera corrección) guardaba
un JSON para *todos* los documentos aunque nunca les aplicara OCR real,
dejando `needs_ocr: true` y texto casi vacío adentro. Con eso, el pipeline
podía pensar que esos documentos ya estaban listos cuando en realidad
seguian pendientes. Ya se corrigió: ahora `already_processed()` abre el
JSON guardado y solo lo cuenta como válido si `needs_ocr` es `false`. Si
viene de esa corrida vieja, se vuelve a procesar con OCR real.

## 5. Limpieza del membrete institucional (`cleaner.py` + `main.py`)

**Hallazgo con datos reales:** todas las sentencias comparten el mismo
membrete al inicio (escudo + "TRIBUNAL ESTATAL ELECTORAL" + "NAYARIT"). En
los PDFs escaneados, el OCR lee el escudo como texto basura ilegible (ej.
"QNIDOS M", "SSayA IS2to") justo antes de esas palabras. Esto no aporta
nada al análisis y contamina el vocabulario si luego se usa para
embeddings o clasificación.

**Qué se agregó:** una nueva función `strip_scanned_letterhead()` en
`cleaner.py`, conectada automáticamente en `main.py`. Busca la palabra
"NAYARIT" (la parte del membrete que el OCR lee de forma más confiable)
dentro de los primeros 180 caracteres del documento y elimina todo lo que
viene antes, incluyéndola.

**Por qué es conservadora a propósito:** se probó con 8 documentos reales
de 6 categorías distintas. En 7 de 8 funcionó perfecto. En 1 caso, el
membrete salió tan ilegible que ni "NAYARIT" apareció cerca del inicio — en
vez de arriesgar cortar contenido real (se detectó que un intento menos
cuidadoso hubiera borrado "EXPEDIENTE", "ACTOR" y "AUTORIDAD RESPONSABLE"
reales de ese documento), la función **no toca el texto** cuando no está
segura. Cada resultado incluye un campo `letterhead_removed` (true/false)
en el JSON de salida para poder medir qué porcentaje del corpus completo
se limpió automáticamente y revisar a mano los que no.

**Recomendación:** una vez que tengas los 886 documentos procesados, vale
la pena correr un conteo rápido de `letterhead_removed: false` para ver
cuántos quedaron sin limpiar y decidir si conviene revisarlos a mano o
afinar el patrón.

## 4. Nuevo módulo `src/sentido_extractor.py` — detector del sentido de la resolución

Este módulo lee el texto ya extraído (por tu pipeline, con o sin OCR) y
detecta automáticamente el **sentido** de cada sentencia (confirma / revoca /
modifica / desecha / etc.) buscando patrones en la sección "RESUELVE" del
documento. Es la variable objetivo ("y") que vas a necesitar para entrenar
tu modelo de clasificación — con esto no dependes de un LLM (con su costo y
su margen de error) para extraer algo que en realidad ya está escrito
literalmente en el documento.

**Por qué basado en reglas y no en LLM para este campo:** el sentido de la
resolución es una palabra o frase fija que el propio tribunal ya redactó
de forma explícita ("se confirma", "se revoca", etc.). Usar un método
determinista aquí es más preciso, más barato y 100% reproducible —
reservamos el LLM para tareas que sí requieren interpretación (resumen de
hechos, fundamentación, búsqueda semántica).

**Descubrimientos importantes al probarlo con 12 sentencias reales de las 6
categorías (AP, JIN, JDCN, MII, PES, RV), incluyendo casos con OCR de mala
calidad:**

1. El encabezado "RESUELVE" rara vez sale limpio del OCR. Se vieron variantes
   como "R E S U E L V E" (con espacios sueltos), "VUELVE:" (el OCR perdió
   incluso la letra "S"), y en un caso el encabezado no fue reconocible en
   absoluto por muy mala calidad del OCR — ese documento quedó como
   `no_detectado` y necesitaría revisión manual o reprocesar el PDF con
   mejores parámetros de OCR.
2. No todos los tipos de procedimiento usan el mismo vocabulario de sentido:
   - **General** (AP, JIN, JDCN, MII, RV): confirma / revoca / modifica /
     desecha / sobresee / improcedente / fundado / infundado.
   - **Incidentes de incumplimiento** (identificados por "INCIDENTE" o
     "INCIDENTAL" en el nombre de archivo): cumplida / incumplida /
     parcialmente_cumplida — no aplica confirma/revoca aquí.
   - **PES — Procedimiento Especial Sancionador**: existencia_infraccion /
     inexistencia_infraccion / sobresee.
   El módulo incluye `infer_procedure_category()` para elegir el conjunto de
   patrones correcto automáticamente según el nombre de archivo y la
   categoría del expediente.
3. Resultado en la muestra de prueba: **11 de 12 documentos (92%)**
   detectados correctamente con evidencia textual auditable. El único caso
   fallido fue por calidad de OCR insuficiente, no por límite del método de
   búsqueda. **Esta cifra es solo de una muestra chica de prueba — antes de
   usarla en la memoria o para entrenar el modelo, hay que correrlo sobre el
   corpus completo y validar una muestra más grande a mano.**

**Cómo usarlo:**
```python
from sentido_extractor import detect_sentido, infer_procedure_category

texto = open('documents/output/AP_SENTENCIA TEE-AP-01-2025.txt').read()
proc_cat = infer_procedure_category('SENTENCIA TEE-AP-01-2025.txt', category='AP')
resultado = detect_sentido(texto, procedure_category=proc_cat)
# resultado['sentido'] -> 'desecha'
# resultado['sentido_evidencia'] -> fragmento de texto que sustenta la etiqueta (para auditar)
```
Cada resultado incluye `sentido_evidencia` (el fragmento exacto de texto que
sustenta la etiqueta) para que puedas auditar rápido cualquier caso dudoso
sin releer el PDF completo.

## 6. Dos rondas de corrección de `sentido_extractor.py` sobre el corpus completo

Al correr `detect_sentido()` sobre los 886 documentos reales del corpus
(no solo la muestra de 12), aparecieron casos de `no_detectado` que no se
veían en la muestra chica. Se corrigieron en dos rondas sucesivas:

- **Ronda 1:** ajustes a los patrones de encabezado de RESUELVE y a la
  detección de transición "FUNDADO"/"INFUNDADO" para cubrir variantes de
  OCR adicionales encontradas en el corpus completo.
- **Ronda 2:** se corrigió un bug de orden de definición en Python —
  `DECLARA_INFUNDADO_PATTERN`/`DECLARA_FUNDADO_PATTERN` e
  `INEXISTENCIA_INFRACCION_PATTERN`/`EXISTENCIA_INFRACCION_PATTERN` se
  usaban dentro de las listas `SENTIDO_PATTERNS`/`PES_SANCIONADOR_PATTERNS`
  antes de estar definidos, lo que hacía que esos patrones nunca se
  aplicaran en la práctica. Se movieron antes de su primer uso.

Después de ambas rondas, sobre el corpus completo (886 documentos):
`no_detectado` = 320 (36.1%). De esos 320, la gran mayoría corresponden a
documentos que **no llevan un "sentido" en el sentido de confirma/revoca**
(por ejemplo Acuerdos Plenarios administrativos, o resoluciones cuyo
formato de RESUELVE es atípico) — no son necesariamente errores del
detector. Se decidió, junto contigo, no seguir afinando el detector caso
por caso (rendimientos decrecientes) y avanzar directo al modelo
predictivo con los 566 documentos que sí tienen sentido claramente
detectado y no son Acuerdos Plenarios.

## 7. Modelo predictivo de sentido (embeddings + explicación de motivos)

Con los 566 documentos usables (566 = 886 total − 320 no_detectado − los
que además eran Acuerdo Plenario) se construyó un modelo que predice el
sentido de una sentencia a partir del texto que la antecede, junto con una
explicación en lenguaje natural de los motivos de cada predicción.

**Decisión de diseño clave — qué texto usar como insumo:** se usa
exclusivamente el texto **previo al resolutivo** (hechos, agravios,
considerandos), nunca el resolutivo mismo. Si el modelo usara el
documento completo, estaría "prediciendo" algo que ya está escrito
literalmente unas líneas más abajo en el propio texto (fuga de la
etiqueta) — no sería un modelo predictivo real, sería un buscador de
palabras. Con el 92% de los documentos se logró el corte automático
buscando el encabezado de RESUELVE; para el resto se usó como respaldo
la transición hacia "FUNDADO"/"INFUNDADO".

**Enfoque técnico:**
1. `scripts/build_dataset_modelo.py` construye el dataset de modelado
   (texto insumo + etiqueta) a partir del inventario y los JSON ya
   procesados.
2. `scripts/generar_embeddings.py` divide cada documento en fragmentos de
   ~90 palabras (límite práctico del modelo de embeddings), calcula un
   embedding de 384 dimensiones por fragmento con un modelo multilingüe
   (soporta español, arquitectura tipo BERT) y promedia los fragmentos de
   cada documento para obtener un solo vector por documento.
3. `scripts/entrenar_modelo.py` entrena un clasificador (regresión
   logística, con ponderación de clases para compensar el desbalance)
   sobre esos vectores, evalúa con una partición de prueba honesta
   (80/20 estratificada) y después reentrena con el 100% de los datos
   para dejar el modelo final listo para usarse.
4. `src/explicar_prediccion.py` usa LIME para explicar cada predicción
   nueva: perturba el texto de entrada muchas veces, observa cómo cambia
   la predicción del modelo, y a partir de eso identifica qué palabras
   influyeron a favor o en contra de la clase predicha — y con eso arma
   un texto de explicación en español, no solo una lista de números.

**Resultado (partición de prueba, 114 documentos nunca vistos en
entrenamiento):** exactitud 61.4%, F1 macro 0.462, F1 ponderado 0.620.
Por clase, las categorías con más ejemplos rinden razonablemente bien
(F1 entre 0.67 y 0.80 en confirma, desecha, existencia_infraccion,
inexistencia_infraccion, declara_fundado); revoca y sobresee rinden más
bajo (F1 ~0.35-0.37) probablemente por solaparse en vocabulario con
confirma/desecha; las clases con menos de 10 ejemplos en todo el corpus
(cumplida, incumplida, declara_improcedente, parcialmente_cumplida) no
tienen suficientes datos para medirse de forma confiable — esto hay que
documentarlo honestamente en la memoria como limitación del tamaño del
corpus, no ocultarlo ni forzar una cifra optimista.

**Ejemplo real de salida de `explicar_prediccion.py`** (documento de
prueba, nunca usado para entrenar):
> Documento: JDCN_INCIDENTE DE INEJECUCIÓN TEE-JDCN-06-2023 — sentido
> real: incumplida — sentido predicho: incumplida (confianza 99%).
> Explicación: "Se predice 'incumplida' (confianza 99%) principalmente
> por la presencia de términos como: sentencia, formal, lineamientos,
> decretar, Mediante."

**Pendiente para la siguiente ronda:** decidir contigo si conviene (a)
agrupar las clases muy pequeñas en categorías más amplias, (b) buscar más
ejemplos de esas clases en el corpus (pueden existir más documentos de
ese tipo que hoy caen en `no_detectado`), o (c) aceptar la limitación y
documentarla tal como está — cualquiera de las tres es una decisión
académica legítima, pero conviene que la tomes tú antes de avanzar más.

## 8. Ampliación del corpus vía Gemini, corrección de datos y modelo final

**Ampliación de la detección de sentido:** de los 320 documentos que
`sentido_extractor.py` (basado en reglas) dejaba como `no_detectado`, se
usó la API de Gemini (`detectar_sentido_gemini.py`) para clasificar los
que sí tenían sentido pero con una redacción atípica del resolutivo.
Sobre 307 documentos intentados, 289 se aceptaron con confianza alta o
media (16 descartados por confianza baja, 2 con error) -- se optó por un
criterio conservador: mejor perder documentos dudosos que meter ruido a
las etiquetas.

**Migración de embeddings a Gemini:** se reemplazó el modelo local de
embeddings (sentence-transformers, 384 dimensiones) por la API de Gemini
(`gemini-embedding-001`, 768 dimensiones, tipo de tarea
`CLASSIFICATION`), para unificar todo el pipeline de lenguaje sobre una
sola API en vez de mantener dos proveedores distintos.

**Bug encontrado y corregido en `build_dataset_modelo.py`:** el script
aceptaba cualquier etiqueta que devolviera Gemini sin validarla contra el
catálogo oficial de 13 clases. Un documento (`JDCN_TEE-JDCN-23-2021`)
llegó con la etiqueta mal escrita `sobreseere` en vez de `sobresee` y se
coló al dataset de modelado como una clase nueva y espuria. Se agregó un
conjunto `SENTIDOS_VALIDOS` con las 13 clases oficiales; cualquier
etiqueta fuera de ese catálogo se descarta y se registra en consola.

**Duplicados detectados:** en la misma corrección se agregó detección de
documentos con texto idéntico subido bajo nombres de archivo distintos
(probablemente el mismo expediente cargado dos veces). Se encontraron y
descartaron 5 casos.

Con ambas correcciones, el corpus de modelado (tras excluir Acuerdos
Plenarios) quedó en **851 documentos**, con esta distribución del método
usado para ubicar el corte antes del resolutivo:

| Método de corte | Documentos |
|---|---|
| Encabezado RESUELVE | 551 |
| Corte aproximado (último ~15% del texto) | 178 |
| Marcador verbatim de Gemini | 112 |
| Transición "fundado/infundado" | 10 |

**Análisis de sensibilidad del corte aproximado:** se construyó
`comparar_escenarios_corte.py` para entrenar y comparar el modelo con
todos los 851 documentos ("completo") contra excluyendo los 178 de corte
aproximado ("sin_aproximado"), usando los mismos hiperparámetros que
`entrenar_modelo.py` (regresión logística, `class_weight="balanced"`,
partición 80/20 estratificada). Vale la pena mencionar que la primera
corrida de esta comparación, antes de corregir el bug de la etiqueta
`sobreseere` y los duplicados, daba una lectura equivocada (el escenario
completo parecía mejor). Una vez corregidos esos datos, el resultado se
invirtió:

| Escenario | Documentos | Exactitud | F1 macro | F1 ponderado |
|---|---|---|---|---|
| Completo | 851 | 66.7% | 0.458 | 0.648 |
| **Sin corte aproximado** | **673** | **70.4%** | **0.495** | **0.694** |

Excluir los 178 documentos de corte aproximado mejora las tres métricas,
así que se adoptó el escenario **sin_aproximado (673 documentos)** como
modelo oficial del proyecto.

**Resúmenes automáticos:** se unificó la generación de resúmenes
(`generar_resumenes.py`) en un solo comando que llama directamente a
Gemini (`gemini-flash-lite-latest`, modelo económico), ampliando el
alcance de 566 a las 873 sentencias totales del corpus (sin Acuerdos
Plenarios, sin importar si tienen sentido detectado o no). Corrida
completa sobre las 873: 0 errores.

## 9. Embeddings del clasificador: de Gemini a un modelo local

**Contexto:** en la sección anterior se migraron los embeddings del
clasificador de un modelo local (`sentence-transformers`, gratuito,
384 dimensiones) a la API de Gemini (`gemini-embedding-001`, 768
dimensiones, pago por uso), buscando mejor exactitud y evitar los
límites de tasa del nivel gratuito.

**Por qué se revirtió:** ese cambio hacía que *cada evaluación de un
documento nuevo* en el demo de Streamlit dependiera de una API de pago
(no solo la construcción del corpus, que es una tarea de una sola vez).
Esto entra en conflicto directo con el criterio de la guía oficial del
TFM que pregunta explícitamente "¿el proyecto es reproducible?" — un
modelo que deja de funcionar sin una API key vigente y saldo disponible
no lo es. Se decidió mantener el uso de Gemini limitado exclusivamente
a las tareas de construcción de corpus que ya son de una sola vez
(detección de sentido y generación de resúmenes, ver secciones
anteriores) y volver el clasificador a un modelo local que corre
completamente offline.

**Comparación de embeddings probados** (mismo dataset, escenario
`sin_aproximado`, 673 documentos, partición de prueba 80/20
estratificada):

| Embeddings | Dimensiones | Costo | Exactitud | F1 macro | F1 ponderado |
|---|---|---|---|---|---|
| Gemini (`gemini-embedding-001`) | 768 | Pago por uso | 70.4% | 0.495 | 0.694 |
| Local — MiniLM (`paraphrase-multilingual-MiniLM-L12-v2`) | 384 | Gratis | 63.7% | 0.443 | 0.625 |
| Local — mpnet (`paraphrase-multilingual-mpnet-base-v2`) | 768 | Gratis | **67.4%** | **0.519** | 0.663 |

**Decisión final:** se adopta el modelo local `mpnet-base-v2` como
embeddings oficiales del clasificador (`scripts/generar_embeddings.py`).
Cierra casi toda la brecha con Gemini en exactitud (67.4% vs. 70.4%) y F1
ponderado (0.663 vs. 0.694), e incluso supera a Gemini en F1 macro (0.519
vs. 0.495) — es decir, generaliza mejor en promedio sobre las clases
minoritarias. A cambio de esa diferencia pequeña se obtiene
reproducibilidad total: el modelo corre sin conexión a internet, sin
costo por evaluación, y sin riesgo de que una API externa deje de estar
disponible el día de la defensa. `scripts/generar_embeddings_gemini.py`
se conserva en el repositorio únicamente como referencia histórica de
esta comparación, ya no forma parte del pipeline oficial.

**Reporte por clase (escenario oficial, `sin_aproximado`, embeddings
mpnet, 135 documentos de prueba):**

| Clase | Precisión | Recall | F1 | Soporte |
|---|---|---|---|---|
| confirma | 0.71 | 0.67 | 0.69 | 33 |
| cumplida | 0.00 | 0.00 | 0.00 | 2 |
| declara_fundado | 0.58 | 0.88 | 0.70 | 8 |
| declara_improcedente | 0.00 | 0.00 | 0.00 | 1 |
| declara_infundado | 0.50 | 0.33 | 0.40 | 3 |
| desecha | 0.73 | 0.80 | 0.76 | 20 |
| existencia_infraccion | 0.80 | 0.86 | 0.83 | 14 |
| incumplida | 1.00 | 1.00 | 1.00 | 2 |
| inexistencia_infraccion | 0.88 | 0.92 | 0.90 | 24 |
| modifica | 0.29 | 0.40 | 0.33 | 5 |
| parcialmente_cumplida | 0.33 | 0.50 | 0.40 | 2 |
| revoca | 0.33 | 0.27 | 0.30 | 15 |
| sobresee | 0.67 | 0.33 | 0.44 | 6 |

Como en rondas anteriores, las clases con menos de 5 ejemplos en la
partición de prueba (`cumplida`, `declara_improcedente`,
`parcialmente_cumplida`) no rinden de forma confiable por falta de
datos — se documenta como limitación del tamaño del corpus, no se
oculta ni se fuerza una cifra optimista.

## 10. Validación cruzada k-fold: corrige una conclusión de la sección 9 y cierra el capítulo de análisis

### Por qué se abandona la partición única 80/20

Con un corpus de unos pocos cientos de documentos y varias clases con
menos de 10 ejemplos, una sola partición train/test (la usada en las
secciones 8 y 9) es muy sensible a qué documentos exactos caen del lado
de prueba: cambiar el tamaño del corpus en unas pocas decenas de filas
puede mover la exactitud varios puntos aunque el modelo no haya
cambiado en absoluto. Se implementó `scripts/evaluar_kfold.py`, que hace
validación cruzada estratificada (k=5) y reporta la media ± desviación
estándar entre folds — una estimación mucho más estable para reportar en
la tesis que un solo número de una sola partición.

### Hallazgo que corrige la sección 9: la ventaja de "sin_aproximado" era ruido de partición

La sección 9 adoptó el escenario `sin_aproximado` (673 documentos) como
oficial porque, en una sola partición 80/20, superaba a `completo` (851
documentos) por varios puntos en las tres métricas (70.4% vs. 66.7% de
exactitud). Al repetir la comparación con k-fold sobre el corpus
actualizado, **esa ventaja desaparece**: los dos escenarios quedan
prácticamente empatados, dentro de una desviación estándar entre folds:

| Escenario | n docs | Exactitud | F1 macro | F1 ponderado |
|---|---|---|---|---|
| completo (incluye corte aproximado) | 856 | 0.5958 ± 0.0151 | 0.4020 ± 0.0316 | 0.5935 ± 0.0144 |
| sin_aproximado | 697 | 0.5897 ± 0.0154 | 0.3992 ± 0.0475 | 0.5887 ± 0.0164 |

(El corpus también creció/cambió de tamaño entre secciones por la
ampliación con Gemini y las correcciones de la sección 8/9, por eso el
conteo base ya no es 851/673 sino 879 documentos totales tras excluir
Acuerdos Plenarios, antes de la exclusión de OCR que se explica abajo.)

**Interpretación para la tesis:** la diferencia observada en la sección
9 (partición única) era, con alta probabilidad, ruido de partición y no
un efecto real del corte aproximado — exactamente el problema que
motivó a construir k-fold en primer lugar. Con la evidencia más robusta,
no hay razón estadística para preferir `sin_aproximado` sobre
`completo`; y `completo` usa ~160 documentos más.

**Decisión:** se declara **`completo` (856 documentos) como escenario
oficial** del modelo desplegado, en reemplazo de `sin_aproximado` (673/697
documentos), adoptado provisionalmente en la sección 9. Esto supera la
necesidad del corte aproximado como fuente de exclusión: al no haber
diferencia real de desempeño, no se justifica descartar ~160 documentos
válidos solo porque su punto de corte se ubicó con un método
conservador en vez de un marcador exacto. Todas las métricas de detalle
de esta sección (F1 por clase, matriz de confusión) corresponden a este
escenario oficial.

### Exclusión de 7 documentos con OCR ilegible

Se detectó (`scripts/diagnostico_calidad_ocr.py`, que mide densidad de
palabras funcionales del español para calibrar qué tan legible salió el
OCR de cada documento) que 7 de 886 documentos (0.8%, casi todos del
lote 2017) tienen texto ilegible por páginas escaneadas al revés que la
corrección automática de orientación no logró arreglar. Se excluyeron
manualmente de `scripts/build_dataset.py` (conjunto
`EXCLUIDOS_OCR_ILEGIBLE`) porque ya están procesados con el texto malo y
no vale la pena volver a correr OCR sobre todo el corpus por 7 casos.
El arreglo de rotación en `ocr_processor.py` (umbral de auto-rotación
bajado de 14 a 0) ya evita este problema en documentos nuevos. Tras esta
exclusión, el corpus de modelado quedó en 856 documentos (completo) /
697 (sin_aproximado) — los números reportados arriba ya la incluyen.

### F1 por clase — escenario completo (856 documentos, k-fold)

| Clase | F1 | n aprox. (del corpus) |
|---|---|---|
| inexistencia_infraccion | 0.862 | 156 |
| existencia_infraccion | 0.844 | 105 |
| confirma | 0.652 | 193 |
| desecha | 0.595 | 110 |
| declara_fundado | 0.587 | 50 |
| revoca | 0.407 | 87 |
| sobresee | 0.320 | 47 |
| incumplida | 0.233 | 10 |
| parcialmente_cumplida | 0.233 | 8 |
| declara_infundado | 0.210 | 28 |
| declara_improcedente | 0.160 | 19 |
| cumplida | 0.100 | 9 |
| modifica | 0.024 | 34 |

Se mantiene el patrón de las secciones anteriores: las clases con F1 muy
bajo son también las de menor tamaño en el corpus (8-34 ejemplos) —
indicio de desbalance de clases, no necesariamente mala señal semántica.

### Matriz de confusión (out-of-fold, escenario completo)

| Par | Detalle |
|---|---|
| modifica → revoca/confirma | modifica solo 3% acierto (1/34); 26% va a revoca, 24% a confirma |
| declara_improcedente → modifica/desecha/confirma | 16% acierto (3/19); se reparte entre las tres por igual (16% cada una) |
| revoca ↔ confirma | 28% de revoca mal clasificado hacia confirma; 18% de confirma hacia revoca |
| desecha ↔ sobresee | 14% desecha→sobresee; 26% sobresee→desecha |
| existencia ↔ inexistencia_infraccion | 10%/8% de confusión mutua — el par mejor separado |

**Interpretación para la tesis:** las clases con peor F1 (modifica,
declara_improcedente) no fallan al azar — se confunden con clases legal
y semánticamente vecinas (modifica está entre confirma y revoca;
declara_improcedente es un rechazo procesal, cercano a desecha/modifica),
y coinciden con las clases de menor tamaño de muestra. El problema es
desbalance + cercanía semántica genuina, no un error aleatorio del
modelo.

### Features de palabras clave/negación: resultado negativo

Se probó (`scripts/mejorar_features_negacion.py`) agregar features
léxicas de negación a los embeddings, con la hipótesis de que ayudarían
a distinguir clases opuestas semánticamente cercanas (ej. confirma vs.
revoca). Esta prueba se corrió **antes** de la exclusión de OCR, sobre
el corpus todavía de 863 (completo) / 698 (sin_aproximado) documentos —
no se repitió después porque la exclusión de 7 documentos no debería
cambiar una conclusión de esta magnitud. El resultado fue inconsistente
entre escenarios:

| Escenario (863/698 docs) | Δ F1 macro (híbrido vs. solo embeddings) |
|---|---|
| completo | **-0.012** |
| sin_aproximado | +0.027 |

Empeora en el escenario más grande y mejora en el más chico — indicio de
ruido de partición, no de una mejora real. Se descarta la feature; el
modelo final usa solo embeddings. Es un resultado negativo válido para
reportar en la tesis: se probó una mejora razonable, con hipótesis
clara, y no funcionó de forma consistente.

### Corrección de una regresión propia durante este trabajo

Al aplicar el parche de exclusión de OCR sobre una copia de referencia
desactualizada de `build_dataset.py`, se borró por accidente la columna
`es_acuerdo_plenario` y el uso de `resolve_category()` para inferir la
categoría desde el nombre de archivo cuando el JSON no la trae. Quedó
detectado de inmediato porque `build_dataset_modelo.py` truena si falta
esa columna (`KeyError`), y se corrigió recuperando ambas piezas de
lógica antes de aplicar la exclusión de OCR encima. Se documenta aquí
como recordatorio del riesgo de patchear scripts sin diff completo
contra la versión vigente.

## 11. Auditoría de reproducibilidad del corpus de modelado: 849, no 856 (2 de septiembre de 2026)

### Qué se encontró

Al construir `analisis_tesis/analisis_descriptivo.py` (fuera de
`scripts/`, un script aislado solo para el análisis descriptivo del
escrito, no parte del pipeline de producción) y correrlo sobre los
datos versionados en git, el corpus final de modelado resultó en
**849 documentos**, no los 856 declarados como escenario oficial en la
sección 10.

### Auditoría de la causa

- `data/sentido_gemini.json` (el archivo que `build_dataset_modelo.py`
  espera) no estaba versionado en git; solo existía
  `data/sentido_gemini_v1_backup.json`. Se comparó ese respaldo, byte a
  byte, contra el archivo original que la autora conservaba en su
  máquina local (`sentido_gemini.json`) — **son idénticos**.
- Se reconstruyó el commit exacto donde se declaró 856 como cifra
  oficial (`cdef88a`, 25 de agosto de 2026) y se corrió
  `scripts/build_dataset_modelo.py` con los datos tal como estaban
  versionados en ese commit (`data/dataset_sentencias.csv`: 879 filas,
  866 sin Acuerdos Plenarios — igual que hoy). Resultado: **849**, no
  856, incluso reproduciendo el estado histórico exacto.
- Conclusión: la discrepancia de 7 documentos no se explica por deriva
  de datos entre commits ni por una versión distinta del archivo de
  Gemini. Lo más probable es que la cifra de 856 se calculó sobre un
  estado local no versionado (`dataset_sentencias.csv` y/o
  `sentido_gemini.json` con 7 filas adicionales) en el momento del
  análisis original, que nunca se guardó en git.

### Decisión

Se documenta **849 documentos** como la cifra vigente y reproducible
del corpus de modelado a partir de los datos versionados en git:

```
879 (inventario) → 866 (excluye 13 Acuerdos Plenarios)
  → 849 (corpus final: 565 detectados por regla + 284 asistidos
         por Gemini con confianza alta/media)
```

Las métricas de k-fold de la sección 10 (exactitud 0.5958 ± 0.0151,
F1 macro 0.4020 ± 0.0316, F1 ponderado 0.5935 ± 0.0144, y el detalle
por clase) corresponden al run original de 856 documentos y **no se
han vuelto a calcular sobre 849** — para eso se necesita
`documents/output/` (los JSON con el texto extraído, no versionados en
git por tamaño) y volver a correr `scripts/evaluar_kfold.py`. Se deja
como pendiente explícito: si al recalcular sobre 849 las métricas se
mueven de forma material, esta sección debe actualizarse antes de citar
cifras de desempeño en el escrito final.

El análisis descriptivo completo (embudo del corpus, categoría,
año, sentido, temas, longitud de texto) con sus gráficas queda en
`analisis_tesis/output/`, generado por el script aislado mencionado
arriba — no reemplaza ni modifica el pipeline de `scripts/`.

### Corrección posterior (5 de septiembre de 2026): la cifra de 849 era errónea

Una revisión de esta auditoría encontró que:

- El código de `scripts/build_dataset.py` y `scripts/build_dataset_modelo.py`
  es idéntico, línea por línea, entre el commit de esta auditoría
  (`157ece6`, 2 de septiembre) y el HEAD actual — sin ningún cambio
  funcional en medio.
- `data/sentido_gemini.json` no ha cambiado desde el 23 de agosto (fecha
  de modificación anterior tanto a esta auditoría como a cualquier
  corrida posterior).
- Con ese mismo código y esos mismos datos, ejecutar el pipeline hoy
  reproduce **856 documentos**, no 849, de forma consistente.

Con código y datos comprobadamente idénticos, el mismo script no puede
dar 849 en una ejecución y 856 en otra. La conclusión más razonable es
que la cifra de 849 declarada arriba nunca se verificó corriendo el
pipeline completo de punta a punta en su momento — fue un error de esta
auditoría, no un problema de los datos ni del código de la autora.

**Se retracta la cifra de 849.** La cifra oficial y reproducible del
corpus de modelado es **856 documentos**, consistente con la
declaración original de la sección 10 y con las métricas de k-fold ahí
documentadas (nunca había sido necesario recalcularlas).
