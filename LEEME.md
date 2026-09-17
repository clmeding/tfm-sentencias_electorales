# LEEME — Demo Streamlit (evaluación de documentos)

## Modelo de embeddings usado

El clasificador y la recuperación de precedentes usan un modelo local de
`sentence-transformers`:

```
sentence-transformers/paraphrase-multilingual-mpnet-base-v2
```

(768 dimensiones). No requiere API key ni conexión a internet para
generar embeddings de documentos nuevos — solo necesita haberse
descargado una vez.

## Descarga previa

La primera vez que se usa el modelo, `sentence-transformers` lo descarga
automáticamente desde Hugging Face (~470 MB) y lo guarda en caché local.
Para no depender de internet el día de la defensa, descárgalo con
anticipación corriendo:

```
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')"
```

Esto puede tardar varios minutos según la conexión. Al terminar, el
modelo queda guardado en:

```
~/.cache/huggingface/hub/
```

(o `~/.cache/torch/sentence_transformers/` en versiones anteriores de la
librería).

## Verificar que quedó en caché

Para confirmar que el modelo ya no necesita descargarse, desconecta el
wifi y corre:

```
python -c "from sentence_transformers import SentenceTransformer; m = SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2'); print(m.encode('prueba').shape)"
```

Si imprime `(768,)` sin error de red, el modelo está listo para usarse
sin conexión.

## Correr la demo

```
streamlit run demo/streamlit_app.py
```

Se abre en el navegador local (`http://localhost:8501`), sin necesidad
de internet una vez que el modelo está en caché.
