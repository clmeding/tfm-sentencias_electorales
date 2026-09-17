"""Detecta el "sentido" de las sentencias que el método basado en reglas
(src/sentido_extractor.py) no pudo clasificar, usando la API de Gemini.

Contexto: build_dataset_modelo.py excluye del corpus de modelado todo
documento con sentido == "no_detectado" (actualmente ~307 de 873 sentencias
sin Acuerdos Plenarios). La mayoría falla por ruido de OCR: o no se
encontró el encabezado "RESUELVE" (método "sin_seccion_resuelve", ~123
casos) o sí se encontró el resolutivo pero ninguna palabra clave hizo
match dentro de él (método "sin_coincidencias", ~184 casos). Un LLM puede
leer el texto con más tolerancia al ruido que un regex.

Este script NO modifica data/dataset_sentencias.csv (la fuente de verdad
del pipeline de extracción). Genera un archivo aparte,
data/sentido_gemini.json, con una fila por documento intentado:

    {
        "json_filename": "...",
        "sentido": uno de los 13 valores conocidos, o "no_aplica",
        "confianza": "alta" | "media" | "baja",
        "marcador_resolutivo": primeras 8-15 palabras VERBATIM de donde
            empieza la parte resolutiva (para poder cortar el texto previo
            al resolutivo sin filtrar la respuesta al modelo predictivo),
        "evidencia": frase corta que justifica el sentido elegido,
    }

build_dataset_modelo.py combina este archivo con el CSV: solo usa el
sentido de Gemini cuando confianza es "alta" o "media" (umbral
configurable abajo); los casos "baja" o "no_aplica" se quedan fuera del
entrenamiento, igual que antes.

Preparación (una sola vez, se reutiliza la misma key de scripts/generar_resumenes.py):
    export GEMINI_API_KEY="tu-api-key-aqui"

Uso (desde la raíz del repo):
    python scripts/detectar_sentido_gemini.py

Guarda el avance después de CADA documento -- si se interrumpe, se puede
volver a correr y continúa donde se quedó.
"""

import json
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from sentido_extractor import infer_procedure_category

CSV_INVENTARIO = Path("data/dataset_sentencias.csv")
CORPUS_DIR = Path("documents/output")
SALIDA = Path("data/sentido_gemini.json")

API_KEY = os.environ.get("GEMINI_API_KEY")

MODELO = "gemini-flash-lite-latest"  # mismo modelo económico usado en generar_resumenes.py
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODELO}:generateContent"

# Umbral de confianza mínimo para que build_dataset_modelo.py use la
# etiqueta de Gemini. Los casos por debajo de este umbral se quedan como
# no_detectado, igual que si nunca se hubiera corrido este script.
CONFIANZA_ACEPTABLE = {"alta", "media"}

LABELS_GENERAL = [
    "revoca", "modifica", "confirma", "desecha", "sobresee",
    "declara_improcedente", "declara_infundado", "declara_fundado",
]
LABELS_INCUMPLIMIENTO = ["cumplida", "incumplida", "parcialmente_cumplida"]
LABELS_PES = ["inexistencia_infraccion", "existencia_infraccion", "sobresee"]

CATEGORY_LABELS = {
    "general": LABELS_GENERAL,
    "incumplimiento": LABELS_INCUMPLIMIENTO,
    "pes_sancionador": LABELS_PES + LABELS_GENERAL,
}

PROMPT_SISTEMA = """Eres un asistente jurídico especializado en sentencias del Tribunal Electoral del Estado de Nayarit (TEEN), para un proyecto académico de tesis de maestría.

El texto que te doy proviene de OCR y puede tener errores de reconocimiento de caracteres (letras cambiadas, palabras cortadas, símbolos sueltos). Tu tarea es encontrar la parte RESOLUTIVA de la sentencia (donde el Tribunal declara su decisión final, normalmente después de un encabezado tipo "RESUELVE", "SE RESUELVE" o "POR LO EXPUESTO Y FUNDADO, SE:", aunque el OCR puede haberlo deformado) y clasificar el sentido de esa decisión.

Elige EXACTAMENTE UNA etiqueta de esta lista (la que mejor describa la decisión principal, la más relevante si hay varias):
{etiquetas}

Usa estas definiciones EXACTAS (no tu propio criterio general) para elegir la etiqueta correcta, sobre todo para distinguir pares que suenan parecido en lenguaje cotidiano pero tienen efectos jurídicos distintos:
- confirma: se declara correcto el acto impugnado; los agravios se consideran infundados; el acto NO se modifica.
- modifica: se da la razón PARCIALMENTE al actor; el acto sigue vigente pero con los cambios ordenados por el Tribunal.
- revoca: se da la razón al actor; el acto impugnado queda TOTALMENTE sin efecto (no como modifica, que solo lo ajusta).
- desecha: el medio de impugnación NO se admite a trámite, generalmente por extemporáneo o por falta de un requisito de forma evidente desde el inicio; NO hay estudio de fondo.
- sobresee: el medio de impugnación SÍ fue admitido a trámite, pero se termina sin resolver el fondo por una causa sobrevenida durante el proceso (desistimiento del actor, el acto impugnado quedó sin materia, fallecimiento del promovente). A diferencia de desecha, aquí el juicio ya se había admitido.
- declara_improcedente: existe un obstáculo jurídico de FONDO (no de forma ni de plazo) que impide analizar el asunto -- por ejemplo falta de interés jurídico del actor o incompetencia del Tribunal. Puede detectarse en cualquier etapa, a diferencia de desecha.
- declara_fundado: se estudia el fondo de un agravio, recurso o incidente (no un acto de otra autoridad electoral como en confirma/modifica/revoca) y se acoge la pretensión del promovente.
- declara_infundado: se estudia el fondo, pero NO se acoge la pretensión del promovente.
- cumplida: la autoridad responsable acató, en tiempo y forma, la TOTALIDAD de lo ordenado en una sentencia previa del Tribunal.
- incumplida: la autoridad responsable NO acató lo ordenado en una sentencia previa (omisión total o insuficiente).
- parcialmente_cumplida: la autoridad responsable acató solo UNA PARTE de lo ordenado en una sentencia previa, quedando pendiente el resto.
- existencia_infraccion: (solo procedimientos especiales sancionadores) se acredita que la conducta denunciada sí violó la normativa electoral; hay sanción.
- inexistencia_infraccion: (solo procedimientos especiales sancionadores) NO se acredita que la conducta denunciada haya constituido una infracción.

Si después de leer el texto no puedes determinar con razonable certeza cuál de esas etiquetas aplica, responde "sentido": "no_aplica" y "confianza": "baja".

Responde ÚNICAMENTE con un objeto JSON válido con exactamente estas claves, sin texto adicional, sin markdown:
- "sentido": una de las etiquetas de la lista, o "no_aplica"
- "confianza": "alta", "media" o "baja"
- "marcador_resolutivo": copia LITERAL (verbatim, tal como aparece en el texto, 8 a 15 palabras) del inicio de la frase donde comienza la parte resolutiva. Debe ser una subcadena exacta del texto que te di, para poder ubicarla con una búsqueda de texto. Si no identificas un punto de inicio claro, usa cadena vacía "".
- "evidencia": frase corta (máx 25 palabras) del texto que justifica el sentido elegido."""


def cargar_pendientes() -> list[dict]:
    df = pd.read_csv(CSV_INVENTARIO)
    pendientes_df = df[(df["es_acuerdo_plenario"] == False) & (df["sentido"] == "no_detectado")]

    documentos = []
    faltantes = []
    for _, row in pendientes_df.iterrows():
        json_path = CORPUS_DIR / row["json_filename"]
        if not json_path.exists():
            faltantes.append(row["json_filename"])
            continue
        data = json.loads(json_path.read_text(encoding="utf-8"))
        documentos.append(
            {
                "json_filename": row["json_filename"],
                "category": row.get("category", ""),
                "texto": data.get("extracted_text_clean", "") or "",
            }
        )

    if faltantes:
        print(f"AVISO: {len(faltantes)} documentos del inventario no tienen archivo en {CORPUS_DIR}/.")

    return documentos


def llamar_gemini(client: httpx.Client, texto: str, etiquetas: list[str], intentos: int = 5) -> dict:
    texto_recortado = texto[:60000]
    prompt = PROMPT_SISTEMA.format(etiquetas=", ".join(etiquetas))
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": f"{prompt}\n\nTEXTO DE LA SENTENCIA:\n{texto_recortado}"}],
            }
        ],
        "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
    }
    espera = 8
    for _ in range(intentos):
        resp = client.post(URL, params={"key": API_KEY}, json=payload, timeout=90)
        if resp.status_code == 429:
            time.sleep(espera)
            espera = min(espera * 2, 60)
            continue
        resp.raise_for_status()
        data = resp.json()
        texto_respuesta = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(texto_respuesta)
    resp.raise_for_status()


def main() -> None:
    if not API_KEY:
        sys.exit(
            "ERROR: no encontré la variable de entorno GEMINI_API_KEY.\n"
            "Defínela antes de correr el script (la misma que usas para generar_resumenes.py)."
        )

    documentos = cargar_pendientes()
    print(f"Documentos 'no_detectado' a intentar clasificar con Gemini: {len(documentos)}")

    resultados = []
    if SALIDA.exists():
        resultados = json.loads(SALIDA.read_text(encoding="utf-8"))
    resultados = [r for r in resultados if "error" not in r]
    ya_hechos = {r["json_filename"] for r in resultados}
    pendientes = [d for d in documentos if d["json_filename"] not in ya_hechos]

    print(f"Ya procesados: {len(ya_hechos)} | pendientes en esta corrida: {len(pendientes)}")

    if not pendientes:
        print("No hay documentos pendientes -- todo está al día.")
        return

    SALIDA.parent.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        for i, doc in enumerate(pendientes, 1):
            jf = doc["json_filename"]
            etiquetas = CATEGORY_LABELS.get(
                infer_procedure_category(jf, doc.get("category")), LABELS_GENERAL
            )
            try:
                r = llamar_gemini(client, doc["texto"], etiquetas)
                sentido = r.get("sentido", "no_aplica")
                confianza = r.get("confianza", "baja")
                usable = sentido != "no_aplica" and confianza in CONFIANZA_ACEPTABLE
                resultados.append(
                    {
                        "json_filename": jf,
                        "sentido": sentido,
                        "confianza": confianza,
                        "marcador_resolutivo": r.get("marcador_resolutivo", ""),
                        "evidencia": r.get("evidencia", ""),
                    }
                )
                estado = "OK" if usable else "descartado (confianza/no_aplica)"
                print(f"[{i}/{len(pendientes)}] {estado}: {jf} -> {sentido} ({confianza})")
            except Exception as error:
                print(f"[{i}/{len(pendientes)}] ERROR en {jf}: {error}", file=sys.stderr)
                resultados.append({"json_filename": jf, "error": str(error)})

            SALIDA.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
            time.sleep(2.5)

    ok = [r for r in resultados if "error" not in r and r.get("sentido") != "no_aplica" and r.get("confianza") in CONFIANZA_ACEPTABLE]
    errores = [r for r in resultados if "error" in r]
    print(f"\nListo. Total intentados: {len(resultados)} | usables (confianza alta/media): {len(ok)} | errores: {len(errores)}")


if __name__ == "__main__":
    main()
