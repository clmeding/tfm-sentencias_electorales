#!/bin/bash
# Mueve los scripts de análisis/diagnóstico (ya usados para el escrito del TFM,
# no forman parte del pipeline en producción) a analisis_tesis/.
# Usa "git mv" para conservar el historial de cada archivo.
#
# Uso: correr desde la raíz del repositorio.
#   bash reorganizar_analisis_tesis.sh

set -e

mkdir -p analisis_tesis

ARCHIVOS=(
  "comparar_escenarios_corte.py"
  "comparar_modelos_clasificacion.py"
  "comparar_precedentes_tfidf_vs_semantico.py"
  "diagnostico_calidad_ocr.py"
  "diagnostico_confusion.py"
  "diagnostico_precedentes.py"
  "diagnostico_sentido.py"
  "evaluar_kfold.py"
  "mejorar_features_negacion.py"
  "resumen_temas_libres.py"
  "validar_reranking_temas.py"
  "generar_embeddings_gemini.py"
)

for archivo in "${ARCHIVOS[@]}"; do
  if [ -f "scripts/$archivo" ]; then
    git mv "scripts/$archivo" "analisis_tesis/$archivo"
    echo "Movido: scripts/$archivo -> analisis_tesis/$archivo"
  else
    echo "AVISO: scripts/$archivo no existe, se omite"
  fi
done

echo ""
echo "Listo. Revisa 'git status' y luego:"
echo "  git commit -m \"refactor: mueve scripts de analisis/diagnostico a analisis_tesis/\""
