"""Genera la Figura -- Matriz de confusion (out-of-fold, escenario completo)
para el escrito del TFM.

Como analisis_descriptivo.py, vive fuera de scripts/ a proposito: es una
capa de reporte, no parte del pipeline de produccion. Lee el CSV que ya
produce scripts/diagnostico_confusion.py (no recalcula nada):

    python scripts/diagnostico_confusion.py --k 5 --escenario completo
    # genera modelo/matriz_confusion_completo.csv (columnas = clase
    # predicha, filas = clase real; ver README.md para el comando completo)

Uso:
    python analisis_tesis/matriz_confusion_heatmap.py

Guarda analisis_tesis/output/figura_matriz_confusion.png (paleta Nexus,
escala secuencial teal, normalizada por fila / recall).
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- datos: matriz de confusión completa (out-of-fold, escenario completo, k=5, n=856) ---
# generada por scripts/diagnostico_confusion.py (no versionada: modelo/ esta
# en .gitignore porque son artefactos regenerables del modelo entrenado)
df = pd.read_csv('/home/user/workspace/repo_check/modelo/matriz_confusion_completo.csv', index_col=0)
labels_raw = list(df.index)
matrix = df.values.astype(float)

# nombres cortos para las etiquetas (legibilidad en ejes)
nombre_corto = {
    'confirma': 'Confirma',
    'cumplida': 'Cumplida',
    'declara_fundado': 'Declara\nfundado',
    'declara_improcedente': 'Declara\nimprocedente',
    'declara_infundado': 'Declara\ninfundado',
    'desecha': 'Desecha',
    'existencia_infraccion': 'Existencia\ninfracción',
    'incumplida': 'Incumplida',
    'inexistencia_infraccion': 'Inexistencia\ninfracción',
    'modifica': 'Modifica',
    'parcialmente_cumplida': 'Parcialmente\ncumplida',
    'revoca': 'Revoca',
    'sobresee': 'Sobresee',
}
labels = [nombre_corto[l] for l in labels_raw]
n = len(labels)

support = matrix.sum(axis=1)
pct = matrix / support[:, None] * 100  # normalizado por fila (recall)

# --- paleta: escala secuencial monocromática (teal, Nexus/diseño) ---
from matplotlib.colors import LinearSegmentedColormap
teal_cmap = LinearSegmentedColormap.from_list('teal_seq', ['#F7F6F2', '#BCE2E7', '#20808D', '#1B474D'])

fig, ax = plt.subplots(figsize=(11, 9.5), dpi=200)
im = ax.imshow(pct, cmap=teal_cmap, vmin=0, vmax=100, aspect='equal')

ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9, color='#28251D')
ax.set_yticklabels(labels, fontsize=9, color='#28251D')
ax.set_xlabel('Clase predicha', fontsize=12, color='#28251D', labelpad=10)
ax.set_ylabel('Clase real', fontsize=12, color='#28251D', labelpad=10)
ax.set_title('Matriz de confusión (out-of-fold, escenario completo, k=5, n=856)',
             fontsize=13, color='#28251D', pad=14, loc='left')

# líneas de separación entre celdas
ax.set_xticks(np.arange(n+1)-0.5, minor=True)
ax.set_yticks(np.arange(n+1)-0.5, minor=True)
ax.grid(which='minor', color='#F7F6F2', linewidth=1.5)
ax.tick_params(which='minor', bottom=False, left=False)
ax.tick_params(which='major', bottom=False, left=False)

# resaltar la diagonal (aciertos) con un borde
for i in range(n):
    ax.add_patch(plt.Rectangle((i-0.5, i-0.5), 1, 1, fill=False,
                                edgecolor='#A84B2F', linewidth=2.2))

# anotaciones: conteo absoluto (y % en fila si > 0)
for i in range(n):
    for j in range(n):
        count = int(matrix[i, j])
        if count == 0:
            continue
        p = pct[i, j]
        text_color = '#F7F6F2' if p > 55 else '#28251D'
        weight = 'bold' if i == j else 'normal'
        ax.text(j, i, f'{count}', ha='center', va='center',
                 fontsize=8.5, color=text_color, weight=weight)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
cbar.set_label('% de la clase real (por fila)', fontsize=10, color='#28251D')
cbar.ax.tick_params(labelsize=8.5, colors='#28251D')

fig.text(0.01, 0.01,
         'Nota: el color y el porcentaje representan la proporción respecto al total de\n'
         'cada clase real (recall por fila); las cifras dentro de cada celda son conteos absolutos.\n'
         'El recuadro naranja marca la diagonal (predicciones correctas).',
         fontsize=8, color='#7A7974')

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig('/home/user/workspace/analisis_tesis/output/figura_matriz_confusion.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print("guardado")
