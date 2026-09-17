"""Genera Figura 4 — Flujo de deteccion del sentido (cascada regla -> API).

Mismo estilo visual que pipeline_extraccion.png (Figura 1): cajas color
crema con borde gris, rombos de decision, flechas grises curvas, y una
caja final resaltada en teal para marcar el resultado.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
from matplotlib.patches import FancyArrowPatch

# Paleta Nexus (misma usada en el resto del proyecto)
BG = "#FFFFFF"
BOX_FILL = "#F9F8F5"
BOX_EDGE = "#B9B6AE"
TEXT = "#28251D"
TEXT_MUTED = "#7A7974"
ARROW = "#9A978F"
FINAL_FILL = "#EAF4F4"
FINAL_EDGE = "#01696F"
FINAL_TEXT = "#0C4E54"
EXCLUDED_FILL = "#F7F1EE"
EXCLUDED_EDGE = "#B98A6A"

fig, ax = plt.subplots(figsize=(9.2, 10.6), dpi=200)
ax.set_xlim(0, 920)
ax.set_ylim(180, 1230)
ax.axis("off")
fig.patch.set_facecolor(BG)


def box(cx, cy, w, h, text, fill=BOX_FILL, edge=BOX_EDGE, tcolor=TEXT, fontsize=13, weight="normal", lw=1.1):
    b = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle=f"round,pad=0,rounding_size=14",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=2,
    )
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
             color=tcolor, weight=weight, linespacing=1.35, zorder=3,
             family="DejaVu Sans")
    return (cx, cy, w, h)


def diamond(cx, cy, w, h, text, fontsize=13):
    pts = [(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)]
    d = Polygon(pts, closed=True, linewidth=1.1, edgecolor=BOX_EDGE,
                facecolor=BOX_FILL, zorder=2)
    ax.add_patch(d)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize,
             color=TEXT, linespacing=1.3, zorder=3, family="DejaVu Sans")
    return (cx, cy, w, h)


def arrow(p1, p2, rad=0.0, label=None, label_pos=0.55, label_color=TEXT_MUTED, lx_off=0, ly_off=0):
    a = FancyArrowPatch(
        p1, p2, connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>", mutation_scale=14, linewidth=1.2,
        color=ARROW, zorder=1,
    )
    ax.add_patch(a)
    if label:
        mx = p1[0] + (p2[0] - p1[0]) * label_pos + lx_off
        my = p1[1] + (p2[1] - p1[1]) * label_pos + ly_off
        ax.text(mx, my, label, ha="center", va="center", fontsize=11.5,
                color=label_color, style="italic", family="DejaVu Sans", zorder=4)


# --- Nodo A: entrada ---
A = box(460, 1170, 560, 78,
        "Texto extraído y limpio\ncategoría de procedimiento → grupo\n(general / incumplimiento / pes_sancionador)")

# --- Rombo B: intento por reglas ---
B = diamond(460, 1010, 460, 170, "¿Coincide algún patrón\nléxico en la sección\n\"RESUELVE\"?")

# --- Caja C: etiqueta por regla (rama Sí) ---
C = box(215, 800, 380, 78, "Etiqueta de sentido + evidencia\ntextual (detección por reglas)")

# --- Caja D: Gemini (rama No) ---
D = box(680, 800, 340, 64, "Segunda opción: API de Gemini",
        weight="normal")

# --- Rombo E: confianza ---
E = diamond(680, 620, 380, 160, "¿Confianza del\nresultado alta\no media?")

# --- Caja F: etiqueta por Gemini (rama Sí) ---
F = box(340, 430, 380, 78, "Etiqueta de sentido + evidencia\n(asistido por Gemini)")

# --- Caja G: excluido (rama No) ---
G = box(750, 430, 300, 90, "Documento excluido del\ncorpus de modelado\n(confianza baja / error de API)",
        fill=EXCLUDED_FILL, edge=EXCLUDED_EDGE, tcolor="#7A4A30", fontsize=11.5)

# --- Caja final H ---
H = box(460, 260, 480, 84, "Corpus de modelado\ncon sentido etiquetado",
        fill=FINAL_FILL, edge=FINAL_EDGE, tcolor=FINAL_TEXT, fontsize=14, weight="bold", lw=1.4)

# Flechas
arrow((460, 1131), (460, 1095))  # A -> B
arrow((280, 940), (215, 839), rad=-0.15, label="Sí", label_pos=0.5, lx_off=-14)   # B -> C
arrow((610, 950), (680, 843), rad=0.15, label="No", label_pos=0.45, lx_off=16)    # B -> D
arrow((680, 768), (680, 700))     # D -> E
arrow((590, 545), (410, 469), rad=-0.18, label="Sí", label_pos=0.5, lx_off=-16)   # E -> F
arrow((750, 545), (750, 475))     # E -> G
arrow((260, 761), (340, 302), rad=-0.12)   # C -> H
arrow((360, 391), (430, 302), rad=-0.08)   # F -> H

plt.tight_layout(pad=0.6)
out = "/home/user/workspace/analisis_tesis/output/04_flujo_deteccion_sentido.png"
plt.savefig(out, facecolor=BG, bbox_inches="tight")
print("guardado en", out)
