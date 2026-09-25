# -*- coding: utf-8 -*-
"""
Shared publication style for all redrawn MPCE figures.
Okabe-Ito color-blind-safe palette; Arial; 8-9 pt; de-junked axes.
Each plot_fig*.py imports this. Do not change colors/sizes without updating all figures.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Okabe-Ito (CVD-safe)
BLUE   = "#0072B2"
VERM   = "#D55E00"
GREEN  = "#009E73"
PURPLE = "#CC79A7"
ORANGE = "#E69F00"
SKY    = "#56B4E9"
GRAY   = "#999999"
GRID   = "#D9D9D9"

rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.dpi": 120,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,   # TrueType in PDF (editable/text)
    "svg.fonttype": "none",
})

# Display width kept equal to original inline width (5.833 in) so the docx
# layout width is unchanged; height follows each figure's aspect ratio.
DISP_W = 5.833  # inches

def despine(ax, keep_left=True, keep_bottom=True):
    """Remove top/right spines, lighten remaining."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(keep_left)
    ax.spines["bottom"].set_visible(keep_bottom)

def ygrid(ax):
    ax.grid(axis="y", color=GRID, alpha=0.45, linewidth=0.7)
    ax.set_axisbelow(True)

def save(fig, out_stem, outdir):
    """Save 600-dpi PNG + vector PDF. outdir e.g. .../grid/figures"""
    import os
    os.makedirs(outdir, exist_ok=True)
    png = os.path.join(outdir, out_stem + ".png")
    pdf = os.path.join(outdir, out_stem + ".pdf")
    fig.savefig(png, dpi=600, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(pdf, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return png, pdf
