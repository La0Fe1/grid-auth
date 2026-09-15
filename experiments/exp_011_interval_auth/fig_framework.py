"""F1：框架图（统一样式：表面/墨色/网格调色板，确定性绘制）。

用法：python -m experiments.exp_011_interval_auth.fig_framework
输出：paper/figures/fig_framework.pdf + paper/figures_png/fig_framework.png
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper"))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from figstyle import C_INK, C_MUTED, C_SURFACE, CAT, SEQ_BLUE, savefig

INK = C_INK


def box(ax, x, y, w, h, text, fc, ec, fs=7.5):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015",
                       fc=fc, ec=ec, lw=1.0)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=INK)


def arrow(ax, x1, y1, x2, y2, label=None, color="#52514e"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=11, lw=1.1, color=color)
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.06, label, ha="center",
                fontsize=6.8, color=C_MUTED)


def main():
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    fig.patch.set_facecolor(C_SURFACE)

    box(ax, 0.3, 5.6, 2.3, 1.0, "Proposers\n(LLM / heuristic)",
        "#f0f4fa", SEQ_BLUE[500])
    box(ax, 0.3, 2.2, 2.3, 1.0, "Grid state\n(telemetry)",
        "#f4f4f2", "#c3c2b7")
    box(ax, 3.9, 4.4, 2.7, 2.9,
        "Authorization layer\n\ninterval worst-case\nverification (LP)\n\nfail-closed",
        "#eef8f1", "#1baf7a")
    box(ax, 7.5, 4.4, 2.2, 2.9, "Grid\n(post-action\nDC power flow)",
        "#fdf2ee", CAT[2])
    box(ax, 0.3, 0.4, 2.3, 1.0, "Telemetry-based\nnetwork identification",
        "#f0f4fa", SEQ_BLUE[300])

    arrow(ax, 2.6, 6.1, 3.9, 6.2, "proposed action")
    arrow(ax, 2.6, 2.7, 3.9, 4.4, "state")
    arrow(ax, 6.6, 6.0, 7.5, 6.0, "release")
    arrow(ax, 6.6, 4.9, 7.5, 4.9, None, color="#d03b3b")
    ax.text(6.95, 5.15, "rejected", fontsize=6.8, color="#d03b3b", ha="center")
    ax.text(0.3, 7.1, "(a) Framework", fontsize=9, fontweight="bold", color=INK)

    ax.text(0.3, 2.05, "(b) Interval check on branch $\\ell$",
            fontsize=9, fontweight="bold", color=INK)
    ax.plot([2.2, 5.0, 7.8], [1.1, 1.1, 1.1], lw=9,
            color="#d8e6f7", solid_capstyle="butt", zorder=1)
    ax.plot([2.2, 7.8], [1.1, 1.1], lw=1.3, color=SEQ_BLUE[500], zorder=2)
    ax.axvline(6.6, ymin=0.08, ymax=0.66, color="#d03b3b", ls="--", lw=0.9)
    ax.text(6.6, 1.52, "rating $(1-m)\\bar f_\\ell$", color="#d03b3b",
            fontsize=6.8, ha="center")
    ax.text(2.2, 1.42, "$f_\\ell^-$", fontsize=8, ha="center", color=INK)
    ax.text(7.8, 1.42, "$f_\\ell^+$", fontsize=8, ha="center", color=INK)
    ax.text(4.6, 0.62, "exact interval from two linear programs",
            fontsize=6.8, ha="center", color=C_MUTED)

    fig.tight_layout()
    savefig(fig, "fig_framework.pdf")


if __name__ == "__main__":
    main()
