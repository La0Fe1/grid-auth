"""F1：框架图——提案源 → 授权层（区间验证）→ 执行 的确定性示意图。

用法：python -m experiments.exp_011_interval_auth.fig_framework
输出：paper/figures/fig_framework.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

C = {"box": "#eef2f7", "edge": "#2b5b84", "text": "#111111"}


def box(ax, x, y, w, h, text, fc=None, fs=9):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02",
                       fc=fc or C["box"], ec=C["edge"], lw=1.1)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=C["text"])


def arrow(ax, x1, y1, x2, y2, label=None):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                        mutation_scale=13, lw=1.2, color=C["edge"])
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.05, label, ha="center",
                fontsize=8, color="#444444")


def main():
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.6)
    ax.axis("off")

    box(ax, 0.3, 4.2, 2.4, 1.0, "Proposers\n(LLM / heuristic)")
    box(ax, 0.3, 1.6, 2.4, 1.0, "Grid state\n(telemetry)")
    box(ax, 3.9, 3.3, 2.6, 2.6,
        "Authorization layer\n\nInterval worst-case\nverification (LP)\n\nfail-closed",
        fc="#eafaf1")
    box(ax, 7.5, 3.3, 2.2, 2.6, "Grid\n(post-action\nDC power flow)", fc="#fdf3f3")
    box(ax, 0.3, 0.2, 2.4, 0.9, "Telemetry-based\nnetwork identification")

    arrow(ax, 2.7, 4.7, 3.9, 5.2, "proposed action")
    arrow(ax, 2.7, 2.1, 3.9, 3.4, "state")
    arrow(ax, 6.5, 4.6, 7.5, 4.6, "release")
    arrow(ax, 6.5, 3.6, 7.5, 3.6, "reject", )
    ax.text(6.9, 3.9, "rejected", fontsize=8, color="#d62728", ha="center")
    ax.text(0.3, 5.35, "(a) Framework", fontsize=10, fontweight="bold")

    # 右下小图：区间示意
    box(ax, 0.3, -1.1, 9.4, 0.0, "")
    ax.text(0.3, 1.35, "(b) Interval check on branch $\\ell$", fontsize=10,
            fontweight="bold")
    ax.plot([2.2, 5.0, 7.8], [0.55, 0.55, 0.55], lw=6, color="#c6d6e8",
            solid_capstyle="butt", zorder=1)
    ax.plot([2.2, 7.8], [0.55, 0.55], lw=1.5, color="#2b5b84", zorder=2)
    ax.axvline(6.6, ymin=0.1, ymax=0.62, color="#d62728", ls="--", lw=1)
    ax.text(6.6, 0.95, "rating $(1-m)\\bar f_\\ell$", color="#d62728",
            fontsize=8, ha="center")
    ax.text(2.2, 0.85, "$f_\\ell^-$", fontsize=9, ha="center")
    ax.text(7.8, 0.85, "$f_\\ell^+$", fontsize=9, ha="center")
    ax.text(4.5, 0.22, "exact interval from two LPs", fontsize=8, ha="center",
            color="#444444")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures",
                       "fig_framework.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
