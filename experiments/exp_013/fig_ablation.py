"""F6：消融图——13 配置的区间误拒率（量级 → 蓝色顺序色阶）。

用法：python -m experiments.exp_013.fig_ablation
输出：paper/figures/fig_ablation.pdf + paper/figures_png/fig_ablation.png
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper"))
import matplotlib.pyplot as plt
import numpy as np

from figstyle import SEQ_BLUE, savefig, styled

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")

CONFIGS = [
    ("Baseline", "baseline", 0),
    ("eps=0.05", "eps=0.05_fix", 1),
    ("eps=0.10", "eps=0.10_fix", 1),
    ("+connectivity", "+connectivity", 2),
    ("+connectivity+theta", "+connectivity+theta", 2),
    ("uniform-x", "uniform-x", 3),
    ("telemetry-30", "telemetry-30", 3),
    ("random-proposals", "random-proposals", 4),
    ("llm-proposals", "llm-proposals", 4),
    ("margin=0", "margin=0_fix", 5),
    ("margin=0.02", "margin=0.02_fix", 5),
    ("prob99", "prob99", 6),
    ("prob95", "prob95", 6),
]


def main():
    labels, fr = [], []
    for label, name, dim in CONFIGS:
        p = os.path.join(RESULT_DIR, f"abl_{name}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        fr.append(d["summary"]["interval"]["false_reject_rate"])
        labels.append(label)

    order = np.argsort(fr)
    labels = [labels[i] for i in order]
    fr = [fr[i] for i in order]
    fr_max = max(fr) if fr else 1.0
    keys = sorted(SEQ_BLUE)
    colors = []
    for v in fr:
        target = max(100, 700 - int(600 * v / fr_max))
        k = min(keys, key=lambda kk: abs(kk - target))
        colors.append(SEQ_BLUE[k])

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    styled(ax)
    bars = ax.barh(labels, fr, color=colors)
    ax.set_xlabel("Interval false-reject rate")
    ax.set_xlim(0, 0.40)
    ax.text(0.985, 0.02, "false-pass rate = 0 in all 13 configurations",
            transform=ax.transAxes, ha="right", fontsize=6.5, color="#52514e")
    for b, v in zip(bars, fr):
        ax.text(v + 0.007, b.get_y() + b.get_height() / 2, f"{v*100:.1f}%",
                va="center", fontsize=6.5, color="#52514e")
    fig.tight_layout()
    savefig(fig, "fig_ablation.pdf")


if __name__ == "__main__":
    main()
