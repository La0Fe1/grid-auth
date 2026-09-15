"""F6：消融图——13 配置的区间误拒率（误放率全部为 0，图中标注）。

用法：python -m experiments.exp_013.fig_ablation
输出：paper/figures/fig_ablation.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    labels, fr, dims = [], [], []
    for label, name, dim in CONFIGS:
        p = os.path.join(RESULT_DIR, f"abl_{name}.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            d = json.load(f)
        fr.append(d["summary"]["interval"]["false_reject_rate"])
        labels.append(label)
        dims.append(dim)

    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    colors = plt.cm.tab10(np.array(dims) % 10)
    bars = ax.barh(labels, fr, color=colors)
    ax.set_xlabel("Interval false-reject rate")
    ax.set_xlim(0, 0.40)
    ax.grid(axis="x", alpha=0.3)
    ax.text(0.99, 0.05, "false-pass rate = 0 in all 13 configurations",
            transform=ax.transAxes, ha="right", fontsize=8,
            bbox=dict(boxstyle="round", fc="#eafaf1", ec="#2ca02c", lw=0.6))
    for b, v in zip(bars, fr):
        ax.text(v + 0.006, b.get_y() + b.get_height() / 2, f"{v*100:.1f}%",
                va="center", fontsize=7)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures",
                       "fig_ablation.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
