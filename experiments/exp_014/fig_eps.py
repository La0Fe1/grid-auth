"""F7（主图）：区间验证误放/误拒率随不确定性宽度 ε 的曲线（真值失配固定 0.2）。

用法：python -m experiments.exp_014.fig_eps
输出：paper/figures/fig_eps.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = os.path.join(os.path.dirname(__file__), "results")


def main():
    eps = [0.0, 0.05, 0.1, 0.2, 0.4]
    fp, fr, fp_nom = [], [], []
    for e in eps:
        name = "eps=0.00" if e == 0 else f"eps={e:.2f}"
        with open(os.path.join(RESULT_DIR, f"rob_{name}.json")) as fh:
            d = json.load(fh)
        s = d["summary"]
        fp.append(s["interval"]["false_pass_rate"])
        fr.append(s["interval"]["false_reject_rate"])
        fp_nom.append(s["nominal"]["false_pass_rate"])

    fig, ax1 = plt.subplots(figsize=(4.6, 3.2))
    ax1.plot(eps, fp, "o-", color="#1f77b4", label="interval false-pass rate")
    ax1.axhline(fp_nom[0], color="#d62728", ls=":", lw=1,
                label="nominal false-pass rate (constant)")
    ax1.axvline(0.2, color="gray", ls="--", lw=1, alpha=0.7)
    ax1.annotate("true mismatch $\\varepsilon_{\\text{ref}}=0.2$",
                 xy=(0.2, 0.01), xytext=(0.06, 0.05),
                 arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=8)
    ax1.set_xlabel("Assumed uncertainty width $\\varepsilon$")
    ax1.set_ylabel("False-pass rate")
    ax1.set_ylim(-0.02, 0.30)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(eps, fr, "s--", color="#2ca02c", label="interval false-reject rate")
    ax2.set_ylabel("False-reject rate")
    ax2.set_ylim(-0.02, 0.20)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center left")
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures",
                       "fig_eps.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"保存: {out}")
    print("  ε → 误放率:", dict(zip(eps, np.round(fp, 4))))
    print("  ε → 误拒率:", dict(zip(eps, np.round(fr, 4))))


if __name__ == "__main__":
    main()
