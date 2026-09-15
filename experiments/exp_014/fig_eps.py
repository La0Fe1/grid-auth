"""F7（主图）：区间验证误放/误拒率随 ε 的曲线——双面板（禁止双轴）。

用法：python -m experiments.exp_014.fig_eps
输出：paper/figures/fig_eps.pdf + paper/figures_png/fig_eps.png
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper"))
import matplotlib.pyplot as plt

from figstyle import CAT, SEQ_BLUE, savefig, styled

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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.4, 3.6), sharex=True)
    styled(ax1)
    ax1.plot(eps, fp, "o-", color=SEQ_BLUE[500], lw=1.5, ms=5)
    ax1.axhline(fp_nom[0], color=CAT[2], ls=":", lw=1.1,
                label="nominal verifier (constant)")
    ax1.axvline(0.2, color="#c3c2b7", ls="--", lw=0.9)
    ax1.annotate("true mismatch\n$\\varepsilon=0.2$", xy=(0.2, 0.02),
                 xytext=(0.03, 0.13), fontsize=7,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color="#898781"),
                 color="#52514e")
    ax1.set_ylabel("False-pass rate")
    ax1.set_ylim(-0.02, 0.30)
    ax1.legend(loc="center left")

    styled(ax2)
    ax2.plot(eps, fr, "s--", color=SEQ_BLUE[400], lw=1.5, ms=5)
    ax2.set_xlabel("Assumed uncertainty width $\\varepsilon$")
    ax2.set_ylabel("False-reject rate")
    ax2.set_ylim(-0.02, 0.20)
    ax2.set_xticks(eps)

    fig.align_ylabels()
    fig.tight_layout()
    savefig(fig, "fig_eps.pdf")


if __name__ == "__main__":
    main()
