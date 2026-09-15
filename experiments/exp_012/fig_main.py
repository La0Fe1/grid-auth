"""F3：主对比分组柱状图 + F4：真值 MC 收敛曲线（统一样式）。

用法：python -m experiments.exp_012.fig_main
输出：paper/figures/fig_main.pdf、fig_convergence.pdf（+figures_png 对应 PNG）
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper"))
import matplotlib.pyplot as plt

from figstyle import CAT, savefig, styled

RESULT = os.path.join(os.path.dirname(__file__), "results", "auth_eval.json")


def main():
    with open(RESULT) as f:
        d = json.load(f)
    s = d["summary"]
    names = ["None", "Nominal", "Monte\nCarlo", "Interval"]
    fp = [s["none"]["false_pass_rate"], s["nominal"]["false_pass_rate"],
          s["mc"]["false_pass_rate"], s["interval"]["false_pass_rate"]]
    fr = [0.0, s["nominal"]["false_reject_rate"],
          s["mc"]["false_reject_rate"], s["interval"]["false_reject_rate"]]

    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    styled(ax)
    x = [0, 1, 2, 3]
    w = 0.36
    b1 = ax.bar([xi - w / 2 for xi in x], fp, w, color=CAT[1],
                label="False-pass rate")
    b2 = ax.bar([xi + w / 2 for xi in x], fr, w, color=CAT[2],
                label="False-reject rate")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 0.22)
    for b in list(b1) + list(b2):
        v = b.get_height()
        if v >= 0.002:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.006,
                    f"{v*100:.1f}%" if v >= 0.01 else "0%",
                    ha="center", fontsize=6.5, color="#52514e")
        elif v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.006, "0%",
                    ha="center", fontsize=6.5, color="#52514e")
    ax.legend(loc="upper right")
    fig.tight_layout()
    savefig(fig, "fig_main.pdf")

    # F4：收敛曲线
    conv = d.get("conv_curves", {})
    ms = sorted(int(k) for k in conv if k.isdigit())
    if len(ms) >= 2:
        fig2, ax2 = plt.subplots(figsize=(3.4, 2.4))
        styled(ax2)
        series = [("nominal", "Nominal", CAT[1]),
                  ("mc", "Monte Carlo", CAT[2]),
                  ("interval", "Interval", CAT[3])]
        for key, label, color in series:
            ax2.plot(ms, [conv[str(m)][key] for m in ms], "o-",
                     color=color, lw=1.5, ms=4.5, label=label)
        ax2.set_xlabel("Ground-truth sample size $M$")
        ax2.set_ylabel("Danger rate\namong released actions")
        ax2.set_xscale("log")
        ax2.legend(loc="upper right")
        fig2.tight_layout()
        savefig(fig2, "fig_convergence.pdf")


if __name__ == "__main__":
    main()
