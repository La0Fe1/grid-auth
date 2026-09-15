"""F3：主对比柱状图（误放率/误拒率 × 四验证器）+ F4：真值 MC 收敛曲线。

用法：python -m experiments.exp_012.fig_main
输出：paper/figures/fig_main.pdf、paper/figures/fig_convergence.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULT = os.path.join(os.path.dirname(__file__), "results", "auth_eval.json")


def main():
    with open(RESULT) as f:
        d = json.load(f)
    s = d["summary"]
    names = ["None", "Nominal", "Monte Carlo", "Interval"]
    fp = [s["none"]["false_pass_rate"], s["nominal"]["false_pass_rate"],
          s["mc"]["false_pass_rate"], s["interval"]["false_pass_rate"]]
    fr = [0.0, s["nominal"]["false_reject_rate"],
          s["mc"]["false_reject_rate"], s["interval"]["false_reject_rate"]]

    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    x = np.arange(4)
    w = 0.38
    ax.bar(x - w / 2, fp, w, color="#d62728", label="False-pass rate")
    ax.bar(x + w / 2, fr, w, color="#2ca02c", label="False-reject rate")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 0.25)
    for xi, v in zip(x - w / 2, fp):
        ax.text(xi, v + 0.006, f"{v*100:.1f}%" if v > 0.001 else "0%",
                ha="center", fontsize=8)
    for xi, v in zip(x + w / 2, fr):
        ax.text(xi, v + 0.006, f"{v*100:.2f}%", ha="center", fontsize=8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures",
                       "fig_main.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"保存: {out}")

    # F4：收敛曲线（仅从完整覆盖的用例重算——主实验的 conv_curves 部分为恢复模式）
    conv = d.get("conv_curves", {})
    ms = sorted(int(k) for k in conv if k.isdigit())
    if len(ms) >= 2:
        fig2, ax2 = plt.subplots(figsize=(4.4, 2.8))
        for key, label, color in (("nominal", "Nominal", "#d62728"),
                                  ("mc", "Monte Carlo", "#9467bd"),
                                  ("interval", "Interval", "#1f77b4")):
            ax2.plot(ms, [conv[str(m)][key] for m in ms], "o-",
                     color=color, label=label)
        ax2.set_xlabel("Ground-truth sample size $M$")
        ax2.set_ylabel("Danger rate among released actions")
        ax2.set_xscale("log")
        ax2.grid(alpha=0.3)
        ax2.legend(fontsize=8)
        fig2.tight_layout()
        out2 = os.path.join(os.path.dirname(__file__), "..", "..", "paper",
                            "figures", "fig_convergence.pdf")
        fig2.savefig(out2, bbox_inches="tight")
        print(f"保存: {out2}")


if __name__ == "__main__":
    main()
