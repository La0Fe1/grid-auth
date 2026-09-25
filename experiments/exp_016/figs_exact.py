"""精确真值标签下的图 3/6/7 重画（沿用审稿人 _pubstyle 配色）。

数据源：
- Fig3: experiments/exp_016/results/exact_gt.json（主对比 12,500 精确标签）
- Fig6/Fig7: experiments/exp_016/results/abl_rob_exact.json（消融/鲁棒性精确重跑）

用法：python -m experiments.exp_016.figs_exact
输出：paper/figures/{fig_main,fig_ablation,fig_eps}.pdf + figures_png PNG
"""
import json
import os
import shutil
import sys

# _pubstyle 随仓库分发（原先指向本机桌面路径，换机器即失效）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import matplotlib.pyplot as plt
import numpy as np

from _pubstyle import BLUE, GRAY, GREEN, VERM, despine, save, ygrid

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
FIGDIR = os.path.join(ROOT, "paper", "figures")
PNGDIR = os.path.join(ROOT, "paper", "figures_png")


def main():
    # ---------- Fig3: 主对比（精确标签） ----------
    d = json.load(open(os.path.join(HERE, "results", "exact_gt.json"), encoding="utf-8"))
    m = d["14bus_12500"]
    labels = ["Nominal", "Monte\nCarlo", "Interval"]
    fp = [m["nominal"]["false_pass_exact"], m["mc"]["false_pass_exact"],
          m["interval"]["false_pass_exact"]]
    rel = [0.4932, 0.4158, 0.3996]

    fig, ax = plt.subplots(figsize=(5.833, 3.0))
    despine(ax)
    ygrid(ax)
    x = np.arange(3)
    ax.bar(x, [v * 100 for v in fp], width=0.5, color=BLUE,
           label="False-pass rate (exact labels)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("False-pass rate (%)")
    ax.set_ylim(0, 22)
    for xi, v in zip(x, fp):
        ax.text(xi, v * 100 + 0.5, f"{v*100:.2f}%", ha="center", fontsize=8,
                color="#333333")
    ax.text(0.985, 0.95,
            "false-reject rate = 0 for all three\n(exact box ground truth)\n"
            "released: 49.3% / 41.6% / 40.0%",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
            color="#52514e")
    fig.tight_layout()
    save(fig, "fig_main", FIGDIR)

    # ---------- Fig7: eps 扫描（精确标签） ----------
    abl = json.load(open(os.path.join(HERE, "results", "abl_rob_exact.json"),
                         encoding="utf-8"))["configs"]
    eps = [0.0, 0.05, 0.10, 0.20, 0.40]
    fp = [abl[f"rob-eps={e:.2f}"]["interval"]["false_pass_exact"] for e in eps]
    fr = [abl[f"rob-eps={e:.2f}"]["interval"]["false_reject_exact"] for e in eps]
    nom = abl["rob-eps=0.00"]["nominal"]["false_pass_exact"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.833, 5.0), sharex=True)
    despine(ax1)
    ygrid(ax1)
    ax1.plot(eps, [v * 100 for v in fp], "o-", color=BLUE, lw=1.3, ms=6)
    ax1.axhline(nom * 100, color=VERM, ls=":", lw=1.2,
                label=f"nominal verifier ({nom*100:.1f}%)")
    ax1.axvline(0.2, color=GRAY, ls="--", lw=0.9)
    ax1.annotate(r"true mismatch $\varepsilon=0.2$", xy=(0.2, 0.5),
                 xytext=(0.05, 12), fontsize=7.5,
                 arrowprops=dict(arrowstyle="->", lw=0.7, color=GRAY),
                 color="#52514e")
    ax1.set_ylabel("False-pass rate (%)")
    ax1.set_ylim(-2, 36)
    ax1.legend(frameon=False, loc="center left")
    despine(ax2)
    ygrid(ax2)
    ax2.plot(eps, [v * 100 for v in fr], "s--", color=GREEN, lw=1.3, ms=6)
    ax2.set_xlabel(r"Assumed uncertainty width $\varepsilon$")
    ax2.set_ylabel("False-reject rate (%)")
    ax2.set_ylim(-1.5, 12)
    fig.tight_layout()
    save(fig, "fig_eps", FIGDIR)

    # ---------- Fig6: 消融（释放比例 + 精确 FP/FR 标注） ----------
    order = ["baseline", "telemetry-30", "true-model", "+connectivity", "prob95", "prob99",
             "random-proposals", "llm-proposals", "uniform-x", "margin=0.02",
             "margin=0", "eps=0.10", "eps=0.05", "+connectivity+theta"]
    names = {"baseline": "Baseline", "telemetry-30": "telemetry-30",
             "true-model": "true-model", "+connectivity": "+connectivity",
             "prob95": "prob95",
             "prob99": "prob99", "random-proposals": "random-proposals",
             "llm-proposals": "llm-proposals", "uniform-x": "uniform-x",
             "margin=0.02": "margin=0.02", "margin=0": "margin=0",
             "eps=0.10": "eps=0.10", "eps=0.05": "eps=0.05",
             "+connectivity+theta": "+connectivity+theta"}
    rel = [abl[k]["interval"]["released_frac"] * 100 for k in order]
    fpv = {k: abl[k]["interval"]["false_pass_exact"] * 100 for k in order}
    frv = {k: abl[k]["interval"]["false_reject_exact"] * 100 for k in order}

    fig, ax = plt.subplots(figsize=(5.833, 4.4))
    despine(ax)
    yrev = list(range(len(order)))[::-1]
    ax.barh(yrev, rel[::-1], color=BLUE, height=0.6)
    ax.set_yticks(yrev)
    ax.set_yticklabels([names[k] for k in order][::-1], fontsize=7.5)
    ax.set_xlabel("Fraction of cases released (%)")
    ax.set_xlim(0, 55)
    for yi, k in zip(yrev, order[::-1]):
        v = rel[order.index(k)]
        note = []
        if fpv[k] > 0:
            note.append(f"FP {fpv[k]:.1f}%")
        if frv[k] > 0:
            note.append(f"FR {frv[k]:.1f}%")
        txt = ", ".join(note) if note else "FP 0, FR 0"
        ax.text(v + 0.8, yi, txt, va="center", fontsize=7, color="#52514e")
    ax.text(0.985, 0.02,
            "exact box ground truth; interval FP/FR\nannotated per configuration",
            transform=ax.transAxes, ha="right", fontsize=7, color="#52514e")
    fig.tight_layout()
    save(fig, "fig_ablation", FIGDIR)

    for name in ("fig_main", "fig_eps", "fig_ablation"):
        shutil.copy(os.path.join(FIGDIR, name + ".png"),
                    os.path.join(PNGDIR, name + ".png"))
    print("figures regenerated with exact labels")


if __name__ == "__main__":
    main()
