"""F5：LLM 提案研究——(a) 双提示词行为；(b) 授权前后失配危险率（顺序色阶按量级）。

用法：python -m experiments.exp_017.fig_llm
输出：paper/figures/fig_llm.pdf + paper/figures_png/fig_llm.png
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper"))
import matplotlib.pyplot as plt

from figstyle import SEQ_BLUE, savefig, styled

PROP = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed",
                    "llm_proposals_100.json")
PROP_SMALL = os.path.join(os.path.dirname(__file__), "..", "..", "data",
                          "processed", "llm_proposals.json")
EVAL = os.path.join(os.path.dirname(__file__), "results", "llm_auth_eval_100.json")


def main():
    with open(PROP, encoding="utf-8") as f:
        prop = json.load(f)
    n = len(prop["proposals"])
    n_active = sum(1 for p in prop["proposals"] if p["verifier_aware"])
    with open(PROP_SMALL, encoding="utf-8") as f:
        prop_small = json.load(f)
    n_small = len(prop_small["proposals"])
    n_frozen = sum(1 for p in prop_small["proposals"] if not p["conservative"])

    with open(EVAL) as f:
        ev = json.load(f)
    s = ev["summary"]
    rates = [s["none"]["false_pass_rate"], s["nominal"]["false_pass_rate"],
             s["mc"]["false_pass_rate"], s["interval"]["false_pass_rate"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.3))

    styled(ax1)
    vals = [n_small - n_frozen, n_active]
    ax1.bar(["Self-verify\nprompt", "Verifier-aware\nprompt"], vals,
            color=SEQ_BLUE[450], width=0.55)
    ax1.set_ylabel("States with\nproposed actions")
    ax1.set_ylim(0, n)
    for xi, v in enumerate(vals):
        ax1.text(xi, v + 1.5, f"{v}/{n_small if xi == 0 else n}",
                 ha="center", fontsize=7, color="#52514e")

    styled(ax2)
    labels = ["None", "Nominal", "Monte\nCarlo", "Interval"]
    steps = [600, 400, 200, 100]  # 量级→颜色深浅（顺序色阶）
    ax2.bar(labels, rates, color=[SEQ_BLUE[k] for k in steps], width=0.55)
    ax2.set_ylabel("Mismatch danger rate")
    ax2.set_ylim(0, 0.66)
    for xi, v in enumerate(rates):
        ax2.text(xi, v + 0.012, f"{v*100:.2f}%", ha="center",
                 fontsize=7, color="#52514e")

    fig.tight_layout()
    savefig(fig, "fig_llm.pdf")


if __name__ == "__main__":
    main()
