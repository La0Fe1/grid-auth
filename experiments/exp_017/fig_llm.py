"""F5：LLM 提案研究——(a) 双提示词行为对比；(b) 授权前后失配危险率。

用法：python -m experiments.exp_017.fig_llm
输出：paper/figures/fig_llm.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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
    none_fp = s["none"]["false_pass_rate"]
    nom_fp = s["nominal"]["false_pass_rate"]
    mc_fp = s["mc"]["false_pass_rate"]
    int_fp = s["interval"]["false_pass_rate"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    ax1.bar(["Self-verify\nprompt", "Verifier-aware\nprompt"],
            [n_small - n_frozen, n_active], color=["#d62728", "#2ca02c"],
            width=0.55)
    ax1.set_ylabel("States with proposed actions")
    ax1.set_ylim(0, n)
    ax1.grid(axis="y", alpha=0.3)
    ax1.text(0, (n_small - n_frozen) + 1.5, f"{n_small - n_frozen}/{n_small}",
             ha="center", fontsize=8)
    ax1.text(1, n_active + 1.5, f"{n_active}/{n}", ha="center", fontsize=8)

    bars = [none_fp, nom_fp, mc_fp, int_fp]
    ax2.bar(["None", "Nominal", "Monte\nCarlo", "Interval"], bars,
            color=["#7f7f7f", "#d62728", "#9467bd", "#2ca02c"], width=0.55)
    ax2.set_ylabel("Mismatch danger rate")
    ax2.set_ylim(0, 0.65)
    ax2.grid(axis="y", alpha=0.3)
    for xi, v in enumerate(bars):
        ax2.text(xi, v + 0.012, f"{v*100:.2f}%", ha="center", fontsize=8)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures",
                       "fig_llm.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"保存: {out}")


if __name__ == "__main__":
    main()
