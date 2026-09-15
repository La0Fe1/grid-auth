"""F2：遥测识别 DC 模型保真度散点图（单系列：无图例，标题即身份）。

用法：python -m experiments.exp_011_interval_auth.fig_telemetry
输出：paper/figures/fig_telemetry.pdf + paper/figures_png/fig_telemetry.png
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "paper"))
import matplotlib.pyplot as plt
import numpy as np

from figstyle import SEQ_BLUE, savefig, styled

from src.utils.dc_interval import dc_flows
from src.utils.env import Grid2OpGraphEnv


def main():
    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    from experiments.exp_011_interval_auth.spike_interval_auth import (
        estimate_lines_from_telemetry,
    )
    lines, n_sub = estimate_lines_from_telemetry(env, n_states=60)

    rng = np.random.RandomState(7)
    all_f, all_p = [], []
    for _ in range(3):
        env.env_glop.set_id(int(rng.randint(0, 100)))
        env.env_glop.reset()
        env.env_glop.fast_forward_chronics(int(rng.randint(100, 900)))
        obs = env.env_glop.get_obs()
        P = np.zeros(n_sub)
        for g, s in enumerate(obs.gen_to_subid):
            P[s] += obs.gen_p[g]
        for ld, s in enumerate(obs.load_to_subid):
            P[s] -= obs.load_p[ld]
        all_f.append(dc_flows(lines, n_sub, P))
        all_p.append(obs.p_or)
    f = np.concatenate(all_f)
    p = np.concatenate(all_p)
    r = np.corrcoef(f, p)[0, 1]

    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    styled(ax)
    ax.scatter(p, f, s=14, color=SEQ_BLUE[450], alpha=0.65,
               edgecolors="none", zorder=3)
    lim = max(abs(p).max(), abs(f).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], ls="--", lw=1.0, color="#898781",
            label="identity")
    ax.set_xlabel("AC active branch flow $p_\\ell$ (MW)")
    ax.set_ylabel("Identified DC branch flow $f_\\ell$ (MW)")
    ax.annotate(f"Pearson $r={r:.4f}$\n$n={len(f)}$ branch observations",
                xy=(0.04, 0.97), xycoords="axes fraction", va="top",
                fontsize=7, color="#52514e")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()
    savefig(fig, "fig_telemetry.pdf")


if __name__ == "__main__":
    main()
