"""F2：遥测识别 DC 模型保真度散点图（DC 流量 vs AC 有功潮流）。

用法：python -m experiments.exp_011_interval_auth.fig_telemetry
输出：paper/figures/fig_telemetry.pdf
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.dc_interval import dc_flows
from src.utils.env import Grid2OpGraphEnv


def main():
    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    from experiments.exp_011_interval_auth.spike_interval_auth import (
        estimate_lines_from_telemetry,
    )
    lines, n_sub = estimate_lines_from_telemetry(env, n_states=60)
    thermal = env.env_glop.get_thermal_limit()

    # 3 个固定状态
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

    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.scatter(p, f, s=18, alpha=0.75, edgecolors="none", label=None)
    lim = max(abs(p).max(), abs(f).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="identity")
    ax.set_xlabel("AC active branch flow $p_\\ell$ (MW)")
    ax.set_ylabel("Identified DC branch flow $f_\\ell$ (MW)")
    ax.annotate(f"Pearson $r={r:.4f}$", xy=(0.03, 0.94), xycoords="axes fraction")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures",
                       "fig_telemetry.pdf")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"保存: {out}（r={r:.4f}，n={len(f)} 条线路观测）")


if __name__ == "__main__":
    main()
