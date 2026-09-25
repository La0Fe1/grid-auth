"""exp_016 补测：遥测识别的量测噪声稳健性。

协议与 fig_telemetry 完全一致（60 状态估计、种子 7 的三状态 60 支路观测评估），
在估计前对遥测量（Δθ=θ_or−θ_ex 与 p_or）注入零均值高斯噪声
（标准差 = 信号绝对均值的 s 倍），重估电抗后计算识别 DC 潮流与 AC 潮流的
Pearson 相关。s ∈ {0, 0.01, 0.05, 0.10}。

用法：python -m experiments.exp_016.ident_noise
输出：experiments/exp_016/results/ident_noise.json
"""
import json
import os

import numpy as np

from src.utils.dc_interval import dc_flows
from src.utils.env import Grid2OpGraphEnv


def estimate_from_noisy(env, n_states=60, s=0.0):
    """同 estimate_lines_from_telemetry 的最小二乘协议，但先对遥测量注入噪声。"""
    from experiments.exp_011_interval_auth.spike_interval_auth import (
        estimate_lines_from_telemetry,
    )
    rng = np.random.RandomState(0)
    n_line = env.n_line
    n_sub = int(env.env_glop.n_sub)
    dtp, pp = [[] for _ in range(n_line)], [[] for _ in range(n_line)]
    n_chron = max(int(env._n_chronics), 1)
    for _ in range(n_states):
        env.env_glop.set_id(int(rng.randint(0, n_chron)))
        env.env_glop.reset()
        step = int(rng.randint(0, 800))
        for _attempt in range(6):
            try:
                env.env_glop.fast_forward_chronics(step)
                break
            except StopIteration:
                step //= 2
        obs = env.env_glop.get_obs()
        for l in range(n_line):
            if obs.line_status[l] == 0:
                continue
            d = float(obs.theta_or[l] - obs.theta_ex[l])
            p = float(obs.p_or[l])
            dtp[l].append(d + rng.normal(0.0, s * abs(d)) if abs(d) > 1e-12 else d)
            pp[l].append(p + rng.normal(0.0, s * abs(p)) if abs(p) > 1e-12 else p)
    lines = []
    for l in range(n_line):
        d, p = np.array(dtp[l]), np.array(pp[l])
        num, den = float((d * p).sum()), float((p * p).sum())
        x_ls = num / den if den > 1e-12 else 0.1
        lines.append({"or": int(obs.line_or_to_subid[l]),
                      "ex": int(obs.line_ex_to_subid[l]),
                      "x": max(x_ls, 1e-6)})
    return lines, n_sub


def fidelity(env, lines, n_sub):
    """同 fig_telemetry：种子 7 的三状态 60 支路观测上的 Pearson r。"""
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
    f, p = np.concatenate(all_f), np.concatenate(all_p)
    return float(np.corrcoef(f, p)[0, 1])


def main():
    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    levels = [0.0, 0.01, 0.05, 0.10]
    out = {"case": "rte_case14_realistic", "protocol": "same as fig_telemetry (60 snapshots LS, seed-7 3-state/60-branch fidelity)", "levels": {}}
    for s in levels:
        lines, n_sub = estimate_from_noisy(env, n_states=60, s=s)
        r = fidelity(env, lines, n_sub)
        out["levels"][str(s)] = {"r": r}
        print(f"noise s={s}: r={r:.4f}", flush=True)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "ident_noise.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"结果已保存: {path}")


if __name__ == "__main__":
    main()
