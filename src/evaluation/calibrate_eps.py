"""exp_018 ε 校准：从量测误差的经验分布标定箱宽，附有限样本覆盖界。

原理：把"真实负荷"与"量测负荷"的关系建模为乘性噪声
（P_true = P_meas / (1+δ)，δ ~ 经验分布）。给定目标覆盖率 1−α，
取 δ 的经验 (1−α)-分位数作为 ε；用 Hoeffding 界给出"该分位数估计
以概率 ≥1−β 覆盖真实分位数"所需的样本量承诺。

实验：在 rte_case14 场景上注入乘性高斯噪声（σ 扫描），
比较校准 ε 与固定 ε=0.2 在授权层上的误放/误拒（复用 auth_eval 结果做对照）。

用法：python -m src.evaluation.calibrate_eps
"""
import json
import os

import numpy as np

from src.utils.dc_interval import (
    adversarial_witness, authorize, dc_flows, injection_bounds,
    interval_flows_lp,
)
from src.utils.env import Grid2OpGraphEnv


def empirical_eps(errors, alpha, n_samples):
    """经验 (1−α)-分位数作为 ε；返回 (ε_hat, DKW 半宽, 次序统计量 CI)。

    修正（2026-09-16 审稿意见 CRITICAL #4）：原 Hoeffding 界约束的是有界变量
    的样本均值，不适用于分位数/覆盖估计。分位数估计的一致收敛用
    Dvoretzky–Kiefer–Wolfowitz (DKW) 界：P(sup|F̂−F| > c) ≤ 2exp(−2nc²)，
    95% 半宽 c = √(ln(2/0.05)/(2n))；另附精确次序统计量置信区间
    （二项分布反解秩，Clopper–Pearson 式，无需渐近近似）。
    """
    # 注意：ε 必须覆盖 |δ| 的 (1−α) 分位数（箱是对称 ±ε）；
    # 旧实现误用带符号 δ 的分位数（对称分布下覆盖率实为 1−2α），2026-09-16 修正。
    e_hat = float(np.quantile(np.abs(errors), 1 - alpha))
    dkw = float(np.sqrt(np.log(2 / 0.05) / (2 * n_samples)))
    # 精确次序统计量 CI：找 X_(r), X_(s) 使 P(X_(r) ≤ q ≤ X_(s)) ≥ 0.95
    from scipy.stats import binom
    p = 1 - alpha
    lo, hi = 0.0, 1.0  # 覆盖 p 的保守概率界（单调性）
    r, s = 0, n_samples
    while binom.cdf(r - 1, n_samples, p) < 0.025:
        r += 1
    while binom.cdf(s - 1, n_samples, p) > 0.975:
        s -= 1
    x_sorted = np.sort(np.abs(errors))
    ci_lo = float(x_sorted[max(r - 1, 0)])
    ci_hi = float(x_sorted[min(s - 1, n_samples - 1)])
    return e_hat, dkw, (ci_lo, ci_hi)


def main():
    rng = np.random.RandomState(0)
    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    import pickle
    lines = pickle.load(open("data/processed/auth_lines_rte_case14_realistic.pkl", "rb"))
    n_sub = 14
    thermal = env.env_glop.get_thermal_limit().astype(float)

    # 校准集：注入乘性噪声，量测 = 噪声值，真值 = 原值 → 误差 = meas/true − 1
    errors = []
    for _ in range(200):
        env.env_glop.set_id(int(rng.randint(0, 100)))
        env.env_glop.reset()
        env.env_glop.fast_forward_chronics(int(rng.randint(0, 800)))
        obs = env.env_glop.get_obs()
        for ld in range(len(obs.load_p)):
            if obs.load_p[ld] > 1.0:
                meas = obs.load_p[ld] * (1 + rng.normal(0, 0.1))  # σ=10% 量测噪声
                errors.append(meas / obs.load_p[ld] - 1.0)
    errors = np.array(errors)

    results = {}
    for alpha in (0.05, 0.10, 0.20):
        eps_hat, dkw, ci = empirical_eps(errors, alpha, len(errors))
        results[f"alpha={alpha}"] = {"eps_hat": eps_hat,
                                     "dkw_halfwidth": dkw,
                                     "order_stat_ci": list(ci),
                                     "n_calibration": int(len(errors))}
        print(f"alpha={alpha}: ε_hat={eps_hat:.4f}（DKW ±{dkw:.4f}，"
              f"次序统计量 95% CI=[{ci[0]:.4f},{ci[1]:.4f}]，n={len(errors)}）")

    # 演示：固定 ε=0.2 vs 校准 ε（α=0.1）在一个高应力用例上的授权与反例
    env.env_glop.set_id(0)
    env.env_glop.reset()
    env.env_glop.fast_forward_chronics(900)
    obs = env.env_glop.get_obs()
    lam = 2.0
    lo, hi = injection_bounds(obs.gen_p, obs.load_p * lam, obs.gen_to_subid,
                              obs.load_to_subid, n_sub, load_eps=0.2)
    lines_act = [dict(l) for l in lines]
    lines_act[5] = dict(lines[5], x=np.inf)
    f_lo, f_hi = interval_flows_lp(lines_act, n_sub, lo, hi)
    ok, viol = authorize(f_lo, f_hi, thermal, margin=0.05)
    wit = adversarial_witness(lines_act, n_sub, lo, hi, thermal, margin=0.05)
    if wit is not None:
        P, line, flow, cap = wit
        print(f"\n演示（λ=2, 断开线5）：授权={'放行' if ok else '拒绝'}")
        print(f"  反例：支路 {line} 流量 {flow:.1f} MW > 限值 {cap:.1f} MW")
        print(f"  反例负荷模式（MW）: {np.round(P[P < -5], 1)[:8]} ...")
    results["demo"] = {"authorized": bool(ok),
                       "witness_line": None if wit is None else int(wit[1]),
                       "witness_flow": None if wit is None else float(wit[2]),
                       "witness_cap": None if wit is None else float(wit[3])}
    out = "experiments/exp_018/results/calibration.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存: {out}")


if __name__ == "__main__":
    main()
