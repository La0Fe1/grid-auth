"""方向 B 重定义可行性尖刺（exp_011）：真实 14-bus 上的区间安全授权。

演示：对若干真实观测状态 + 候选 toggle 动作，
比较"标称验证"（单一模型）与"区间验证"（±20% 负荷箱，LP 精确最坏情况）的授权结果；
统计：区间验证比标称验证多拒绝的动作数（= 标称验证在失配下会误放的动作，即论文核心量）。

冒烟判据：①区间验证比标称验证严格（多拒绝 ≥1 个动作且无漏放）；
②区间验证的运行时间可接受（每个动作 ≤0.5s）。

用法：python -m experiments.exp_011_interval_auth.spike_interval_auth
"""
import json
import os
import time

import numpy as np

from src.utils.dc_interval import (
    authorize, dc_flows, injection_bounds, interval_flows_lp,
)
from src.utils.env import Grid2OpGraphEnv

MARGIN = 0.05
LOAD_EPS = 0.2


def estimate_lines_from_telemetry(env, n_states=60):
    """从遥测识别 DC 支路模型：x_ℓ = (θ_or − θ_ex)/p_or 的多状态中位数。

    grid2op 观测含电压相角（theta_or/theta_ex）与有功潮流（p_or）——
    授权层无需离线电网文件即可自识别网络模型（与不确定性主题一致）。
    返回 (lines, n_sub)。
    """
    rng = np.random.RandomState(0)
    n_line = env.n_line
    n_sub = int(env.env_glop.n_sub)
    dtp, pp = [[] for _ in range(n_line)], [[] for _ in range(n_line)]
    n_chron = max(int(env._n_chronics), 1)
    for _ in range(n_states):
        env.env_glop.set_id(int(rng.randint(0, n_chron)))
        env.env_glop.reset()
        step = int(rng.randint(0, 800))
        for _attempt in range(6):  # dev 模式 chronics 可能短于 800 步
            try:
                env.env_glop.fast_forward_chronics(step)
                break
            except StopIteration:
                step //= 2
        obs = env.env_glop.get_obs()
        for l in range(n_line):
            if obs.line_status[l] == 0:
                continue
            dtp[l].append(obs.theta_or[l] - obs.theta_ex[l])
            pp[l].append(obs.p_or[l])
    lines = []
    for l in range(n_line):
        d, p = np.array(dtp[l]), np.array(pp[l])
        # 最小二乘：x = Σ(Δθ·p)/Σ(p²)（抗中位数偏置）
        num, den = float((d * p).sum()), float((p * p).sum())
        x_ls = num / den if den > 1e-12 else 0.1
        lines.append({"or": int(obs.line_or_to_subid[l]),
                      "ex": int(obs.line_ex_to_subid[l]),
                      "x": max(x_ls, 1e-6)})
    return lines, n_sub


def main():
    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    print("从遥测识别支路电抗（60 状态中位数）……")
    lines, n_sub = estimate_lines_from_telemetry(env)
    n_line = len(lines)
    thermal = env.env_glop.get_thermal_limit().astype(float)
    print(f"n_sub={n_sub}, n_line={n_line}, thermal 范围 "
          f"[{thermal.min():.0f}, {thermal.max():.0f}] MW")

    rng = np.random.RandomState(0)
    n_states, n_candidates = 20, 5
    nominal_reject, interval_extra_reject = 0, 0
    t_interval_total = 0.0
    samples = []
    dc_vs_ac = []
    # 说明：grid2op 的 rho 是安培基准；DC 授权用 MW 热限。rte_case14 在 MW 意义上
    # 天然轻载（观测最大 ~0.22×限值），故用负荷增长系数 λ 扫描制造 MW 应力
    # （安全验证演示的标准做法），在应力区比较标称 vs 区间授权。

    for st in range(n_states):
        env.env_glop.set_id(int(rng.randint(0, 100)))
        env.env_glop.reset()
        env.env_glop.fast_forward_chronics(int(rng.randint(0, 800)))
        obs = env.env_glop.get_obs()

        gen_to_sub = obs.gen_to_subid
        load_to_sub = obs.load_to_subid
        load_by_sub = np.zeros(n_sub)
        for ld, s in enumerate(load_to_sub):
            load_by_sub[s] += obs.load_p[ld]
        P_base = np.zeros(n_sub)
        for g, s in enumerate(gen_to_sub):
            P_base[s] += obs.gen_p[g]
        P_base -= load_by_sub

        # DC 模型保真度（动作前、λ=1）
        f_base = dc_flows(lines, n_sub, P_base)
        dc_vs_ac.append(np.corrcoef(f_base, obs.p_or)[0, 1])
        if st < 3:
            slope = np.polyfit(f_base, obs.p_or, 1)[0]
            print(f"  [诊断 state{st}] DC/AC 斜率={slope:.3f}")

        # 候选动作 = 断开 rho 最高的若干条连通线路
        cand = np.argsort(-obs.rho)[:n_candidates]
        for lid in cand:
            if obs.line_status[lid] == 0:
                continue
            l0 = lines[lid]
            lines_act = [dict(l) for l in lines]
            lines_act[lid] = dict(l0, x=np.inf)  # 断开 = 电抗无穷大
            for lam in (2.0, 3.0, 4.0, 5.0):  # 负荷增长扫描制造 MW 应力
                P_lam = P_base - (lam - 1.0) * load_by_sub
                lo, hi = injection_bounds(
                    obs.gen_p, obs.load_p * lam, gen_to_sub, load_to_sub,
                    n_sub, load_eps=LOAD_EPS)
                # 标称验证：λ 增长后的注入点
                f_nom = dc_flows(lines_act, n_sub, P_lam)
                nom_ok, _ = authorize(f_nom, f_nom, thermal, margin=MARGIN)
                # 区间验证：负荷箱最坏情况
                t0 = time.time()
                f_lo, f_hi = interval_flows_lp(lines_act, n_sub, lo, hi)
                t_interval_total += time.time() - t0
                int_ok, _ = authorize(f_lo, f_hi, thermal, margin=MARGIN)

                if not nom_ok:
                    nominal_reject += 1
                elif not int_ok:
                    interval_extra_reject += 1
                samples.append({"state": st, "line": int(lid), "lambda": lam,
                                "nominal_ok": bool(nom_ok),
                                "interval_ok": bool(int_ok),
                                "worst_vs_cap": float(np.max(
                                    np.maximum(np.abs(f_lo), np.abs(f_hi))
                                    / (thermal * (1 - MARGIN))))})

    result = {
        "n_checked": len(samples),
        "nominal_reject": nominal_reject,
        "interval_extra_reject": interval_extra_reject,
        "extra_reject_ratio": float(interval_extra_reject / max(len(samples), 1)),
        "avg_interval_seconds": round(t_interval_total / max(len(samples), 1), 4),
        "dc_ac_corr_median": float(np.median(dc_vs_ac)) if dc_vs_ac else float("nan"),
    }
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "spike_interval_auth.json")
    with open(out_path, "w") as f:
        json.dump({"args": {"margin": MARGIN, "load_eps": LOAD_EPS},
                   "result": result, "samples": samples}, f, indent=2)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"判据: 区间验证多拒绝动作数={interval_extra_reject} (>0 且无漏放) | "
          f"平均每动作耗时 {result['avg_interval_seconds']}s (≤0.5s: "
          f"{result['avg_interval_seconds'] <= 0.5})")
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
