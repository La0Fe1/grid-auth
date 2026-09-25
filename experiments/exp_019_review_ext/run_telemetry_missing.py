"""exp_022: 遥测缺失/偏差/异步模式的辨识与授权决策稳健性（评审 Q5/W4）。

模式（均从 T=60 快照最小二乘辨识出发）：
  clean   基线（无扰动）；
  drop50/drop80  每条支路每个快照以概率 ρ 缺失成对 (Δθ, p_or) 样本
                 （ρ ∈ {0.5, 0.8}；<2 个有效样本的支路回退网络文件电抗，
                 与 §2.2 的回退策略一致）；
  bias05/bias10  每状态每变电站角度读数加独立 N(0, σ_b) 偏差
                 （σ_b ∈ {0.05, 0.10} rad，不抵消、进入 Δθ 作为相关误差）；
  lag1    异步配对：Δθ 序列与 p_or 序列错位 1 个快照；
  nopmu30 30% 支路完全无遥测（部分 PMU 覆盖）→ 网络文件回退。
评估：种子 7 协议下的 AC 保真度 r（同 ident_noise）+ 7,500 压力用例
（5 种子 × 100 状态 × 5 候选 × λ∈{2,3,4}）上以各退化模型做区间验证，
真值 = 真实模型上的精确角点判据 → 误放行/误拒/释放比例。

用法：python -m experiments.exp_019_review_ext.run_telemetry_missing
输出：experiments/exp_019_review_ext/results/telemetry_missing.json
"""
import json
import os
import pickle
import time

import numpy as np

from src.utils.dc_interval import (
    dc_flows, injection_bounds, interval_flows_closedform,
)
from src.utils.env import Grid2OpGraphEnv

LAMDAS = [2.0, 3.0, 4.0]
LOAD_EPS = 0.2
MARGIN = 0.05


def _connected(lines_act, n_sub, slack=0):
    adj = [[] for _ in range(n_sub)]
    for l in lines_act:
        if np.isfinite(l["x"]):
            adj[l["or"]].append(l["ex"])
            adj[l["ex"]].append(l["or"])
    seen, stack = {slack}, [slack]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n_sub


def collect_telemetry(env, n_states, rng, drop_p=0.0, bias_sigma=0.0, lag=0,
                      drop_branch_frac=0.0):
    """采集快照并施加缺失/偏差/异步扰动。返回 {line_id: (d_array, p_array)}。"""
    n_line = env.n_line
    dtp, pp = [[] for _ in range(n_line)], [[] for _ in range(n_line)]
    prev_d = [None] * n_line
    dead = np.zeros(n_line, dtype=bool)  # 整支路缺失（部分 PMU 覆盖）
    if drop_branch_frac > 0:
        n_dead = max(1, int(round(n_line * drop_branch_frac)))
        dead[np.random.RandomState(20260916).choice(
            n_line, n_dead, replace=False)] = True
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
        bus_bias = rng.normal(0.0, bias_sigma,
                              size=int(env.env_glop.n_sub))
        for l in range(n_line):
            if obs.line_status[l] == 0 or dead[l]:
                continue
            d = float(obs.theta_or[l] - obs.theta_ex[l])
            p = float(obs.p_or[l])
            if bias_sigma > 0:
                d += float(bus_bias[obs.line_or_to_subid[l]]
                           - bus_bias[obs.line_ex_to_subid[l]])
            if drop_p > 0 and rng.rand() < drop_p:
                prev_d[l] = d
                continue
            if lag > 0:
                if prev_d[l] is None:
                    prev_d[l] = d
                    continue
                d_use = prev_d[l]
                prev_d[l] = d
            else:
                d_use = d
            dtp[l].append(d_use)
            pp[l].append(p)
    return dtp, pp


def identify(dtp, pp, n_line, n_sub, fallback_lines, obs_proto):
    """最小二乘辨识；有效样本 <2 的支路回退网络文件电抗。"""
    lines = []
    n_fallback = 0
    for l in range(n_line):
        d, p = np.array(dtp[l]), np.array(pp[l])
        if len(d) < 2 or np.abs(p).sum() < 1e-9:
            lines.append(dict(fallback_lines[l]))
            n_fallback += 1
            continue
        num, den = float((d * p).sum()), float((p * p).sum())
        x_ls = num / den if den > 1e-12 else float(fallback_lines[l]["x"])
        lines.append({"or": int(obs_proto.line_or_to_subid[l]),
                      "ex": int(obs_proto.line_ex_to_subid[l]),
                      "x": max(x_ls, 1e-6)})
    return lines, n_fallback


def fidelity_r(env, lines, n_sub):
    """种子 7 三状态 60 支路观测的 Pearson r（同 ident_noise 协议）。"""
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
    return float(np.corrcoef(f, p)[0, 1])


def main():
    t0 = time.time()
    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    n_sub = int(env.env_glop.n_sub)
    n_line = env.n_line
    thermal = env.env_glop.get_thermal_limit().astype(float)
    caps = thermal * (1.0 - MARGIN)
    true_lines = pickle.load(open("data/processed/auth_lines_rte_case14_realistic.pkl", "rb"))
    states = pickle.load(open("data/processed/auth_states_rte_case14_realistic.pkl", "rb"))
    seeds = [0, 1, 2, 3, 4]

    patterns = {
        "clean": dict(),
        "drop50": dict(drop_p=0.5),
        "drop80": dict(drop_p=0.8),
        "bias05": dict(bias_sigma=0.05),
        "bias10": dict(bias_sigma=0.10),
        "lag1": dict(lag=1),
        "nopmu30": dict(drop_branch_frac=0.3),
    }
    models = {}
    obs_proto = None
    rng_id = np.random.RandomState(20260916)
    for name, kw in patterns.items():
        dtp, pp = collect_telemetry(env, 60, rng_id, **kw)
        if obs_proto is None:
            env.env_glop.reset()
            obs_proto = env.env_glop.get_obs()
        lines, n_fb = identify(dtp, pp, n_line, n_sub, true_lines, obs_proto)
        models[name] = (lines, n_fb)
        print(f"  model {name}: n_fallback={n_fb}, {time.time()-t0:.0f}s", flush=True)

    # AC 保真度
    fidelity = {name: round(fidelity_r(env, lines, n_sub), 4)
                for name, (lines, _) in models.items()}

    # 授权决策评估（7,500 压力用例，真值 = 真实模型精确角点）
    eval_acc = {name: {"released": 0, "danger_released": 0, "rejected": 0,
                       "safe_rejected": 0} for name in models}
    n_cases = 0
    for seed in seeds:
        for si, st in enumerate(states):
            gen_to_sub, load_to_sub = st["gen_to_sub"], st["load_to_sub"]
            load_by_sub = np.zeros(n_sub)
            for ld, s in enumerate(load_to_sub):
                load_by_sub[s] += st["load_p"][ld]
            P_base = np.zeros(n_sub)
            for g, s in enumerate(gen_to_sub):
                P_base[s] += st["gen_p"][g]
            P_base -= load_by_sub
            cand = np.argsort(-st["rho"])[:5]
            for lid in cand:
                if st["line_status"][lid] != 1:
                    continue
                lines_act_true = [dict(l) for l in true_lines]
                lines_act_true[lid] = dict(true_lines[lid], x=np.inf)
                if not _connected(lines_act_true, n_sub):
                    continue
                for lam in LAMDAS:
                    P_lam = P_base - (lam - 1.0) * load_by_sub
                    lo, hi = injection_bounds(st["gen_p"], st["load_p"] * lam,
                                              gen_to_sub, load_to_sub, n_sub,
                                              load_eps=LOAD_EPS)
                    # 真值（真实模型精确角点）
                    t_lo, t_hi = interval_flows_closedform(lines_act_true, n_sub, lo, hi)
                    danger = bool(np.any(np.maximum(np.abs(t_lo), np.abs(t_hi)) > caps))
                    for name, (mlines, _) in models.items():
                        lines_act = [dict(l) for l in mlines]
                        lines_act[lid] = dict(mlines[lid], x=np.inf)
                        if not _connected(lines_act, n_sub):
                            ok = False
                        else:
                            v_lo, v_hi = interval_flows_closedform(lines_act, n_sub, lo, hi)
                            worst = np.maximum(np.abs(v_lo), np.abs(v_hi))
                            ok = bool(np.all(worst <= caps) and np.all(np.isfinite(worst)))
                        acc = eval_acc[name]
                        if ok:
                            acc["released"] += 1
                            acc["danger_released"] += int(danger)
                        else:
                            acc["rejected"] += 1
                            acc["safe_rejected"] += int(not danger)
                    n_cases += 1
        print(f"  eval seed {seed} done, {time.time()-t0:.0f}s", flush=True)

    out = {
        "case": "rte_case14_realistic",
        "protocol": "identification from 60 telemetry snapshots under missing/bias/"
                    "asynchrony patterns; eval on 7500 = 5 seeds x 100 states x "
                    "5 candidates x lambda in {2,3,4}, truth = exact corner "
                    "criterion on the true model (eps=0.2, m=0.05)",
        "fidelity_pearson_r": fidelity,
        "models": {name: {
            "n_fallback_branches": n_fb,
            "released_fraction": round(eval_acc[name]["released"] / max(n_cases, 1), 4),
            "false_pass_rate": round(eval_acc[name]["danger_released"] / max(eval_acc[name]["released"], 1), 4),
            "false_reject_rate": round(eval_acc[name]["safe_rejected"] / max(eval_acc[name]["rejected"], 1), 4),
        } for name, (_, n_fb) in models.items()},
        "n_cases": n_cases,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    os.makedirs("experiments/exp_019_review_ext/results", exist_ok=True)
    path = "experiments/exp_019_review_ext/results/telemetry_missing.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"结果已保存: {path}")


if __name__ == "__main__":
    main()
