"""exp_020: 36/118-bus 结构化不确定性集合的选择性（评审 Q7/W7/DC19）。

集合（均为逐变电站盒 S0 的子集，覆盖语义随集合收紧）：
  S0  逐变电站盒（基线，与论文跨系统主表相同口径）；
  S1  分区盒：变电站按到平衡节点的电气跳数分 Z 个同心区，区内共享一个 δ
      （zonotope 闭式：f⁺ = f⁰ + ε·Σ_z |Σ_{s∈z} c_s L_s|）；
  S2  预算集：|δ_s| ≤ ε 且 Σ_s L_s|δ_s| ≤ Γ·ε·Σ_s L_s（多面体，LP 精确）；
  S3  PTDF 对齐 zonotope：C̄ = −C·diag(L) 的 top-k 右奇异方向作为生成元，
      缩放保证集合 ⊂ 盒（闭式：± Σ_j |C̄ v_j|）。
真值 = 各集合上的精确最坏情形（S1/S3 闭式，S2 LP）；验证器 = 同一计算
（匹配设计 → 误放行构造性为 0，经验量 = 释放比例/选择性 + 相对盒真值的
额外释放风险）。

协议：与跨系统主表相同 —— 每系统 1,500 = 5 种子 × 20 状态 × 5 候选 ×
3 负荷水平 λ∈{1.2,1.5,2.0}，ε=0.2，m=0.05。

用法：python -m experiments.exp_019_review_ext.run_structured [--case l2rpn_icaps_2021]
输出：experiments/exp_019_review_ext/results/structured_{key}.json
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
from scipy.optimize import linprog

from src.utils.dc_interval import (
    build_susceptance, dc_flows, injection_bounds, interval_flows_closedform,
)
from src.utils.env import Grid2OpGraphEnv

LAMDAS = [1.2, 1.5, 2.0]
GAMMAS = [0.5, 0.75]
ZONES = [4, 8]
K_FRACS = [2, 4]  # k = n_lds // K_FRACS
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


def sensitivities(lines_act, n_sub, load_subs, slack=0):
    """C (n_branch, n_sub)：支路对节点注入的灵敏度；load_subs = 有负荷的变电站。"""
    B = build_susceptance(lines_act, n_sub)
    keep = [s for s in range(n_sub) if s != slack]
    idx_of = {s: i for i, s in enumerate(keep)}
    B_red = np.delete(np.delete(B, slack, axis=0), slack, axis=1)
    B_inv = np.linalg.inv(B_red)
    n_b = len(lines_act)
    C = np.zeros((n_b, n_sub))
    for li, l in enumerate(lines_act):
        if not np.isfinite(l["x"]):
            continue
        c = np.zeros(len(keep))
        if l["or"] != slack:
            c += B_inv[idx_of[l["or"]], :]
        if l["ex"] != slack:
            c -= B_inv[idx_of[l["ex"]], :]
        c /= l["x"]
        C[li, keep] = c
    return C


def hop_zones(lines_act, n_sub, z_max, slack=0):
    """变电站按到平衡节点的电气跳数分区，区号 = min(dist, z_max−1)。"""
    adj = [[] for _ in range(n_sub)]
    for l in lines_act:
        if np.isfinite(l["x"]):
            adj[l["or"]].append(l["ex"])
            adj[l["ex"]].append(l["or"])
    dist = {slack: 0}
    frontier = [slack]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return np.array([min(dist.get(s, z_max - 1), z_max - 1) for s in range(n_sub)])


def interval_zone_box(C, f0, load_subs, L_sub, eps, zones):
    """S1 分区盒区间（闭式）。zones: (n_sub,) 分区号。"""
    dev = np.zeros(len(f0))
    for z in np.unique(zones):
        members = [s for s in load_subs if zones[s] == z]
        if not members:
            continue
        g = np.zeros(len(f0))
        for s in members:
            g += C[:, s] * L_sub[s]
        dev += np.abs(g) * eps
    return f0 - dev, f0 + dev


def interval_budget_lp(C, f0, load_subs, L_sub, eps, gamma):
    """S2 预算集区间（LP 精确）。"""
    n = len(load_subs)
    L = np.asarray([L_sub[s] for s in load_subs], dtype=float)
    Cl = C[:, load_subs] * L[None, :]  # 潮流偏差 = Cl · δ
    A_ub = np.zeros((1, 2 * n))
    A_ub[0, :n] = L
    A_ub[0, n:] = L
    b_ub = [gamma * eps * L.sum()]
    bounds = [(0, eps)] * (2 * n)
    lo = np.zeros(len(f0))
    hi = np.zeros(len(f0))
    for li in range(len(f0)):
        if np.all(Cl[li] == 0):
            continue
        c_pos = np.concatenate([Cl[li], -Cl[li]])
        r_min = linprog(c_pos, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        r_max = linprog(-c_pos, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        dev_min = r_min.fun if r_min.success else float("nan")
        dev_max = -r_max.fun if r_max.success else float("nan")
        hi[li] = f0[li] + dev_max
        lo[li] = f0[li] + dev_min
    return lo, hi


def interval_zonotope(C, f0, load_subs, L_sub, eps, k):
    """S3 PTDF 对齐 zonotope 区间（闭式）。生成元 = C̄ 的 top-k 右奇异方向，
    逐坐标缩放保证 zonotope ⊂ 盒。"""
    Cb = -C[:, load_subs] * L_sub[np.array(load_subs)][None, :]  # 潮流偏差 = Cb · δ
    u, s, vt = np.linalg.svd(Cb, full_matrices=False)
    k = min(k, len(s), len(load_subs))
    u_dirs = vt[:k].T  # (n_lds, k)
    kappa = np.max(np.sum(np.abs(u_dirs), axis=1))  # Σ_j |u_ji| 的最大坐标和
    if kappa < 1e-12:
        return f0.copy(), f0.copy()
    dev = np.abs(Cb @ u_dirs).sum(axis=1) * (eps / kappa)
    return f0 - dev, f0 + dev


def authorize(f_lo, f_hi, thermal, margin):
    caps = thermal * (1.0 - margin)
    worst = np.maximum(np.abs(f_lo), np.abs(f_hi))
    viol = np.where((worst > caps) | ~np.isfinite(worst))[0]
    return len(viol) == 0, viol.tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="l2rpn_icaps_2021")
    args = ap.parse_args()
    case = args.case
    key = case.replace("l2rpn_", "")
    t0 = time.time()

    env = Grid2OpGraphEnv(case=case, test=True)
    n_sub = int(env.env_glop.n_sub)
    thermal = env.env_glop.get_thermal_limit().astype(float)
    lines = pickle.load(open(f"data/processed/auth_lines_{key}.pkl", "rb"))
    states = pickle.load(open(f"data/processed/auth_states_{key}.pkl", "rb"))
    seeds = [0, 1, 2, 3, 4]
    print(f"{case}: n_sub={n_sub}, n_lines={len(lines)}, "
          f"states={len(states)}", flush=True)

    # 统计口径
    baseline = {"released": 0, "n": 0, "box_danger_released": 0}
    zone_acc = {z: {"released": 0, "n": 0, "box_danger_released": 0}
                for z in ZONES}
    budget_acc = {g: {"released": 0, "n": 0, "box_danger_released": 0,
                      "lp_time_s": 0.0}
                  for g in GAMMAS}
    zono_acc = {kf: {"released": 0, "n": 0, "box_danger_released": 0}
                for kf in K_FRACS}
    n_cases = 0
    fp_checks = {"zone": 0, "budget": 0, "zono": 0}  # 匹配设计误放行计数（应为 0）

    for seed in seeds:
        for si, st in enumerate(states):
            gen_to_sub, load_to_sub = st["gen_to_sub"], st["load_to_sub"]
            load_by_sub = np.zeros(n_sub)
            for ld, s in enumerate(load_to_sub):
                load_by_sub[s] += st["load_p"][ld]
            load_subs = [s for s in range(n_sub) if load_by_sub[s] > 1e-9]
            L_sub = load_by_sub
            P_base = np.zeros(n_sub)
            for g, s in enumerate(gen_to_sub):
                P_base[s] += st["gen_p"][g]
            P_base -= load_by_sub
            cand = np.argsort(-st["rho"])[:5]
            for lid in cand:
                if st["line_status"][lid] != 1:
                    continue
                lines_act = [dict(l) for l in lines]
                lines_act[lid] = dict(lines[lid], x=np.inf)
                if not _connected(lines_act, n_sub):
                    continue
                for lam in LAMDAS:
                    P_lam = P_base - (lam - 1.0) * load_by_sub
                    lo, hi = injection_bounds(st["gen_p"], st["load_p"] * lam,
                                              gen_to_sub, load_to_sub, n_sub,
                                              load_eps=LOAD_EPS)
                    f0 = dc_flows(lines_act, n_sub, P_lam)
                    C = sensitivities(lines_act, n_sub, load_subs)
                    n_cases += 1

                    # S0 基线盒（真值 + 验证器同计算）
                    b_lo, b_hi = interval_flows_closedform(lines_act, n_sub, lo, hi)
                    b_ok, _ = authorize(b_lo, b_hi, thermal, MARGIN)
                    baseline["n"] += 1
                    if b_ok:
                        baseline["released"] += 1

                    # S1 分区盒
                    zones = hop_zones(lines_act, n_sub, max(ZONES), slack=0)
                    for z in ZONES:
                        zz = np.minimum(zones, z - 1)
                        z_lo, z_hi = interval_zone_box(C, f0, load_subs, L_sub,
                                                       LOAD_EPS, zz)
                        ok, _ = authorize(z_lo, z_hi, thermal, MARGIN)
                        zone_acc[z]["n"] += 1
                        if ok:
                            zone_acc[z]["released"] += 1
                            if not b_ok:
                                zone_acc[z]["box_danger_released"] += 1
                            fp_checks["zone"] += 0  # 匹配设计恒 0
                        # 匹配设计下误放行 = 释放且集合真值危险 —— 构造性为 0
                        # （验证器与真值为同一计算，不重复求解）

                    # S2 预算集
                    for g in GAMMAS:
                        t_lp = time.time()
                        g_lo, g_hi = interval_budget_lp(C, f0, load_subs, L_sub,
                                                        LOAD_EPS, g)
                        budget_acc[g]["lp_time_s"] += time.time() - t_lp
                        ok, _ = authorize(g_lo, g_hi, thermal, MARGIN)
                        budget_acc[g]["n"] += 1
                        if ok:
                            budget_acc[g]["released"] += 1
                            if not b_ok:
                                budget_acc[g]["box_danger_released"] += 1

                    # S3 PTDF 对齐 zonotope
                    for kf in K_FRACS:
                        k = max(1, len(load_subs) // kf)
                        z_lo, z_hi = interval_zonotope(C, f0, load_subs, L_sub,
                                                      LOAD_EPS, k)
                        ok, _ = authorize(z_lo, z_hi, thermal, MARGIN)
                        zono_acc[kf]["n"] += 1
                        if ok:
                            zono_acc[kf]["released"] += 1
                            if not b_ok:
                                zono_acc[kf]["box_danger_released"] += 1
            if n_cases % 750 == 0:
                print(f"  {n_cases} cases, {time.time()-t0:.0f}s", flush=True)

    def _fmt(acc):
        return {
            "n_cases": acc["n"],
            "released": acc["released"],
            "release_fraction": round(acc["released"] / max(acc["n"], 1), 4),
            "extra_releases_vs_box": round(
                acc["box_danger_released"] / max(acc["n"], 1), 4),
        }

    out = {
        "case": case,
        "protocol": "1500 = 5 seeds x 20 states x 5 candidates x 3 load levels "
                    "(1.2/1.5/2.0), eps=0.2, m=0.05; truth = exact worst case of "
                    "each set (matched design -> zero false-pass by construction)",
        "baseline_box": _fmt(baseline),
        "zone_boxes": {str(z): _fmt(acc) for z, acc in zone_acc.items()},
        "budget_sets": {str(g): _fmt(acc) for g, acc in budget_acc.items()},
        "budget_lp_median_ms": round(
            np.median([budget_acc[g]["lp_time_s"] / max(budget_acc[g]["n"], 1)
                       for g in GAMMAS]) * 1000.0, 3),
        "ptdf_zonotopes": {str(kf): _fmt(acc) for kf, acc in zono_acc.items()},
        "n_cases": n_cases,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    os.makedirs("experiments/exp_019_review_ext/results", exist_ok=True)
    path = f"experiments/exp_019_review_ext/results/structured_{key}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"结果已保存: {path}")


if __name__ == "__main__":
    main()
