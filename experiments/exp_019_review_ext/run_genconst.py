"""exp_021: 发电机约束与分布式平衡（评审 Q1/Q2/DC16）。

Part A 分布式平衡参与因子灵敏度（14-bus，7,500 压力用例 = 5 种子 ×
100 状态 × 5 候选 × λ∈{2,3,4}）：
  参与模型 α ∈ {slack（单平衡，基线）、uniform（发电机变电站等权）、
  prop（按观测出力占比）}；参与调整后灵敏度 c̃_s = −c_s + Σ_k α_k c_k
  （eq.(7)），角点区间闭式。
  真值 × 验证器 = 3×3 矩阵：对角线 = 匹配设计（误放行构造性 0），
  非对角线 = 参与模型失配的泄漏/保守性（评审 Q1 的灵敏度答案）。

Part B 平衡容量约束（14-bus，1,500 压力用例 = 5 种子 × 20 状态 × 5 候选 ×
λ∈{2,3,4}）：
  平衡节点注入约束 |P_s − P_s⁰| ≤ η·ΣL（η ∈ {0.05, 0.10, 0.20=无约束}，
  ΣL = 系统总负荷）。真值 = 有界集合上的精确最坏情形（角点可行则闭式，
  否则 LP）；验证器有界 = 同一计算（匹配 → 0 误放行）；
  对比无界验证器（现有闭式）在同一真值下的误拒率 → 量化"发电机约束
  恢复被不真实角点误拒的动作"（评审 Q2 的释放/误拒/时延影响）。

用法：python -m experiments.exp_019_review_ext.run_genconst
输出：experiments/exp_019_review_ext/results/genconst.json
"""
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

LAMDAS = [2.0, 3.0, 4.0]
LOAD_EPS = 0.2
MARGIN = 0.05
ETAS = [0.05, 0.10, 0.20]


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


def ptdf_rows(lines_act, n_sub, slack=0):
    """C (n_branch, n_sub)：支路对节点注入的灵敏度（向量化）。"""
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


def participation_interval(C, f0, load_subs, L_sub, alpha, eps):
    """参与模型 α 下的角点区间：f ± ε·Σ_s |c̃_s|·L_s，c̃_s = −c_s + Σ_k α_k c_k。"""
    c_sum = np.zeros(len(f0))
    for k in alpha:
        c_sum += alpha[k] * C[:, k]
    dev = np.zeros(len(f0))
    for s in load_subs:
        ct = -C[:, s] + c_sum
        dev += np.abs(ct) * L_sub[s]
    return f0 - eps * dev, f0 + eps * dev


def slack_bounded_worst(C_row, lo, hi, slack_nom, bound, keep):
    """平衡容量约束下单支路最坏情形：角点可行用闭式，否则 LP。
    返回 (f_lo, f_hi, used_lp)。"""
    lo_v = np.asarray([lo[s] for s in keep], dtype=float)
    hi_v = np.asarray([hi[s] for s in keep], dtype=float)
    c = C_row[keep]
    used_lp = [False]

    def worst_dir(sign):
        P_red = np.where(sign * c >= 0, hi_v, lo_v)
        f_val = float(sign * np.sum(np.where(sign * c >= 0, sign * c * hi_v,
                                             sign * c * lo_v)))
        P_s = -float(np.sum(P_red))  # 隐含平衡注入
        if abs(P_s - slack_nom) <= bound + 1e-9:
            return f_val
        used_lp[0] = True
        # LP：max sign·c·P s.t. |−ΣP − slack_nom| ≤ bound
        A_ub = np.zeros((2, len(keep)))
        A_ub[0, :] = -1.0
        A_ub[1, :] = 1.0
        b_ub = [slack_nom + bound, -slack_nom + bound]
        r = linprog(-sign * c, A_ub=A_ub, b_ub=b_ub,
                    bounds=list(zip(lo_v, hi_v)), method="highs")
        if not r.success:
            return float("nan")
        return float(sign * -r.fun)

    f_hi = worst_dir(1.0)
    f_lo = worst_dir(-1.0)
    return f_lo, f_hi, used_lp[0]


def main():
    t0 = time.time()
    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    n_sub = int(env.env_glop.n_sub)
    thermal = env.env_glop.get_thermal_limit().astype(float)
    caps = thermal * (1.0 - MARGIN)
    lines = pickle.load(open("data/processed/auth_lines_rte_case14_realistic.pkl", "rb"))
    states = pickle.load(open("data/processed/auth_states_rte_case14_realistic.pkl", "rb"))
    seeds = [0, 1, 2, 3, 4]
    print(f"rte_case14: n_sub={n_sub}, n_lines={len(lines)}, states={len(states)}",
          flush=True)

    # ---------------- Part A：参与因子（100 状态，7,500 用例） ----------------
    variants = ["slack", "uniform", "prop"]
    # 每用例：真值 danger 按三种参与模型分别计算；验证器判定按三种模型分别计算
    part_acc = {(tv, vv): {"released": 0, "danger_released": 0, "rejected": 0,
                           "safe_rejected": 0}
                for tv in variants for vv in variants}
    n_cases_a = 0
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
            gen_subs = list(set(gen_to_sub.tolist()))
            gen_by_sub = {}
            for g, s in enumerate(gen_to_sub):
                gen_by_sub[s] = gen_by_sub.get(s, 0.0) + float(st["gen_p"][g])
            alphas = {
                "slack": {0: 1.0},
                "uniform": {k: 1.0 / len(gen_subs) for k in gen_subs},
                "prop": {k: v / max(sum(gen_by_sub.values()), 1e-9)
                         for k, v in gen_by_sub.items()},
            }
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
                    f0 = dc_flows(lines_act, n_sub, P_lam)
                    C = ptdf_rows(lines_act, n_sub)
                    n_cases_a += 1
                    ints = {v: participation_interval(C, f0, load_subs, L_sub,
                                                      alphas[v], LOAD_EPS)
                            for v in variants}
                    # 真值与判定
                    for tv in variants:
                        t_lo, t_hi = ints[tv]
                        danger = bool(np.any(np.maximum(np.abs(t_lo), np.abs(t_hi)) > caps))
                        for vv in variants:
                            v_lo, v_hi = ints[vv]
                            worst = np.maximum(np.abs(v_lo), np.abs(v_hi))
                            ok = bool(np.all(worst <= caps) and np.all(np.isfinite(worst)))
                            acc = part_acc[(tv, vv)]
                            if ok:
                                acc["released"] += 1
                                acc["danger_released"] += int(danger)
                            else:
                                acc["rejected"] += 1
                                acc["safe_rejected"] += int(not danger)
        print(f"  PartA {n_cases_a} cases, {time.time()-t0:.0f}s", flush=True)

    part = {}
    for tv in variants:
        for vv in variants:
            acc = part_acc[(tv, vv)]
            part[f"truth={tv},verifier={vv}"] = {
                "released_fraction": round(acc["released"] / max(n_cases_a, 1), 4),
                "false_pass_rate": round(acc["danger_released"] / max(acc["released"], 1), 4),
                "false_reject_rate": round(acc["safe_rejected"] / max(acc["rejected"], 1), 4),
            }

    # ---------------- Part B：平衡容量约束（20 状态，1,500 用例） ----------------
    n_states_b = 20
    bound_acc = {eta: {"n": 0, "truth_danger": 0, "bounded_released": 0,
                       "bounded_danger_released": 0, "unbounded_released": 0,
                       "unbounded_safe_rejected": 0, "unbounded_rejected": 0,
                       "lp_time_s": 0.0, "n_lp": 0, "n_corner_ok": 0}
                 for eta in ETAS}
    n_cases_b = 0
    for seed in seeds:
        for si, st in enumerate(states[:n_states_b]):
            gen_to_sub, load_to_sub = st["gen_to_sub"], st["load_to_sub"]
            load_by_sub = np.zeros(n_sub)
            for ld, s in enumerate(load_to_sub):
                load_by_sub[s] += st["load_p"][ld]
            L_tot = float(load_by_sub.sum())
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
                    keep = [s for s in range(n_sub) if s != 0]
                    slack_nom = -float(np.sum(P_lam[keep]))
                    f0 = dc_flows(lines_act, n_sub, P_lam)
                    C = ptdf_rows(lines_act, n_sub)
                    n_cases_b += 1
                    # 无界验证器（现有闭式）
                    u_lo, u_hi = interval_flows_closedform(lines_act, n_sub, lo, hi)
                    u_ok = bool(np.all(np.maximum(np.abs(u_lo), np.abs(u_hi)) <= caps))
                    for eta in ETAS:
                        bound = eta * L_tot
                        acc = bound_acc[eta]
                        acc["n"] += 1
                        if u_ok:
                            acc["unbounded_released"] += 1
                        else:
                            acc["unbounded_rejected"] += 1
                        # 真值与有界验证器 = 同一计算（匹配设计）
                        danger = False
                        released = True
                        for li in range(len(lines_act)):
                            if not np.isfinite(lines_act[li]["x"]):
                                continue
                            t_lp = time.time()
                            f_lo, f_hi, used_lp = slack_bounded_worst(
                                C[li], lo, hi, slack_nom, bound, keep)
                            dt = time.time() - t_lp
                            if used_lp:
                                acc["n_lp"] += 2
                                acc["lp_time_s"] += dt
                            else:
                                acc["n_corner_ok"] += 1
                            if not np.isfinite(f_lo) or not np.isfinite(f_hi):
                                released = False
                                danger = True
                                continue
                            if max(abs(f_lo), abs(f_hi)) > caps[li]:
                                released = False
                                danger = True
                        acc["truth_danger"] += int(danger)
                        if released:
                            acc["bounded_released"] += 1
                            if danger:
                                acc["bounded_danger_released"] += 1
                        elif not danger:
                            # 有界真值下安全但被有界验证器拒绝（不应发生，匹配设计）
                            pass
                        if u_ok is False and not danger:
                            acc["unbounded_safe_rejected"] += 1
        print(f"  PartB {n_cases_b} cases, {time.time()-t0:.0f}s", flush=True)

    bound = {}
    for eta in ETAS:
        acc = bound_acc[eta]
        bound[str(eta)] = {
            "n_cases": acc["n"],
            "truth_danger_fraction": round(acc["truth_danger"] / max(acc["n"], 1), 4),
            "bounded_verifier_release_fraction": round(
                acc["bounded_released"] / max(acc["n"], 1), 4),
            "bounded_verifier_false_pass": round(
                acc["bounded_danger_released"] / max(acc["bounded_released"], 1), 4),
            "unbounded_verifier_release_fraction": round(
                acc["unbounded_released"] / max(acc["n"], 1), 4),
            "unbounded_verifier_false_reject_rate_vs_bounded_truth": round(
                acc["unbounded_safe_rejected"] / max(acc["unbounded_rejected"], 1), 4),
            "release_recovery_vs_unbounded": round(
                (acc["bounded_released"] - acc["unbounded_released"]) / max(acc["n"], 1), 4),
            "n_lp_solves": acc["n_lp"],
            "lp_ms_per_action": round(
                acc["lp_time_s"] / max(acc["n"], 1) * 1000.0, 3),
            "corner_feasible_no_lp_count": acc["n_corner_ok"],
        }
    # 有界 LP 每用例时延（含角点可行免 LP 的用例）
    out = {
        "case": "rte_case14_realistic",
        "partA_protocol": "7500 = 5 seeds x 100 states x 5 candidates x lambda in {2,3,4}",
        "partA_participation": part,
        "partB_protocol": "1500 = 5 seeds x 20 states x 5 candidates x lambda in {2,3,4}; "
                          "slack bound |P_s - P_s0| <= eta * total_load",
        "partB_slack_bounds": bound,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    os.makedirs("experiments/exp_019_review_ext/results", exist_ok=True)
    path = "experiments/exp_019_review_ext/results/genconst.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"结果已保存: {path}")


if __name__ == "__main__":
    main()
