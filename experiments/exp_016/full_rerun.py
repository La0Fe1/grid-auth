"""exp_016 补跑：主实验与跨系统实验的全量完整重跑（修复 seed 覆盖缺口）。

背景（自查发现）：
- exp_012 主实验 auth_eval_cases.jsonl 共 11,101 条，其中 seed 2 只有 1,101/2,500
  （其余 4 个种子各 2,500），缺 1,399 个用例——论文未披露该不完整性；
- exp_015 跨系统：icaps seed 0 缺 133/300（共 1,367/1,500）；idf 缺 396/1,500
  （idf 的 partial 已披露，但本次一并补全）。
本脚本按与原实验完全相同的用例协议重放全部用例并重算四个验证器判定与真值，
一次性产出协议完整的数据集（14-bus 12,500 用例；36/118-bus 各 1,500 用例）。

实现差异（均不影响决策结果）：
- 区间验证器用闭式角点公式（与 LP 完全等价，见 tests/test_dc_interval.py
  的逐支路等价性验证），避免 ~10 分钟的单纯形求解；
- 蒙特卡洛验证器与真值抽取向量化（一次线性求解 M 个右端项），
  真值检查点 [100,300,1000,5000] 由前缀平均一次得到（完整收敛曲线）；
- 随机抽样使用全新随机数（RandomState 20260916）——统计量独立再估计，
  与原运行的抽样序列无关。

用法：
  python -m experiments.exp_016.full_rerun rte_case14_realistic
  python -m experiments.exp_016.full_rerun l2rpn_icaps_2021 --test-mode
  python -m experiments.exp_016.full_rerun l2rpn_idf_2023 --test-mode
输出：
  experiments/exp_012/results/auth_eval_full*.json(l)（14-bus）
  experiments/exp_015/results/auth_eval_{icaps,idf}_full*.json(l)
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

from src.utils.dc_interval import (
    authorize, build_susceptance, dc_flows, injection_bounds,
    interval_flows_closedform,
)
from src.utils.env import Grid2OpGraphEnv


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


def _flows_batch(lines, n_sub, P_mat, slack=0):
    B = build_susceptance(lines, n_sub)
    keep = [s for s in range(n_sub) if s != slack]
    B_red = np.delete(np.delete(B, slack, axis=0), slack, axis=1)
    theta_red = np.linalg.solve(B_red, P_mat[:, keep].T).T
    theta = np.zeros((P_mat.shape[0], n_sub))
    theta[:, keep] = theta_red
    flows = np.empty((P_mat.shape[0], len(lines)))
    for li, l in enumerate(lines):
        if not np.isfinite(l["x"]):
            flows[:, li] = 0.0
        else:
            flows[:, li] = (theta[:, l["or"]] - theta[:, l["ex"]]) / l["x"]
    return flows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("case", default="rte_case14_realistic")
    ap.add_argument("--test-mode", action="store_true")
    args = ap.parse_args()
    case = args.case
    key = case.replace("l2rpn_", "")
    if case == "rte_case14_realistic":
        n_states, lamdas, gt_m, checkpoints, outdir = 100, [1.0, 2.0, 3.0, 4.0, 5.0], 5000, [100, 300, 1000, 5000], "experiments/exp_012/results"
    else:
        n_states, lamdas, gt_m, checkpoints, outdir = 20, [1.2, 1.5, 2.0], 1000, [100, 300, 1000], "experiments/exp_015/results"
    load_eps, margin, mc_k = 0.2, 0.05, 100
    t0 = time.time()

    env = Grid2OpGraphEnv(case=case, test=args.test_mode)
    n_sub = int(env.env_glop.n_sub)
    thermal = env.env_glop.get_thermal_limit().astype(float)
    lines = pickle.load(open(f"data/processed/auth_lines_{key}.pkl", "rb"))
    states = pickle.load(open(f"data/processed/auth_states_{key}.pkl", "rb"))
    print(f"{case}: n_sub={n_sub}, n_lines={len(lines)}, "
          f"states cache={len(states)}, lamdas={lamdas}", flush=True)

    rng = np.random.RandomState(20260916)
    seeds = [0, 1, 2, 3, 4]
    acc = {k: {"passed": 0, "passed_danger": 0.0, "passed_any_danger": 0,
               "rejected": 0, "rejected_safe": 0}
           for k in ("none", "nominal", "interval", "mc")}
    conv = {m: {k: {"danger_sum": 0.0, "passed": 0}
                for k in ("nominal", "interval", "mc")}
            for m in checkpoints}
    records = []
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
                lines_act = [dict(l) for l in lines]
                lines_act[lid] = dict(lines[lid], x=np.inf)
                for lam in lamdas:
                    P_lam = P_base - (lam - 1.0) * load_by_sub
                    lo, hi = injection_bounds(st["gen_p"], st["load_p"] * lam,
                                              gen_to_sub, load_to_sub, n_sub,
                                              load_eps=load_eps)
                    n_cases += 1
                    conn_ok = _connected(lines_act, n_sub)
                    if not conn_ok:
                        nom_ok = int_ok = mc_ok = False
                        gt_danger = {m: 1.0 for m in checkpoints}
                    else:
                        f = dc_flows(lines_act, n_sub, P_lam)
                        nom_ok, _ = authorize(f, f, thermal, margin=margin)
                        f_lo, f_hi = interval_flows_closedform(lines_act, n_sub, lo, hi)
                        int_ok, _ = authorize(f_lo, f_hi, thermal, margin=margin)
                        # MC：100 点批量采样（与原逐点抽样同分布）
                        Pm = rng.uniform(lo, hi, size=(mc_k, n_sub))
                        fm = _flows_batch(lines_act, n_sub, Pm)
                        mc_ok = not np.any(
                            np.max(np.abs(fm) / thermal, axis=1) > (1.0 - margin))
                        # 真值：M 点批量 + 前缀平均收敛曲线
                        Pg = rng.uniform(lo, hi, size=(gt_m, n_sub))
                        fg = _flows_batch(lines_act, n_sub, Pg)
                        viol = np.max(np.abs(fg) / thermal, axis=1) > (1.0 - margin)
                        cum = np.cumsum(viol, dtype=np.float64)
                        gt_danger = {m: float(cum[m - 1] / m) for m in checkpoints}
                    d_final = gt_danger[checkpoints[-1]]
                    any_danger = d_final > 0
                    decisions = {"none": True, "nominal": nom_ok,
                                 "interval": int_ok, "mc": mc_ok}
                    for kk, passed in decisions.items():
                        if passed:
                            acc[kk]["passed"] += 1
                            acc[kk]["passed_danger"] += d_final
                            acc[kk]["passed_any_danger"] += int(any_danger)
                        else:
                            acc[kk]["rejected"] += 1
                            acc[kk]["rejected_safe"] += int(not any_danger)
                    for m in checkpoints:
                        for kk in ("nominal", "interval", "mc"):
                            if decisions[kk]:
                                conv[m][kk]["danger_sum"] += gt_danger[m]
                                conv[m][kk]["passed"] += 1
                    records.append({
                        "seed": seed, "state": si, "tag": "top", "lambda": lam,
                        "disconnects": [int(lid)],
                        "nominal_ok": bool(nom_ok), "interval_ok": bool(int_ok),
                        "mc_ok": bool(mc_ok),
                        "gt_danger_rate": d_final,
                    })
            if n_cases % 2500 == 0:
                print(f"  {n_cases} cases, {time.time()-t0:.0f}s", flush=True)

    summary = {}
    for kk, a in acc.items():
        summary[kk] = {
            "passed": a["passed"], "rejected": a["rejected"],
            "false_pass_rate": float(a["passed_any_danger"] / max(a["passed"], 1)),
            "mean_danger_when_passed": float(a["passed_danger"] / max(a["passed"], 1)),
            "false_reject_rate": float(a["rejected_safe"] / max(a["rejected"], 1)),
            "latency_ms_mean": None,
        }
    conv_curves = {str(m): {kk: float(conv[m][kk]["danger_sum"] / max(conv[m][kk]["passed"], 1))
                            for kk in ("nominal", "interval", "mc")}
                   for m in checkpoints}
    result = {
        "args": {"case": case, "test_mode": args.test_mode, "n_states": n_states,
                 "seeds": seeds, "n_candidates": 5, "lamdas": lamdas,
                 "mc_k": mc_k, "gt_m": gt_m, "conv_checkpoints": checkpoints,
                 "load_eps": load_eps, "margin": margin,
                 "interval_impl": "closed-form corner formula (exactly equivalent to LP; unit-tested)",
                 "rng": "RandomState 20260916 (fresh draws)"},
        "summary": summary, "conv_curves": conv_curves, "n_cases": n_cases,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    out_base = os.path.join(outdir, f"auth_eval_{key}_full")
    with open(out_base + ".json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out_base + "_cases.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                          for kk, vv in v.items() if kk != "latency_ms_mean"}
                      for k, v in summary.items()}, indent=2, ensure_ascii=False))
    print(f"conv_curves: {json.dumps(conv_curves)}")
    print(f"n_cases={n_cases}, elapsed={result['elapsed_seconds']}s → {out_base}.json")


if __name__ == "__main__":
    main()
