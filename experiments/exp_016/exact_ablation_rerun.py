"""exp_016 补跑：消融/鲁棒性配置的精确真值统一重跑（审稿意见 Q1 的推广）。

背景：消融/鲁棒性实验原在 20 状态、采样真值（M=1000）下运行，且案例文件
无 state 字段无法回放。本脚本按统一口径重跑全部 13 个消融 + 7 个鲁棒性配置：
- 案例集升级为主实验同款：100 状态 × 5 种子 × 5 候选 × λ∈{2,3,4} = 7,500 例/配置；
- 真值统一为**精确判定**（真值盒 ε=0.2、m=0.05 上的闭式角点公式，孤岛=危险），
  与主实验口径一致；gaussian 量测噪声变体保持采样口径（其真值无界，标注）；
- 验证器判定与各配置原协议一致（标称/闭式区间/批量 MC/θ 代理/连通性/
  uniform-x 线路/遥测-30 线路/LLM 提案/随机提案/概率判据/裕度/ε 宽度）。

用法：python -m experiments.exp_016.exact_ablation_rerun
输出：experiments/exp_016/results/abl_rob_exact.json
"""
import json
import os
import pickle
import time

import numpy as np

from src.utils.dc_interval import (
    authorize, dc_flows, injection_bounds, interval_flows_closedform,
    theta_interval_closedform,
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


def run_config(name, states, lines, thermal, n_sub, *, load_eps=0.2, margin=0.05,
               gt_eps=0.2, gt_margin=0.05, require_connected=False,
               theta_cap=None, lines_variant=None, proposal_source="top",
               prob_threshold=None, gt_dist="uniform", missing_frac=0.0,
               llm_props=None, gt_lines=None):
    """gt_lines: 真值使用的线路模型（默认=验证器同模型，历史协议；
    2026-09-17 修正：真值必须用真实（遥测全量）模型，与验证器模型解耦）。"""
    acc = {k: {"passed": 0, "passed_danger": 0.0, "passed_any": 0,
               "rejected": 0, "rejected_safe": 0}
           for k in ("none", "nominal", "interval", "mc")}
    n_cases = 0
    rng_seed = np.random.RandomState(20260916)
    for seed in range(5):
        rng = np.random.RandomState(seed)
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
            if proposal_source == "llm":
                props = llm_props.get(si, []) if llm_props else []
                cases = [("llm", [int(l) for l in props])] if props else []
            elif proposal_source == "random":
                on_lines = np.nonzero(st["line_status"] == 1)[0]
                k = int(rng.randint(1, 4))
                picked = rng.choice(on_lines, size=min(k, len(on_lines)),
                                    replace=False).tolist()
                cases = [("random", picked)]
            else:
                cases = [("top", [int(lid)]) for lid in cand
                         if st["line_status"][lid] == 1]
            for tag, disconnects in cases:
                lines_act = [dict(l) for l in lines]
                for lid in disconnects:
                    lines_act[lid] = dict(lines[lid], x=np.inf)
                gt_lines = lines if gt_lines is None else gt_lines
                gt_lines_act = [dict(l) for l in gt_lines]
                for lid in disconnects:
                    gt_lines_act[lid] = dict(gt_lines[lid], x=np.inf)
                for lam in (2.0, 3.0, 4.0):
                    n_cases += 1
                    lo, hi = injection_bounds(st["gen_p"], st["load_p"] * lam,
                                              gen_to_sub, load_to_sub, n_sub,
                                              load_eps=load_eps)
                    gt_lo, gt_hi = injection_bounds(st["gen_p"], st["load_p"] * lam,
                                                    gen_to_sub, load_to_sub, n_sub,
                                                    load_eps=gt_eps)
                    if missing_frac > 0:
                        # 与原实验一致：不可观测负荷的验证器箱与真值箱同加宽到 ±100%
                        n_missing = max(1, int(len(load_to_sub) * missing_frac))
                        missing = rng_seed.choice(len(load_to_sub), n_missing,
                                                  replace=False)
                        for ld in missing:
                            s_ld = load_to_sub[ld]
                            lo[s_ld] = min(lo[s_ld], lo[s_ld] - st["load_p"][ld] * lam)
                            hi[s_ld] = max(hi[s_ld], hi[s_ld] + st["load_p"][ld] * lam)
                            gt_lo[s_ld] = min(gt_lo[s_ld], gt_lo[s_ld] - st["load_p"][ld] * lam)
                            gt_hi[s_ld] = max(gt_hi[s_ld], gt_hi[s_ld] + st["load_p"][ld] * lam)
                    conn_ok = _connected(lines_act, n_sub)
                    if not conn_ok or (require_connected and not conn_ok):
                        nom_ok = int_ok = mc_ok = False
                    else:
                        P_lam = P_base - (lam - 1.0) * load_by_sub
                        f = dc_flows(lines_act, n_sub, P_lam)
                        nom_ok, _ = authorize(f, f, thermal, margin=margin)
                        f_lo, f_hi = interval_flows_closedform(lines_act, n_sub, lo, hi)
                        int_ok, _ = authorize(f_lo, f_hi, thermal, margin=margin)
                        Pm = rng.uniform(lo, hi, size=(100, n_sub))
                        from src.utils.dc_interval import build_susceptance
                        B = build_susceptance(lines_act, n_sub)
                        keep = [s for s in range(n_sub) if s != 0]
                        B_red = np.delete(np.delete(B, 0, axis=0), 0, axis=1)
                        theta_red = np.linalg.solve(B_red, Pm[:, keep].T).T
                        flows = np.zeros((100, len(lines_act)))
                        for li, l in enumerate(lines_act):
                            if not np.isfinite(l["x"]):
                                continue
                            th = np.zeros((100, n_sub))
                            th[:, keep] = theta_red
                            flows[:, li] = (th[:, l["or"]] - th[:, l["ex"]]) / l["x"]
                        safe = np.max(np.abs(flows) / thermal, axis=1) <= (1.0 - margin)
                        if prob_threshold is not None:
                            mc_ok = bool(safe.sum() >= prob_threshold * 100)
                        else:
                            mc_ok = bool(safe.all())
                        if theta_cap is not None:
                            th_lo, th_hi = theta_interval_closedform(lines_act, n_sub, lo, hi)
                            theta_ok = bool(np.max(np.maximum(np.abs(th_lo), np.abs(th_hi)))
                                            <= theta_cap)
                            nom_ok = nom_ok and theta_ok
                            int_ok = int_ok and theta_ok
                            mc_ok = mc_ok and theta_ok
                    # 精确真值（gt 盒上的闭式角点；gaussian 变体用采样）
                    # 2026-09-17 修正：真值固定用 gt_lines_act（真实模型），
                    # 与验证器的 lines_act 解耦（旧协议用同一模型使变体真值失真）。
                    if not conn_ok:
                        danger = True
                    elif gt_dist == "gaussian":
                        f_lo, f_hi = interval_flows_closedform(gt_lines_act, n_sub, gt_lo, gt_hi)
                        caps = thermal * (1.0 - gt_margin)
                        danger = bool(np.any(np.maximum(np.abs(f_lo), np.abs(f_hi)) > caps))
                        # gaussian 无界 → 采样判定（保持原协议，标注）
                        rng2 = np.random.RandomState(20260916 + n_cases)
                        mid = (np.asarray(gt_lo) + np.asarray(gt_hi)) / 2
                        half = (np.asarray(gt_hi) - np.asarray(gt_lo)) / 2
                        danger = False
                        for _ in range(1000):
                            Pv = rng2.normal(mid, half / 2)
                            fv = dc_flows(gt_lines_act, n_sub, Pv)
                            ok, _ = authorize(fv, fv, thermal, margin=gt_margin)
                            if not ok:
                                danger = True
                                break
                    else:
                        f_lo, f_hi = interval_flows_closedform(gt_lines_act, n_sub, gt_lo, gt_hi)
                        caps = thermal * (1.0 - gt_margin)
                        danger = bool(np.any(np.maximum(np.abs(f_lo), np.abs(f_hi)) > caps))
                    decisions = {"none": True, "nominal": nom_ok,
                                 "interval": int_ok, "mc": mc_ok}
                    for kk, passed in decisions.items():
                        if passed:
                            acc[kk]["passed"] += 1
                            acc[kk]["passed_any"] += int(danger)
                        else:
                            acc[kk]["rejected"] += 1
                            acc[kk]["rejected_safe"] += int(not danger)
    out = {}
    for kk, a in acc.items():
        out[kk] = {
            "released_frac": round(a["passed"] / max(n_cases, 1), 4),
            "false_pass_exact": round(a["passed_any"] / max(a["passed"], 1), 4),
            "false_reject_exact": round(a["rejected_safe"] / max(a["rejected"], 1), 4),
        }
    out["n_cases"] = n_cases
    return out


def main():
    t0 = time.time()
    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    n_sub = int(env.env_glop.n_sub)
    thermal = env.env_glop.get_thermal_limit().astype(float)
    lines = pickle.load(open("data/processed/auth_lines_rte_case14_realistic.pkl", "rb"))
    states = pickle.load(open("data/processed/auth_states_rte_case14_realistic.pkl", "rb"))
    states = states[:100]

    # 变体线路模型
    uniform_lines = [dict(l, x=0.1) for l in lines]
    from experiments.exp_011_interval_auth.spike_interval_auth import (
        estimate_lines_from_telemetry,
    )
    t30_lines, _ = estimate_lines_from_telemetry(env, n_states=30)
    llm_props = {p["state_id"]: p["verifier_aware"]
                 for p in json.load(open("data/processed/llm_proposals_100.json",
                                         encoding="utf-8"))["proposals"]}

    results = {}
    # true-model verifier（A4 基线）：官方 pandapower 电抗（LSGrid x_pu/Sbase，
    # 与遥测辨识的 rad/MW 单位一致）
    n_line = len(lines)
    g_lsg = env.env_glop.backend._grid
    obs0 = env.env_glop.get_obs()
    or_sub = np.asarray(obs0.line_or_to_subid, dtype=int)
    ex_sub = np.asarray(obs0.line_ex_to_subid, dtype=int)
    pairs = {(int(or_sub[i]), int(ex_sub[i])): i for i in range(n_line)}
    gj_lines = [None] * n_line
    for pl in list(g_lsg.get_lines()):
        pair = (int(pl.sub1_id), int(pl.sub2_id))
        for cand in (pair, (pair[1], pair[0])):
            if cand in pairs:
                gj_lines[pairs[cand]] = {"or": cand[0], "ex": cand[1],
                                         "x": float(pl.x_pu / 100.0)}
    for tr in list(g_lsg.get_trafos()):
        pair = (int(tr.sub1_id), int(tr.sub2_id))
        for cand in (pair, (pair[1], pair[0])):
            if cand in pairs:
                gj_lines[pairs[cand]] = {"or": cand[0], "ex": cand[1],
                                         "x": float(tr.x_pu / 100.0)}
    assert all(l is not None for l in gj_lines)

    configs = [
        ("baseline", dict()),
        ("eps=0.05", dict(load_eps=0.05)),
        ("eps=0.10", dict(load_eps=0.10)),
        ("+connectivity", dict(require_connected=True)),
        ("+connectivity+theta", dict(require_connected=True, theta_cap=0.3)),
        # 2026-09-17 修正：模型变体配置的真值固定用真实（遥测全量）模型
        ("uniform-x", dict(lines_variant=uniform_lines, gt_lines=lines)),
        ("telemetry-30", dict(lines_variant=t30_lines, gt_lines=lines)),
        ("true-model", dict(lines_variant=gj_lines, gt_lines=lines)),
        ("random-proposals", dict(proposal_source="random")),
        ("llm-proposals", dict(proposal_source="llm", llm_props=llm_props)),
        ("margin=0", dict(margin=0.0)),
        ("margin=0.02", dict(margin=0.02)),
        ("prob99", dict(prob_threshold=0.99)),
        ("prob95", dict(prob_threshold=0.95)),
        ("rob-eps=0.00", dict(load_eps=0.0)),
        ("rob-eps=0.05", dict(load_eps=0.05)),
        ("rob-eps=0.10", dict(load_eps=0.10)),
        ("rob-eps=0.20", dict(load_eps=0.20)),
        ("rob-eps=0.40", dict(load_eps=0.40)),
        ("rob-gaussian", dict(gt_dist="gaussian")),
        ("rob-missing20", dict(missing_frac=0.2)),
    ]
    for name, kw in configs:
        lv = kw.pop("lines_variant", lines)
        results[name] = run_config(name, states, lv, thermal, n_sub, **kw)
        s = results[name]
        print(f"{name}: cases={s['n_cases']} "
              f"int_FP={s['interval']['false_pass_exact']} "
              f"int_FR={s['interval']['false_reject_exact']} "
              f"int_rel={s['interval']['released_frac']} "
              f"nom_FP={s['nominal']['false_pass_exact']} "
              f"mc_FP={s['mc']['false_pass_exact']} ({time.time()-t0:.0f}s)", flush=True)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "abl_rob_exact.json")
    with open(path, "w") as f:
        json.dump({"protocol": "100 states x 5 seeds x 5 candidates x 3 load levels "
                               "= 7,500 cases/config; exact box ground truth "
                               "(eps=0.2, m=0.05); gaussian variant sampled",
                   "configs": results}, f, indent=2)
    print(f"结果已保存: {path}")


if __name__ == "__main__":
    main()
