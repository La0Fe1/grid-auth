"""exp_023: 丰富可观测性 LLM 提案评估（评审 Q10 可观测性维度）。

解析 llm_rich_responses.jsonl（100 状态，llm-chat MCP / deepseek-chat 逐状态调用，
验证器感知系统提示 + 全部 20 条线路 rho/裕度/总量摘要），按原文 LLM 研究协议
评估：行动状态 × 5 种子 × 5 负荷水平 λ∈{1..5}，真值 = 真实模型精确角点判据，
验证器 none/nominal/MC(K=100)/interval。与最小摘要基线（46/100 行动、
60.00/16.36/0.65/0.00%）对比。
"""
import json
import os
import pickle

import numpy as np

from src.utils.dc_interval import (
    dc_flows, injection_bounds, interval_flows_closedform,
)

LAMDAS = [1.0, 2.0, 3.0, 4.0, 5.0]
LOAD_EPS = 0.2
MARGIN = 0.05
MC_K = 100


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


def main():
    true_lines = pickle.load(open("data/processed/auth_lines_rte_case14_realistic.pkl", "rb"))
    states = pickle.load(open("data/processed/auth_states_rte_case14_realistic.pkl", "rb"))
    states = states[:100]
    n_sub = len(true_lines)  # 占位，下面从 env 取
    from src.utils.env import Grid2OpGraphEnv
    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    n_sub = int(env.env_glop.n_sub)
    thermal = env.env_glop.get_thermal_limit().astype(float)
    caps = thermal * (1.0 - MARGIN)

    # 解析响应
    actions = {}
    with open("experiments/exp_019_review_ext/results/llm_rich_responses.jsonl",
              encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            try:
                obj = json.loads(d["raw"])
                acts = [int(a) for a in obj.get("actions", [])]
            except Exception:
                acts = []
            actions[d["state_id"]] = acts
    active = {i: a for i, a in actions.items() if a}
    n_active = len(active)
    print(f"n_active = {n_active}/100; actions = {json.dumps(active, ensure_ascii=False)}")

    # 评估协议（与原文 LLM 研究一致：5 种子 × 5 λ）
    seeds = [0, 1, 2, 3, 4]
    rng = np.random.RandomState(20260916)
    acc = {k: {"passed": 0, "danger_passed": 0, "rejected": 0, "safe_rejected": 0}
           for k in ("none", "nominal", "mc", "interval")}
    n_cases = 0
    for si, acts in sorted(active.items()):
        st = states[si]
        gen_to_sub, load_to_sub = st["gen_to_sub"], st["load_to_sub"]
        load_by_sub = np.zeros(n_sub)
        for ld, s in enumerate(load_to_sub):
            load_by_sub[s] += st["load_p"][ld]
        P_base = np.zeros(n_sub)
        for g, s in enumerate(gen_to_sub):
            P_base[s] += st["gen_p"][g]
        P_base -= load_by_sub
        for lid in acts:
            if not (0 <= lid < 20):
                continue
            if st["line_status"][lid] != 1:
                continue
            lines_act = [dict(l) for l in true_lines]
            lines_act[lid] = dict(true_lines[lid], x=np.inf)
            if not _connected(lines_act, n_sub):
                continue
            for seed in seeds:
                for lam in LAMDAS:
                    P_lam = P_base - (lam - 1.0) * load_by_sub
                    lo, hi = injection_bounds(st["gen_p"], st["load_p"] * lam,
                                              gen_to_sub, load_to_sub, n_sub,
                                              load_eps=LOAD_EPS)
                    # 真值（精确角点，真实模型）
                    t_lo, t_hi = interval_flows_closedform(lines_act, n_sub, lo, hi)
                    danger = bool(np.any(np.maximum(np.abs(t_lo), np.abs(t_hi)) > caps))
                    # nominal
                    f = dc_flows(lines_act, n_sub, P_lam)
                    nom_ok = bool(np.all(np.abs(f) <= caps))
                    # interval（匹配设计，FP 构造性 0）
                    v_lo, v_hi = interval_flows_closedform(lines_act, n_sub, lo, hi)
                    int_ok = bool(np.all(np.maximum(np.abs(v_lo), np.abs(v_hi)) <= caps))
                    # MC K=100
                    Pm = rng.uniform(lo, hi, size=(MC_K, n_sub))
                    from src.utils.dc_interval import build_susceptance
                    B = build_susceptance(lines_act, n_sub)
                    keep = [s for s in range(n_sub) if s != 0]
                    B_red = np.delete(np.delete(B, 0, axis=0), 0, axis=1)
                    th = np.linalg.solve(B_red, Pm[:, keep].T).T
                    theta = np.zeros((MC_K, n_sub))
                    theta[:, keep] = th
                    fm = np.empty((MC_K, len(lines_act)))
                    for li, l in enumerate(lines_act):
                        fm[:, li] = ((theta[:, l["or"]] - theta[:, l["ex"]]) / l["x"]
                                     if np.isfinite(l["x"]) else 0.0)
                    mc_ok = bool(np.all(np.abs(fm) <= caps[None, :]))
                    for kk, ok in (("none", True), ("nominal", nom_ok),
                                   ("mc", mc_ok), ("interval", int_ok)):
                        a = acc[kk]
                        if ok:
                            a["passed"] += 1
                            a["danger_passed"] += int(danger)
                        else:
                            a["rejected"] += 1
                            a["safe_rejected"] += int(not danger)
                    n_cases += 1

    out = {
        "protocol": "rich-observability variant (all 20 lines with rho/margin + totals + "
                    "±20% band), verifier-aware system prompt, deepseek-chat via llm-chat "
                    "MCP; eval: active states x 5 seeds x 5 load levels (1..5), exact "
                    "box-corner ground truth on the true model",
        "n_states": 100,
        "n_active": n_active,
        "active_states": active,
        "baseline_minimal_summary": {"n_active": 46, "none": 0.6000, "nominal": 0.1636,
                                     "mc": 0.0065, "interval": 0.0},
        "n_cases": n_cases,
        "summary": {kk: {
            "released_fraction": round(a["passed"] / max(n_cases, 1), 4),
            "false_pass_rate": round(a["danger_passed"] / max(a["passed"], 1), 4),
            "false_reject_rate": round(a["safe_rejected"] / max(a["rejected"], 1), 4),
        } for kk, a in acc.items()},
    }
    os.makedirs("experiments/exp_019_review_ext/results", exist_ok=True)
    path = "experiments/exp_019_review_ext/results/llm_rich_eval.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
