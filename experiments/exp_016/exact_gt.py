"""exp_016 补测：精确真值标签（审稿意见 Q1）。

审稿人指出：真值"危险"用 M=5000 采样判定，而方法本身可精确计算盒内最坏情况。
本脚本用闭式角点公式对真值盒（load_eps=0.2, margin=0.05）计算精确判定：
  dangerous = 真值盒内存在支路 |f| > (1−m)×rating 的角点
（孤岛动作按协议视为危险）。重算全部验证器的精确误放/误拒率与
真值条件 FPR/FNR，并保留采样口径的 mean danger rate 作对照。

覆盖：主实验 12,500 用例（full cases 带 state 字段）、跨系统 icaps/idf
各 1,500、LLM 1,150（由 llm_proposals_100.json 重放）。

用法：python -m experiments.exp_016.exact_gt
输出：experiments/exp_016/results/exact_gt.json
"""
import json
import os
import pickle

import numpy as np

from src.utils.dc_interval import (
    injection_bounds, interval_flows_closedform,
)


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


def exact_dangerous(lines_act, n_sub, st, lam, thermal, margin=0.05, load_eps=0.2):
    """真值盒上的精确危险判定（闭式角点公式，非采样）。"""
    if not _connected(lines_act, n_sub):
        return True
    lo, hi = injection_bounds(st["gen_p"], st["load_p"] * lam,
                              st["gen_to_sub"], st["load_to_sub"], n_sub,
                              load_eps=load_eps)
    f_lo, f_hi = interval_flows_closedform(lines_act, n_sub, lo, hi)
    caps = thermal * (1.0 - margin)
    return bool(np.any(np.maximum(np.abs(f_lo), np.abs(f_hi)) > caps))


def summarize(records, ok_key):
    out = {}
    for k, kk in ok_key.items():
        rel_d = rel_s = rej_d = rej_s = 0
        for r in records:
            danger = r["exact_dangerous"]
            if kk is None or r[kk]:
                if danger:
                    rel_d += 1
                else:
                    rel_s += 1
            else:
                if danger:
                    rej_d += 1
                else:
                    rej_s += 1
        tot_d, tot_s = rel_d + rej_d, rel_s + rej_s
        out[k] = {
            "false_pass_exact": round(rel_d / max(rel_d + rel_s, 1), 4),
            "false_reject_exact": round(rej_s / max(rej_d + rej_s, 1), 4),
            "FPR_exact": round(rel_d / max(tot_d, 1), 4),
            "FNR_exact": round(rej_s / max(tot_s, 1), 4),
            "counts": [rel_d, rel_s, rej_d, rej_s],
        }
    return out


def main():
    thermal = None
    results = {}

    # ---- 主实验 12,500 用例 ----
    case = "rte_case14_realistic"
    lines = pickle.load(open("data/processed/auth_lines_rte_case14_realistic.pkl", "rb"))
    states = pickle.load(open("data/processed/auth_states_rte_case14_realistic.pkl", "rb"))
    from src.utils.env import Grid2OpGraphEnv
    env = Grid2OpGraphEnv(case=case, test=False)
    thermal = env.env_glop.get_thermal_limit().astype(float)
    n_sub = int(env.env_glop.n_sub)
    recs = [json.loads(l) for l in open(
        "experiments/exp_012/results/auth_eval_rte_case14_realistic_full_cases.jsonl",
        encoding="utf-8")]
    for r in recs:
        st = states[r["state"]]
        lines_act = [dict(l) for l in lines]
        lines_act[r["disconnects"][0]] = dict(lines[r["disconnects"][0]], x=np.inf)
        r["exact_dangerous"] = exact_dangerous(lines_act, n_sub, st, r["lambda"],
                                               thermal)
    results["14bus_12500"] = summarize(recs, {"none": None, "nominal": "nominal_ok",
                                              "interval": "interval_ok", "mc": "mc_ok"})
    print("14bus:", results["14bus_12500"], flush=True)

    # ---- 跨系统 ----
    for key, cache, casefile in [
        ("36bus_1500", "icaps_2021", "experiments/exp_015/results/auth_eval_icaps_2021_full_cases.jsonl"),
        ("118bus_1500", "idf_2023", "experiments/exp_015/results/auth_eval_idf_2023_full_cases.jsonl"),
    ]:
        env2 = Grid2OpGraphEnv(case="l2rpn_" + ("icaps_2021" if "36" in key else "idf_2023"),
                               test=True)
        lines2 = pickle.load(open(f"data/processed/auth_lines_{cache}.pkl", "rb"))
        states2 = pickle.load(open(f"data/processed/auth_states_{cache}.pkl", "rb"))
        thermal2 = env2.env_glop.get_thermal_limit().astype(float)
        n_sub2 = int(env2.env_glop.n_sub)
        recs2 = [json.loads(l) for l in open(casefile, encoding="utf-8")]
        for r in recs2:
            st = states2[r["state"]]
            lines_act = [dict(l) for l in lines2]
            lines_act[r["disconnects"][0]] = dict(lines2[r["disconnects"][0]], x=np.inf)
            r["exact_dangerous"] = exact_dangerous(lines_act, n_sub2, st, r["lambda"],
                                                   thermal2)
        results[key] = summarize(recs2, {"nominal": "nominal_ok",
                                         "interval": "interval_ok", "mc": "mc_ok"})
        print(key, ":", results[key], flush=True)

    # ---- LLM 1,150 用例（重放：state → 提案动作）----
    prop = json.load(open("data/processed/llm_proposals_100.json", encoding="utf-8"))
    props_by_state = {p["state_id"]: p["verifier_aware"] for p in prop["proposals"]}
    llm_recs = [json.loads(l) for l in open(
        "experiments/exp_017/results/llm_auth_eval_100_cases.jsonl", encoding="utf-8")]
    # LLM cases 文件无 state 字段 → 按 (seed, lambda, disconnects) 与提案重放对齐。
    # 提案按状态唯一（46 个活跃状态 × 动作列表）；构造 (seed, lambda, actions) → state 映射
    key2state = {}
    for sid, acts in props_by_state.items():
        if acts:
            for seed in range(5):
                for lam in [1.0, 2.0, 3.0, 4.0, 5.0]:
                    key2state[(seed, lam, tuple(sorted(acts)))] = sid
    n_matched = 0
    for r in llm_recs:
        k = (r["seed"], r["lambda"], tuple(sorted(r["disconnects"])))
        if k in key2state:
            st = states[key2state[k]]
            lines_act = [dict(l) for l in lines]
            for lid in r["disconnects"]:
                lines_act[lid] = dict(lines[lid], x=np.inf)
            r["exact_dangerous"] = exact_dangerous(lines_act, n_sub, st,
                                                   r["lambda"], thermal)
            n_matched += 1
        else:
            r["exact_dangerous"] = None
    ok = [r for r in llm_recs if r["exact_dangerous"] is not None]
    assert len(ok) == len(llm_recs), (len(ok), len(llm_recs))
    results["llm_1150"] = summarize(llm_recs, {"none": None, "nominal": "nominal_ok",
                                               "interval": "interval_ok", "mc": "mc_ok"})
    print("llm:", results["llm_1150"], flush=True)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "exact_gt.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"结果已保存: {path}")


if __name__ == "__main__":
    main()
