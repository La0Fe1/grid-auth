"""统计退化辨识模型与 clean 模型在 7,500 压力用例上的决策翻转数。

复用 run_telemetry_missing 的采集/辨识/评估逻辑，输出每模型 vs clean 的
翻转用例数（释放/拒绝方向分别计数），供论文如实报告。
"""
import json
import os
import pickle

import numpy as np

from src.utils.dc_interval import injection_bounds, interval_flows_closedform
from src.utils.env import Grid2OpGraphEnv
from experiments.exp_019_review_ext.run_telemetry_missing import (
    collect_telemetry, identify,
)

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


def main():
    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    n_sub = int(env.env_glop.n_sub)
    n_line = env.n_line
    thermal = env.env_glop.get_thermal_limit().astype(float)
    caps = thermal * (1.0 - MARGIN)
    true_lines = pickle.load(open("data/processed/auth_lines_rte_case14_realistic.pkl", "rb"))
    states = pickle.load(open("data/processed/auth_states_rte_case14_realistic.pkl", "rb"))
    seeds = [0, 1, 2, 3, 4]

    patterns = {
        "clean": dict(), "drop50": dict(drop_p=0.5), "drop80": dict(drop_p=0.8),
        "bias05": dict(bias_sigma=0.05), "bias10": dict(bias_sigma=0.10),
        "lag1": dict(lag=1), "nopmu30": dict(drop_branch_frac=0.3),
    }
    rng_id = np.random.RandomState(20260916)
    env.env_glop.reset()
    obs_proto = env.env_glop.get_obs()
    models = {}
    for name, kw in patterns.items():
        dtp, pp = collect_telemetry(env, 60, rng_id, **kw)
        lines, n_fb = identify(dtp, pp, n_line, n_sub, true_lines, obs_proto)
        models[name] = lines

    decisions = {name: [] for name in models}
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
                if not _connected([dict(true_lines[lid], x=np.inf)
                                   if l == lid else true_lines[l]
                                   for l in range(n_line)], n_sub):
                    continue
                for lam in LAMDAS:
                    lo, hi = injection_bounds(st["gen_p"], st["load_p"] * lam,
                                              gen_to_sub, load_to_sub, n_sub,
                                              load_eps=LOAD_EPS)
                    for name, mlines in models.items():
                        lines_act = [dict(l) for l in mlines]
                        lines_act[lid] = dict(mlines[lid], x=np.inf)
                        if not _connected(lines_act, n_sub):
                            ok = False
                        else:
                            v_lo, v_hi = interval_flows_closedform(lines_act, n_sub, lo, hi)
                            worst = np.maximum(np.abs(v_lo), np.abs(v_hi))
                            ok = bool(np.all(worst <= caps) and np.all(np.isfinite(worst)))
                        decisions[name].append(ok)

    out = {"n_cases": len(decisions["clean"])}
    for name in models:
        if name == "clean":
            continue
        d_clean = np.array(decisions["clean"])
        d_name = np.array(decisions[name])
        flips = int((d_clean != d_name).sum())
        clean_rel = int((d_clean & ~d_name).sum())
        name_rel = int((d_name & ~d_clean).sum())
        out[name] = {"flipped_cases": flips,
                     "clean_releases_name_rejects": clean_rel,
                     "name_releases_clean_rejects": name_rel}
    os.makedirs("experiments/exp_019_review_ext/results", exist_ok=True)
    path = "experiments/exp_019_review_ext/results/telemetry_flips.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
