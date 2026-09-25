"""exp_016 算力专项：四验证器的单动作延迟（各系统、100 次重复）。

用法：python -m src.evaluation.measure_latency [case] [--test-mode]
"""
import json
import os
import sys
import time

import numpy as np

from src.utils.dc_interval import (
    authorize, dc_flows, injection_bounds, interval_flows_closedform,
    interval_flows_lp,
)
from src.utils.env import Grid2OpGraphEnv


def main():
    import pickle
    case = sys.argv[1] if len(sys.argv) > 1 else "rte_case14_realistic"
    test_mode = "--test-mode" in sys.argv
    env = Grid2OpGraphEnv(case=case, test=test_mode)
    cache = f"data/processed/auth_lines_{case.replace('l2rpn_', '')}.pkl"
    if os.path.exists(cache):
        lines = pickle.load(open(cache, "rb"))
        n_sub = int(env.env_glop.n_sub)
    else:
        from experiments.exp_011_interval_auth.spike_interval_auth import (
            estimate_lines_from_telemetry,
        )
        lines, n_sub = estimate_lines_from_telemetry(env, n_states=60)
    thermal = env.env_glop.get_thermal_limit().astype(float)

    rng = np.random.RandomState(0)
    env.env_glop.set_id(0)
    env.env_glop.reset()
    env.env_glop.fast_forward_chronics(100)
    obs = env.env_glop.get_obs()
    gen_to_sub, load_to_sub = obs.gen_to_subid, obs.load_to_subid
    lo, hi = injection_bounds(obs.gen_p, obs.load_p * 2.0, gen_to_sub, load_to_sub,
                              n_sub, load_eps=0.2)
    P = np.zeros(n_sub)
    for g, s in enumerate(gen_to_sub):
        P[s] += obs.gen_p[g]
    for ld, s in enumerate(load_to_sub):
        P[s] -= obs.load_p[ld] * 2.0
    lines_act = [dict(l) for l in lines]
    lines_act[5] = dict(lines[5], x=np.inf)

    n_rep = 100
    res = {}
    t = time.perf_counter()
    for _ in range(n_rep):
        f = dc_flows(lines_act, n_sub, P)
        authorize(f, f, thermal, margin=0.05)
    res["nominal_ms"] = (time.perf_counter() - t) / n_rep * 1e3

    t = time.perf_counter()
    for _ in range(n_rep):
        f_lo, f_hi = interval_flows_lp(lines_act, n_sub, lo, hi)
        authorize(f_lo, f_hi, thermal, margin=0.05)
    res["interval_ms"] = (time.perf_counter() - t) / n_rep * 1e3

    # 闭式角点公式（与 LP 等价，见 src/utils/dc_interval.py 与单元测试）
    t = time.perf_counter()
    for _ in range(n_rep):
        f_lo, f_hi = interval_flows_closedform(lines_act, n_sub, lo, hi)
        authorize(f_lo, f_hi, thermal, margin=0.05)
    res["interval_cf_ms"] = (time.perf_counter() - t) / n_rep * 1e3

    t = time.perf_counter()
    for _ in range(n_rep):
        for _ in range(100):
            Ps = rng.uniform(lo, hi)
            f = dc_flows(lines_act, n_sub, Ps)
            authorize(f, f, thermal, margin=0.05)
    res["mc100_ms"] = (time.perf_counter() - t) / n_rep * 1e3

    out = f"experiments/exp_016/results/latency_{case.replace('l2rpn_', '')}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"case": case, "n_repetitions": n_rep, **res}, f, indent=2)
    print(case, res)


if __name__ == "__main__":
    main()
