"""exp_016 补测：DC 区间对 AC 最坏点的覆盖（审稿意见 CRITICAL #3 / AQ4）。

方法：
- AC 潮流用 lightsim2grid.TimeSerie：对每个拓扑（完整网 / 断开某条线路的
  动作后网）构造一个 TimeSerie，再对每个探针负荷向量调用
  compute_V_from_inj(gen_p, load_p, load_q) + compute_P() 得到 20 条
  grid2op 线路的有功潮流（AC Newton）。
- 探针 = 负荷箱内 200 个均匀点 + 每支路 2 个 DC 极值角点；
- 指标：AC 支路潮流落在 DC 区间 [f_ℓ⁻, f_ℓ⁺] 内的比例（覆盖/漏包）、
  漏包方向与幅度（MW）、DC 判定"安全"的用例在 DC 最坏角点上的 AC 越限率。

用法：python -m experiments.exp_016.dc_ac_containment
输出：experiments/exp_016/results/dc_ac_containment.json
"""
import json
import os
import pickle
import time
import warnings

import numpy as np

from lightsim2grid import TimeSerie

from src.utils.dc_interval import (
    build_susceptance, injection_bounds, interval_flows_closedform,
)
from src.utils.env import Grid2OpGraphEnv


def _match_elements(env, g):
    """grid2op 线路 id → (AC 元件类型, 元件序号)：15 powerline + 5 线化 trafo，
    按 (or,ex) 变电站对匹配。"""
    obs0 = env.env_glop.get_obs()
    or_sub = np.asarray(obs0.line_or_to_subid, dtype=int)
    ex_sub = np.asarray(obs0.line_ex_to_subid, dtype=int)
    pairs = {(int(or_sub[i]), int(ex_sub[i])): i for i in range(env.n_line)}
    lines_el = [None] * env.n_line
    for j, pl in enumerate(list(g.get_lines())):
        pair = (int(pl.sub1_id), int(pl.sub2_id))
        for cand in (pair, (pair[1], pair[0])):
            if cand in pairs:
                lines_el[pairs[cand]] = ("pl", j)
    for k, tr in enumerate(list(g.get_trafos())):
        pair = (int(tr.sub1_id), int(tr.sub2_id))
        for cand in (pair, (pair[1], pair[0])):
            if cand in pairs:
                lines_el[pairs[cand]] = ("tr", k)
    assert all(x is not None for x in lines_el), lines_el
    return lines_el, None


def _sample_states(env, n_states, rng):
    """按与主实验相同的采样协议采状态（含 q；只保留全线路连通的状态）。"""
    states = []
    n_chron = max(int(env._n_chronics), 1)
    while len(states) < n_states:
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
        if not bool(np.all(obs.line_status == 1)):
            continue  # 只保留全连通状态（与 DC 模型的全线连通假设一致）
        states.append({
            "gen_p": obs.gen_p.astype(np.float64).copy(),
            "load_p": obs.load_p.astype(np.float64).copy(),
            "load_q": obs.load_q.astype(np.float64).copy(),
            "gen_to_sub": obs.gen_to_subid.astype(int).copy(),
            "load_to_sub": obs.load_to_subid.astype(int).copy(),
            "rho": obs.rho.astype(np.float64).copy(),
        })
    return states


def main():
    t0 = time.time()
    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    n_sub = int(env.env_glop.n_sub)
    thermal = env.env_glop.get_thermal_limit().astype(float)
    lines = pickle.load(open("data/processed/auth_lines_rte_case14_realistic.pkl", "rb"))
    n_line = len(lines)
    rng = np.random.RandomState(20260916)

    env.env_glop.reset()
    ts_intact = TimeSerie(env.env_glop)  # 完整网拓扑

    # 校验：基态注入的 AC PF vs obs.p_or（完整网）
    obs0 = env.env_glop.get_obs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts_intact.compute_V_from_inj(obs0.gen_p.reshape(1, -1),
                                     obs0.load_p.reshape(1, -1),
                                     obs0.load_q.reshape(1, -1))
    p0 = np.asarray(ts_intact.compute_P(), dtype=float).reshape(-1)
    print(f"[校验] 完整网基态 AC PF vs obs.p_or: 最大差 "
          f"{float(np.max(np.abs(p0 - obs0.p_or))):.3f} MW", flush=True)

    # 动作拓扑：每个候选动作断开一条线路 → 对应 TimeSerie
    # （直接改 LSGrid 拓扑 + TimeSerie 快照，不推进时序）
    g = env.env_glop.backend._grid
    lines_el, _ = _match_elements(env, g)

    def _set_line(g, lid, active):
        kind, idx = lines_el[lid]
        if kind == "pl":
            (g.reactivate_powerline if active else g.deactivate_powerline)(idx)
        else:
            (g.reactivate_trafo if active else g.deactivate_trafo)(idx)

    ts_action = {}
    for lid in range(n_line):
        env.env_glop.reset()
        _set_line(g, lid, False)
        ts_action[lid] = TimeSerie(env.env_glop)
        _set_line(g, lid, True)  # 恢复，避免影响后续拓扑

    states = _sample_states(env, 20, rng)
    lamdas = [2.0, 3.0, 4.0]
    n_probes = 200
    margin = 0.05
    load_eps = 0.2

    n_ok = n_total = 0
    max_miss_lo = max_miss_hi = 0.0
    miss_by_branch = {li: 0 for li in range(n_line)}
    branch_total = {li: 0 for li in range(n_line)}
    ac_danger_when_dc_safe = 0
    ac_checked = 0
    rej_checked = 0
    rej_ac_confirmed = 0
    n_fail = 0
    by_lam = {lam: {"ok": 0, "total": 0} for lam in lamdas}
    corner_ok = corner_total = 0
    corner_by_lam = {lam: {"ok": 0, "total": 0} for lam in lamdas}
    danger_by_lam = {lam: {"danger": 0, "checked": 0} for lam in lamdas}

    for si, st in enumerate(states):
        gen_to_sub, load_to_sub = st["gen_to_sub"], st["load_to_sub"]
        load_by_sub = np.zeros(n_sub)
        for ld, s in enumerate(load_to_sub):
            load_by_sub[s] += st["load_p"][ld]
        P_base = np.zeros(n_sub)
        for gg, s in enumerate(gen_to_sub):
            P_base[s] += st["gen_p"][gg]
        P_base -= load_by_sub
        for lid in range(n_line):
            lines_act = [dict(l) for l in lines]
            lines_act[lid] = dict(lines[lid], x=np.inf)
            if not _connected(lines_act, n_sub):
                continue  # 孤岛动作被授权层直接拒绝，无需探针
            ts = ts_action[lid]
            for lam in lamdas:
                lo, hi = injection_bounds(st["gen_p"], st["load_p"] * lam,
                                          gen_to_sub, load_to_sub, n_sub,
                                          load_eps=load_eps)
                load_lam = load_by_sub * lam
                f_lo, f_hi = interval_flows_closedform(lines_act, n_sub, lo, hi)
                caps = thermal * (1 - margin)
                dc_safe = bool(np.max(np.maximum(np.abs(f_lo), np.abs(f_hi)) /
                                      caps) <= 1.0)

                def probe(Pv):
                    scale = np.ones(n_sub)
                    for s in range(n_sub):
                        sampled_load = float(np.sum(st["gen_p"][gen_to_sub == s])) - Pv[s]
                        scale[s] = sampled_load / load_lam[s] if load_lam[s] > 1e-9 else 1.0
                    rows_p = np.array([st["load_p"][ld] * lam *
                                       scale[int(load_to_sub[ld])]
                                       for ld in range(len(load_to_sub))])
                    rows_q = st["load_q"] * np.array(
                        [scale[int(load_to_sub[ld])]
                         for ld in range(len(load_to_sub))])
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            ts.compute_V_from_inj(
                                st["gen_p"].reshape(1, -1),
                                rows_p.reshape(1, -1),
                                rows_q.reshape(1, -1))
                        return np.asarray(ts.compute_P(), dtype=float).reshape(-1)
                    except RuntimeError:
                        return None

                uniform_probes = rng.uniform(lo, hi, size=(n_probes, n_sub))
                corner_probes = []
                for li in range(n_line):
                    if not np.isfinite(lines_act[li]["x"]):
                        continue
                    corner_probes.append(np.where(_sens_gt0(lines_act, n_sub, li), hi, lo))
                    corner_probes.append(np.where(_sens_gt0(lines_act, n_sub, li), lo, hi))
                for is_corner, probes in ((False, uniform_probes), (True, corner_probes)):
                    for Pv in probes:
                        ac = probe(np.asarray(Pv))
                        if ac is None:
                            n_fail += 1
                            continue
                        for li in range(n_line):
                            if not np.isfinite(lines_act[li]["x"]):
                                continue
                            branch_total[li] += 1
                            n_total += 1
                            by_lam[lam]["total"] += 1
                            if is_corner:
                                corner_total += 1
                                corner_by_lam[lam]["total"] += 1
                            fv = ac[li]
                            if f_lo[li] - 1e-9 <= fv <= f_hi[li] + 1e-9:
                                n_ok += 1
                                by_lam[lam]["ok"] += 1
                                if is_corner:
                                    corner_ok += 1
                                    corner_by_lam[lam]["ok"] += 1
                            else:
                                miss_by_branch[li] += 1
                                if fv < f_lo[li]:
                                    max_miss_lo = max(max_miss_lo, f_lo[li] - fv)
                                else:
                                    max_miss_hi = max(max_miss_hi, fv - f_hi[li])
                # AC 级复核：DC-safe 用例在 DC 最坏角点上的 AC 越限
                if dc_safe:
                    ac_checked += 1
                    danger_by_lam[lam]["checked"] += 1
                    violated = False
                    for li in range(n_line):
                        if not np.isfinite(lines_act[li]["x"]):
                            continue
                        for corner in (np.where(_sens_gt0(lines_act, n_sub, li), hi, lo),
                                       np.where(_sens_gt0(lines_act, n_sub, li), lo, hi)):
                            ac = probe(corner)
                            if ac is not None and np.any(np.abs(ac) > caps):
                                ac_danger_when_dc_safe += 1
                                danger_by_lam[lam]["danger"] += 1
                                violated = True
                                break
                        if violated:
                            break
                else:
                    # 拒绝侧 AC 复核：见证角点上 AC 是否确认越限（审稿 Q3）
                    rej_checked += 1
                    confirmed = False
                    for li in range(n_line):
                        if not np.isfinite(lines_act[li]["x"]):
                            continue
                        for corner in (np.where(_sens_gt0(lines_act, n_sub, li), hi, lo),
                                       np.where(_sens_gt0(lines_act, n_sub, li), lo, hi)):
                            ac = probe(corner)
                            if ac is not None and np.any(np.abs(ac) > caps):
                                rej_ac_confirmed += 1
                                confirmed = True
                                break
                        if confirmed:
                            break
        print(f"  state {si+1}/20, {time.time()-t0:.0f}s", flush=True)

    result = {
        "case": "rte_case14_realistic",
        "protocol": "20 states x 20 actions x 3 load levels x "
                    "(200 uniform + per-branch DC-corner probes), AC via lightsim2grid TimeSerie",
        "containment_rate": float(n_ok / max(n_total, 1)),
        "n_probe_points": n_total,
        "max_miss_below_lo_mw": float(max_miss_lo),
        "max_miss_above_hi_mw": float(max_miss_hi),
        "ac_pf_failures": n_fail,
        "ac_danger_when_dc_safe": ac_danger_when_dc_safe,
        "dc_safe_cases_checked": ac_checked,
        "dc_rejected_cases_checked": rej_checked,
        "ac_confirms_rejection": rej_ac_confirmed,
        "ac_rejection_confirmation_rate": round(
            rej_ac_confirmed / max(rej_checked, 1), 4),
        "containment_by_lam": {str(k): round(v["ok"] / max(v["total"], 1), 4)
                               for k, v in by_lam.items()},
        "corner_containment": float(corner_ok / max(corner_total, 1)),
        "corner_containment_by_lam": {str(k): round(v["ok"] / max(v["total"], 1), 4)
                                      for k, v in corner_by_lam.items()},
        "ac_danger_rate_by_lam": {str(k): round(v["danger"] / max(v["checked"], 1), 4)
                                  for k, v in danger_by_lam.items()},
        "worst_branch_miss_rates_top5": sorted(
            ((miss_by_branch[li] / max(branch_total[li], 1), li)
             for li in range(n_line)), reverse=True)[:5],
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "dc_ac_containment.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"结果已保存: {path}")


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


def _sens_gt0(lines_act, n_sub, li, slack=0):
    B = build_susceptance(lines_act, n_sub)
    keep = [s for s in range(n_sub) if s != slack]
    idx_of = {s: i for i, s in enumerate(keep)}
    B_red = np.delete(np.delete(B, slack, axis=0), slack, axis=1)
    B_inv = np.linalg.inv(B_red)
    c = np.zeros(n_sub)
    l = lines_act[li]
    for s in keep:
        t_or = B_inv[idx_of[s], idx_of[l["or"]]] if l["or"] != slack else 0.0
        t_ex = B_inv[idx_of[s], idx_of[l["ex"]]] if l["ex"] != slack else 0.0
        c[s] = (t_or - t_ex) / l["x"]
    return c >= 0


if __name__ == "__main__":
    main()
