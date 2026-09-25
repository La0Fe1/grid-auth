"""exp_019: 两层部署门端到端评估（评审 Q6/W3）。

第 1 层 = 区间 DC 筛查（Algorithm 1）；第 2 层 = 压力触发时在每条活跃支路的
DC 最坏角点上做 AC 潮流复核（lightsim2grid TimeSerie），全部角点满足支路
额定值判据 (1−m) 才放行；任一角点 AC 越限即拒绝（防御纵深）。

协议：与 §3.6 相同的压力用例（20 状态 × 20 动作 × λ∈{2,3,4}，状态采样
RandomState(20260916)，与 dc_ac_containment 一致）。
触发阈值 τ：最坏情形载荷比 ≥ τ·(1−m) 时触发第 2 层（τ=0.9 主结果，
τ∈{0.5, 0.0} 灵敏度：0 = 所有 DC-safe 压力动作都进第 2 层）。
指标：门释放率、释放集在 DC 角点上的 AC 越限率（构造性 0）、释放集的
200 均匀探针 AC 越限率（端到端残差）、第 2 层每用例 AC 求解数、
第 2 层延迟（5×200 微基准 + 全程墙钟）。

用法：python -m experiments.exp_019_review_ext.run_twotier
输出：experiments/exp_019_review_ext/results/twotier.json
"""
import json
import os
import pickle
import statistics
import time
import warnings

import numpy as np
from lightsim2grid import TimeSerie

from src.utils.dc_interval import (
    build_susceptance, dc_flows, injection_bounds, interval_flows_closedform,
)
from src.utils.env import Grid2OpGraphEnv
from experiments.exp_016.dc_ac_containment import (
    _connected, _match_elements, _sample_states,
)

LOADS = [2.0, 3.0, 4.0]
TAUS = [0.9, 0.5, 0.0]
N_UNIFORM = 200
MARGIN = 0.05
LOAD_EPS = 0.2


def corner_matrix(lines_act, n_sub, lo, hi, slack=0):
    """每条活跃支路的两个 DC 最坏角点（闭式），去重后返回 (k, n_sub) 数组。"""
    B = build_susceptance(lines_act, n_sub)
    keep = [s for s in range(n_sub) if s != slack]
    idx_of = {s: i for i, s in enumerate(keep)}
    B_red = np.delete(np.delete(B, slack, axis=0), slack, axis=1)
    B_inv = np.linalg.inv(B_red)
    lo_v = np.asarray([lo[s] for s in keep], dtype=float)
    hi_v = np.asarray([hi[s] for s in keep], dtype=float)
    corners = []
    for li, l in enumerate(lines_act):
        if not np.isfinite(l["x"]):
            continue
        c = np.zeros(len(keep))
        if l["or"] != slack:
            c += B_inv[idx_of[l["or"]], :]
        if l["ex"] != slack:
            c -= B_inv[idx_of[l["ex"]], :]
        c /= l["x"]
        P_lo = np.zeros(n_sub)
        P_hi = np.zeros(n_sub)
        P_lo[keep] = np.where(c >= 0, lo_v, hi_v)
        P_hi[keep] = np.where(c >= 0, hi_v, lo_v)
        # 平衡节点注入取平衡隐含值（DC 角点语义：松弛节点隐式平衡）
        P_lo[slack] = -float(np.sum(P_lo[keep]))
        P_hi[slack] = -float(np.sum(P_hi[keep]))
        corners.append(P_lo)
        corners.append(P_hi)
    if not corners:
        return np.zeros((0, n_sub))
    corners = np.unique(np.round(np.array(corners), 9), axis=0)
    return corners


def make_probe(ts, st, lam, n_sub, load_lam, gen_to_sub, load_to_sub):
    """AC 探针（返回 (n, n_line) 支路潮流；None 表示该探针 AC 无解）。

    逐行调用 compute_V_from_inj：lightsim2grid 的多行批量调用在同一 TimeSerie
    上不稳定（批量后会污染后续求解，实测 200 行批量整体失败而逐行全收敛），
    单行调用与 dc_ac_containment 的原始协议一致且稳健。
    """
    gen_at_s = np.zeros(n_sub)
    for g, s in enumerate(gen_to_sub):
        gen_at_s[s] += st["gen_p"][g]
    idx_ld = np.array(load_to_sub, dtype=int)
    gen_p_1 = st["gen_p"].reshape(1, -1)

    def probe(Pv_mat):
        Pv_mat = np.asarray(Pv_mat, dtype=float)
        if Pv_mat.ndim == 1:
            Pv_mat = Pv_mat.reshape(1, -1)
        n_ld = len(idx_ld)
        rows = []
        for i in range(Pv_mat.shape[0]):
            Pv = Pv_mat[i]
            sl = gen_at_s - Pv
            scale = np.where(load_lam > 1e-9, sl / np.maximum(load_lam, 1e-9), 1.0)
            rows_p = st["load_p"] * lam * scale[idx_ld]
            rows_q = st["load_q"] * scale[idx_ld]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ts.compute_V_from_inj(gen_p_1, rows_p.reshape(1, n_ld),
                                          rows_q.reshape(1, n_ld))
                rows.append(np.asarray(ts.compute_P(), dtype=float).reshape(-1))
            except RuntimeError:
                return None
        return np.vstack(rows)

    return probe


def main():
    t0 = time.time()
    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    n_sub = int(env.env_glop.n_sub)
    n_line = env.n_line
    thermal = env.env_glop.get_thermal_limit().astype(float)
    lines = pickle.load(open("data/processed/auth_lines_rte_case14_realistic.pkl", "rb"))
    caps = thermal * (1.0 - MARGIN)

    rng = np.random.RandomState(20260916)
    states = _sample_states(env, 20, rng)  # 与 §3.6 同一采样协议

    # 每个断线动作的 TimeSerie（与 dc_ac_containment 相同的拓扑操作，
    # 含完整网 TimeSerie 的基态校验——该调用使后续单行求解与原始协议一致）
    env.env_glop.reset()
    ts_intact = TimeSerie(env.env_glop)
    obs0 = env.env_glop.get_obs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts_intact.compute_V_from_inj(obs0.gen_p.reshape(1, -1),
                                     obs0.load_p.reshape(1, -1),
                                     obs0.load_q.reshape(1, -1))
    p0 = np.asarray(ts_intact.compute_P(), dtype=float).reshape(-1)
    print(f"[校验] 完整网基态 AC PF vs obs.p_or: 最大差 "
          f"{float(np.max(np.abs(p0 - obs0.p_or))):.3f} MW", flush=True)
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
        _set_line(g, lid, True)

    agg = {tau: {
        "dc_safe": 0, "triggered": 0, "released": 0,
        "rejected_corner": 0, "rejected_unresolved": 0,
        "corner_solves": 0, "tier2_time_s": 0.0,
        "released_uniform_viol": 0.0, "released_uniform_resolved": 0,
        "dcsafe_uniform_viol": 0.0, "dcsafe_uniform_resolved": 0,
    } for tau in TAUS}
    by_lam = {str(lam): {"dc_safe": 0, "released": 0, "rejected_corner": 0}
              for lam in LOADS}
    n_tier1_reject = 0
    n_unresolved_probes = 0

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
                continue
            ts = ts_action[lid]
            for lam in LOADS:
                P_lam = P_base - (lam - 1.0) * load_by_sub
                lo, hi = injection_bounds(st["gen_p"], st["load_p"] * lam,
                                          gen_to_sub, load_to_sub, n_sub,
                                          load_eps=LOAD_EPS)
                load_lam = load_by_sub * lam
                f_lo, f_hi = interval_flows_closedform(lines_act, n_sub, lo, hi)
                worst = np.maximum(np.abs(f_lo), np.abs(f_hi))
                worst_ratio = float(np.max(worst / caps))
                dc_safe = worst_ratio <= 1.0
                if not dc_safe:
                    n_tier1_reject += 1
                    continue
                by_lam[str(lam)]["dc_safe"] += 1
                probe = make_probe(ts, st, lam, n_sub, load_lam,
                                   gen_to_sub, load_to_sub)
                corners = corner_matrix(lines_act, n_sub, lo, hi)

                # 预门（全部 DC-safe 用例）的均匀采样 AC 越限率：200 逐行探针
                uni = rng.uniform(lo, hi, size=(N_UNIFORM, n_sub))
                u_viol = 0.0
                u_res = 0
                for row in uni:
                    ac_u = probe(np.asarray([row]))
                    if ac_u is None:
                        n_unresolved_probes += 1
                        continue
                    u_res += 1
                    if np.any(np.abs(ac_u[0]) > caps):
                        u_viol += 1.0
                for tau in TAUS:
                    agg[tau]["dcsafe_uniform_viol"] += u_viol
                    agg[tau]["dcsafe_uniform_resolved"] += u_res

                for tau in TAUS:
                    agg[tau]["dc_safe"] += 1
                    if worst_ratio < tau * (1.0 - MARGIN):
                        # 未触发第 2 层：直接放行，端到端残差沿用预门采样
                        agg[tau]["released"] += 1
                        agg[tau]["released_uniform_viol"] += u_viol
                        agg[tau]["released_uniform_resolved"] += u_res
                        if tau == 0.9:
                            by_lam[str(lam)]["released"] += 1
                        continue
                    agg[tau]["triggered"] += 1
                    t2 = time.time()
                    ok = True
                    unresolved = False
                    for Pv in corners:
                        agg[tau]["corner_solves"] += 1
                        ac = probe(np.asarray([Pv]))
                        if ac is None:
                            unresolved = True
                            ok = False
                            break
                        if np.any(np.abs(ac[0]) > caps):
                            ok = False
                            break
                    agg[tau]["tier2_time_s"] += time.time() - t2
                    if ok:
                        agg[tau]["released"] += 1
                        agg[tau]["released_uniform_viol"] += u_viol
                        agg[tau]["released_uniform_resolved"] += u_res
                        if tau == 0.9:
                            by_lam[str(lam)]["released"] += 1
                    elif unresolved:
                        agg[tau]["rejected_unresolved"] += 1
                    else:
                        agg[tau]["rejected_corner"] += 1
                        if tau == 0.9:
                            by_lam[str(lam)]["rejected_corner"] += 1
        print(f"  state {si+1}/20, {time.time()-t0:.0f}s", flush=True)

    # 第 2 层单角点 AC 求解微基准（5 批 × 200，协议同 §3.8）
    st0 = states[0]
    lam = 2.0
    load_by_sub = np.zeros(n_sub)
    for ld, s in enumerate(st0["load_to_sub"]):
        load_by_sub[s] += st0["load_p"][ld]
    lo, hi = injection_bounds(st0["gen_p"], st0["load_p"] * lam,
                              st0["gen_to_sub"], st0["load_to_sub"], n_sub,
                              load_eps=LOAD_EPS)
    lines_act = [dict(l) for l in lines]
    lines_act[0] = dict(lines[0], x=np.inf)
    corners0 = corner_matrix(lines_act, n_sub, lo, hi)
    load_lam = load_by_sub * lam
    probe0 = make_probe(ts_action[0], st0, lam, n_sub, load_lam,
                        st0["gen_to_sub"], st0["load_to_sub"])
    Pv = corners0[0]
    probe0(np.asarray([Pv]))  # 预热
    batch_meds = []
    for _ in range(5):
        ts = []
        for _ in range(200):
            t1 = time.perf_counter()
            probe0(np.asarray([Pv]))
            ts.append((time.perf_counter() - t1) * 1000.0)
        batch_meds.append(statistics.median(ts))
    probe_ms = statistics.median(batch_meds)

    out = {"case": "rte_case14_realistic",
           "protocol": "20 states x 20 actions x lambda in {2,3,4} (same sampling "
                       "as Sec. 3.6, RandomState 20260916); tier 2 = AC check at "
                       "deduplicated DC-worst corners of every active branch",
           "n_tier1_reject": n_tier1_reject,
           "corner_probe_median_ms": round(probe_ms, 3),
           "tau_results": {}}
    for tau in TAUS:
        a = agg[tau]
        out["tau_results"][str(tau)] = {
            "dc_safe_cases": a["dc_safe"],
            "tier2_triggered": a["triggered"],
            "gate_released": a["released"],
            "release_fraction_of_dc_safe": round(
                a["released"] / max(a["dc_safe"], 1), 4),
            "rejected_at_corner": a["rejected_corner"],
            "rejected_unresolved": a["rejected_unresolved"],
            "corner_solves_total": a["corner_solves"],
            "corner_solves_per_triggered": round(
                a["corner_solves"] / max(a["triggered"], 1), 2),
            "tier2_mean_ms_per_triggered": round(
                a["tier2_time_s"] / max(a["triggered"], 1) * 1000.0, 3),
            "sampled_ac_violation_rate_dcsafe": round(
                a["dcsafe_uniform_viol"] / max(a["dcsafe_uniform_resolved"], 1), 4),
            "sampled_ac_violation_rate_released": round(
                a["released_uniform_viol"] / max(a["released_uniform_resolved"], 1), 4),
        }
    out["by_lam_tau0.9"] = {
        k: {"dc_safe": v["dc_safe"], "released": v["released"],
            "rejected_corner": v["rejected_corner"]} for k, v in by_lam.items()}
    out["n_unresolved_uniform_probes"] = n_unresolved_probes
    out["elapsed_seconds"] = round(time.time() - t0, 1)

    os.makedirs("experiments/exp_019_review_ext/results", exist_ok=True)
    path = "experiments/exp_019_review_ext/results/twotier.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"结果已保存: {path}")


if __name__ == "__main__":
    main()
