"""exp_012 主实验：授权层对比评估（B0 无验证 / B1 标称 / B2 区间 / B3 蒙特卡洛）。

协议（experiment_plan.md v2 §4）：
- 每种子采样 100 个状态 × 5 候选动作（top-rho 线路断开）× 5 个负荷增长档 λ；
- 每用例：四验证器判定 + 真值 MC（在 ±ε 负荷箱内采样 M 点精确直流潮流，
  危险 = 任一支路越热限）；
- 指标：误放率 = P(真值存在危险 | 验证器放行)、误拒率 = P(验证器拒绝 | 真值全安全)、
  单动作延迟、危险率 vs M 的收敛曲线（铁律：保存中间状态与逐步收敛曲线）。

规模（铁律：大规模仿真 ≥2h）：100×5×5×5=12,500 用例/全种子 × M=5000 真值 MC
≈ 6×10^7 次 DC 潮流 ≈ 4-5h；M 检查点 [100, 300, 1000, 5000] 落盘收敛曲线。
冒烟：--smoke（2 状态 × 1 候选 × 1 λ × M=100）。

用法：python -m src.evaluation.auth_eval --out experiments/exp_012/results/auth_eval.json
"""
import argparse
import json
import os
import pickle
import time

import numpy as np

from src.utils.dc_interval import (
    authorize, dc_flows, injection_bounds, interval_flows_lp, theta_interval_lp,
)
from src.utils.env import Grid2OpGraphEnv

def _connected(lines_act, n_sub, slack=0):
    """连通性检查：去掉断开线路后，所有非松弛节点仍与主连通分量相连。"""
    adj = [[] for _ in range(n_sub)]
    for l in lines_act:
        if np.isfinite(l["x"]):
            adj[l["or"]].append(l["ex"])
            adj[l["ex"]].append(l["or"])
    seen = {slack}
    stack = [slack]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return len(seen) == n_sub


def sample_states(env, n_states, rng):
    """采样状态并保存原始数据（可复现）。返回 states 列表。

    场景数与步数边界自适应（dev 模式的 chronics 较少）。"""
    n_chron = max(int(env._n_chronics), 1)
    states = []
    for _ in range(n_states):
        env.env_glop.set_id(int(rng.randint(0, n_chron)))
        env.env_glop.reset()
        max_step = min(int(env.env_glop.chronics_handler.max_episode_duration() - 100),
                       1500)  # dev 模式 chronics 很短，设安全上限
        step = int(rng.randint(0, max(100, max_step)))
        for _attempt in range(6):
            try:
                env.env_glop.fast_forward_chronics(step)
                break
            except StopIteration:
                step //= 2  # 越界回退（dev 模式）
        obs = env.env_glop.get_obs()
        states.append({
            "gen_p": obs.gen_p.astype(np.float32).copy(),
            "load_p": obs.load_p.astype(np.float32).copy(),
            "gen_to_sub": obs.gen_to_subid.astype(int).copy(),
            "load_to_sub": obs.load_to_subid.astype(int).copy(),
            "rho": obs.rho.astype(np.float32).copy(),
            "line_status": obs.line_status.astype(np.float32).copy(),
        })
    return states


def verify_nominal(lines, n_sub, P, thermal, margin):
    f = dc_flows(lines, n_sub, P)
    ok, _ = authorize(f, f, thermal, margin=margin)
    return ok


def verify_interval(lines, n_sub, lo, hi, thermal, margin):
    f_lo, f_hi = interval_flows_lp(lines, n_sub, lo, hi)
    ok, _ = authorize(f_lo, f_hi, thermal, margin=margin)
    return ok


def verify_mc(lines, n_sub, lo, hi, thermal, K, rng, margin):
    """K 点蒙特卡洛经验最坏情况验证（B3，统计近似无构造保证）。"""
    for _ in range(K):
        P = rng.uniform(lo, hi)
        f = dc_flows(lines, n_sub, P)
        ok, _ = authorize(f, f, thermal, margin=margin)
        if not ok:
            return False
    return True


def gt_danger_rate(lines, n_sub, lo, hi, thermal, M, rng, margin, dist="uniform"):
    """真值：M 点采样的危险率（存在越限的采样点比例）。

    dist: uniform=箱内均匀；gaussian=以标称点为中心的高斯量测噪声（σ=箱半宽/2）。
    """
    danger = 0
    max_ratio = 0.0
    mid = (np.asarray(lo) + np.asarray(hi)) / 2
    half = (np.asarray(hi) - np.asarray(lo)) / 2
    for _ in range(M):
        if dist == "gaussian":
            P = rng.normal(mid, half / 2)
        else:
            P = rng.uniform(lo, hi)
        f = dc_flows(lines, n_sub, P)
        ok, _ = authorize(f, f, thermal, margin=margin)
        if not ok:
            danger += 1
        max_ratio = max(max_ratio, float(np.max(np.abs(f) / thermal)))
    return danger / M, max_ratio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-states", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-candidates", type=int, default=5)
    ap.add_argument("--lamdas", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0, 5.0])
    ap.add_argument("--mc-k", type=int, default=100, help="B3 蒙特卡洛验证点数")
    ap.add_argument("--gt-m", type=int, default=5000, help="真值 MC 点数")
    ap.add_argument("--conv-checkpoints", type=int, nargs="+",
                    default=[100, 300, 1000, 5000])
    ap.add_argument("--out", default="experiments/exp_012/results/auth_eval.json")
    ap.add_argument("--states-cache", default="data/processed/auth_states.pkl",
                    help="采样状态缓存（跨运行复用同一状态集）")
    ap.add_argument("--lines-cache", default="data/processed/auth_lines.pkl")
    ap.add_argument("--proposals", default=None,
                    help="LLM 提案数据集 JSON（exp_017：用提案动作替代 top-rho 候选）")
    ap.add_argument("--load-eps", type=float, default=0.2,
                    help="验证器负荷不确定性宽度（消融维度 1：0.05/0.1/0.2/0.4）")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="验证器安全裕度（消融维度 5：0/0.02/0.05）")
    ap.add_argument("--gt-eps", type=float, default=None,
                    help="真值失配箱宽度（默认=load-eps；跨配置比较时固定为参考值）")
    ap.add_argument("--gt-margin", type=float, default=None,
                    help="真值安全裕度（默认=margin；跨配置比较时固定为参考值）")
    ap.add_argument("--require-connected", action="store_true",
                    help="验证量增加连通性检查（消融维度 2）")
    ap.add_argument("--theta-cap", type=float, default=None,
                    help="验证量增加电压角界 |θ|≤cap（DC 电压近似，消融维度 2 第三变体）")
    ap.add_argument("--lines-file", default=None,
                    help="自定义线路模型 pickle（消融维度 3：uniform/少样本识别）")
    ap.add_argument("--proposal-source", choices=["top", "random"], default="top",
                    help="提案源（消融维度 4）：top-rho 候选 / 随机 1-3 条线路")
    ap.add_argument("--prob-threshold", type=float, default=None,
                    help="概率安全判据（消融维度 6）：MC 验证 ≥阈值 采样点安全即放行")
    ap.add_argument("--case", default="rte_case14_realistic",
                    help="电网场景（exp_015 跨系统：l2rpn_icaps_2021 / l2rpn_idf_2023）")
    ap.add_argument("--test-mode", action="store_true",
                    help="使用内置开发模式数据（icaps/idf 无完整数据集）")
    ap.add_argument("--gt-dist", choices=["uniform", "gaussian"], default="uniform",
                    help="真值失配分布（exp_014 鲁棒性：高斯量测噪声变体）")
    ap.add_argument("--missing-frac", type=float, default=0.0,
                    help="不可观测负荷比例（exp_014：缺失负荷的箱加宽到 ±100%）")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：跳过已存在于 *_cases.jsonl 的用例")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_states, args.n_candidates, args.lamdas = 2, 1, [3.0]
        args.gt_m = 100
        args.conv_checkpoints = [100]

    t0 = time.time()
    env = Grid2OpGraphEnv(case=args.case, test=args.test_mode)
    n_sub = int(env.env_glop.n_sub)
    thermal = env.env_glop.get_thermal_limit().astype(float)

    # 遥测识别 DC 模型（复用尖刺实现，缓存按场景隔离）
    from experiments.exp_011_interval_auth.spike_interval_auth import (
        estimate_lines_from_telemetry,
    )
    lines_cache = args.lines_cache.replace(".pkl", f"_{args.case.replace('l2rpn_', '')}.pkl")
    if args.lines_file:
        with open(args.lines_file, "rb") as f:
            lines = pickle.load(f)
        print(f"lines 模型从自定义文件加载: {args.lines_file}")
    elif os.path.exists(lines_cache):
        with open(lines_cache, "rb") as f:
            lines = pickle.load(f)
        print("lines 模型从缓存加载")
    else:
        lines, _n_sub_est = estimate_lines_from_telemetry(env, n_states=60)
        os.makedirs(os.path.dirname(lines_cache), exist_ok=True)
        with open(lines_cache, "wb") as f:
            pickle.dump(lines, f)
        print("lines 模型已识别并缓存")

    # 状态采样（缓存；缓存文件名含场景名，跨系统隔离）
    states_cache = args.states_cache.replace(
        ".pkl", f"_{args.case.replace('l2rpn_', '')}.pkl")
    if os.path.exists(states_cache):
        with open(states_cache, "rb") as f:
            states = pickle.load(f)
        print(f"状态缓存加载：{len(states)} 个")
    else:
        rng0 = np.random.RandomState(0)
        states = sample_states(env, args.n_states, rng0)
        os.makedirs(os.path.dirname(states_cache), exist_ok=True)
        with open(states_cache, "wb") as f:
            pickle.dump(states, f)
        print(f"采样 {len(states)} 个状态并缓存")

    # 每验证器累计
    acc = {k: {"passed": 0, "passed_danger": 0, "passed_any_danger": 0,
               "rejected": 0, "rejected_safe": 0, "latency": []}
           for k in ("none", "nominal", "interval", "mc")}
    conv = {m: {k: {"danger_sum": 0, "passed": 0} for k in ("nominal", "interval", "mc")}
            for m in args.conv_checkpoints}
    case_records = []
    cases_path = args.out.replace(".json", "_cases.jsonl")
    done_cases = set()
    if args.resume and os.path.exists(cases_path):
        with open(cases_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done_cases.add((r["seed"], r.get("state", -1), r["tag"],
                                r["lambda"], tuple(r["disconnects"])))
        print(f"断点续跑：已跳过 {len(done_cases)} 个已记录用例")
    cases_fh = open(cases_path, "a", encoding="utf-8") if args.resume else None

    # LLM 提案模式（exp_017）：提案动作集替代 top-rho 候选
    proposals_by_state = None
    if args.proposals:
        with open(args.proposals, encoding="utf-8") as f:
            prop = json.load(f)
        proposals_by_state = {p["state_id"]: p["verifier_aware"] for p in prop["proposals"]}

    for seed in args.seeds:
        rng = np.random.RandomState(seed)
        for si, st in enumerate(states):
            gen_to_sub = st["gen_to_sub"]
            load_to_sub = st["load_to_sub"]
            load_by_sub = np.zeros(n_sub)
            for ld, s in enumerate(load_to_sub):
                load_by_sub[s] += st["load_p"][ld]
            P_base = np.zeros(n_sub)
            for g, s in enumerate(gen_to_sub):
                P_base[s] += st["gen_p"][g]
            P_base -= load_by_sub

            if proposals_by_state is not None:
                props = proposals_by_state.get(si, [])
                cases = [("llm", [int(l) for l in props])] if props else []
            elif args.proposal_source == "random":
                on_lines = np.nonzero(st["line_status"] == 1)[0]
                k = int(rng.randint(1, 4))
                picked = rng.choice(on_lines, size=min(k, len(on_lines)),
                                    replace=False).tolist()
                cases = [("random", picked)]
            else:
                cand = np.argsort(-st["rho"])[: args.n_candidates]
                cases = [("top", [int(lid)]) for lid in cand
                         if st["line_status"][lid] == 1]
            for tag, disconnects in cases:
                lines_act = [dict(l) for l in lines]
                for lid in disconnects:
                    lines_act[lid] = dict(lines[lid], x=np.inf)  # 断开提案线路
                for lam in args.lamdas:
                    if args.resume and (seed, si, tag, lam, tuple(disconnects)) in done_cases:
                        continue  # 断点续跑：跳过已完成用例
                    P_lam = P_base - (lam - 1.0) * load_by_sub
                    lo, hi = injection_bounds(st["gen_p"], st["load_p"] * lam,
                                              gen_to_sub, load_to_sub, n_sub,
                                              load_eps=args.load_eps)
                    gt_eps = args.gt_eps if args.gt_eps is not None else args.load_eps
                    gt_lo, gt_hi = injection_bounds(
                        st["gen_p"], st["load_p"] * lam, gen_to_sub, load_to_sub,
                        n_sub, load_eps=gt_eps)
                    if args.missing_frac > 0:
                        # 不可观测负荷：箱加宽到 ±100%（保守覆盖）；真值箱同加宽
                        n_loads = len(load_to_sub)
                        n_missing = max(1, int(n_loads * args.missing_frac))
                        missing = rng.choice(n_loads, n_missing, replace=False)
                        load_obs = st["load_p"] * lam
                        for ld in missing:
                            s_ld = load_to_sub[ld]
                            lo[s_ld] = min(lo[s_ld], lo[s_ld] - load_obs[ld])
                            hi[s_ld] = max(hi[s_ld], hi[s_ld] + load_obs[ld])
                            gt_lo[s_ld] = min(gt_lo[s_ld], gt_lo[s_ld] - load_obs[ld])
                            gt_hi[s_ld] = max(gt_hi[s_ld], gt_hi[s_ld] + load_obs[ld])
                    # 连通性是无条件前置：断网动作对全部验证器直接拒绝（物理无效）
                    conn_ok = _connected(lines_act, n_sub)
                    if not conn_ok:
                        nom_ok = int_ok = mc_ok = False
                        for key in ("nominal", "interval", "mc"):
                            acc[key]["latency"].append(0.0)
                    else:
                        theta_ok = True
                        if args.theta_cap is not None:
                            th_lo, th_hi = theta_interval_lp(lines_act, n_sub, lo, hi)
                            theta_ok = bool(np.max(np.maximum(np.abs(th_lo),
                                                              np.abs(th_hi)))
                                            <= args.theta_cap)
                        # B1 标称
                        t1 = time.time()
                        nom_ok = verify_nominal(lines_act, n_sub, P_lam, thermal,
                                                args.margin)
                        acc["nominal"]["latency"].append(time.time() - t1)
                        # B2 区间
                        t2 = time.time()
                        int_ok = verify_interval(lines_act, n_sub, lo, hi, thermal,
                                                 args.margin)
                        acc["interval"]["latency"].append(time.time() - t2)
                        # B3 蒙特卡洛（--prob-threshold 时用概率安全判据）
                        t3 = time.time()
                        if args.prob_threshold is not None:
                            n_safe = sum(1 for _ in range(args.mc_k)
                                         if verify_nominal(lines_act, n_sub,
                                                           rng.uniform(lo, hi),
                                                           thermal, args.margin))
                            mc_ok = n_safe >= args.prob_threshold * args.mc_k
                        else:
                            mc_ok = verify_mc(lines_act, n_sub, lo, hi, thermal,
                                              args.mc_k, rng, args.margin)
                        acc["mc"]["latency"].append(time.time() - t3)
                        nom_ok = nom_ok and theta_ok
                        int_ok = int_ok and theta_ok
                        mc_ok = mc_ok and theta_ok

                    decisions = {"none": True,
                                 "nominal": nom_ok,
                                 "interval": int_ok,
                                 "mc": mc_ok}
                    # 真值（分 checkpoint 累计收敛曲线）；
                    # 断网动作的真值直接定义为危险（孤岛 = 物理无效）
                    if not conn_ok:
                        gt_danger = {m: (1.0, float("inf"))
                                     for m in args.conv_checkpoints}
                        d_final, ratio_final = 1.0, float("inf")
                    else:
                        gt_margin = (args.gt_margin if args.gt_margin is not None
                                     else args.margin)
                        gt_danger = {}
                        for m in args.conv_checkpoints:
                            d, ratio = gt_danger_rate(lines_act, n_sub, gt_lo, gt_hi,
                                                      thermal, m, rng, gt_margin,
                                                      dist=args.gt_dist)
                            gt_danger[m] = (d, ratio)
                        d_final, ratio_final = gt_danger[args.conv_checkpoints[-1]]
                    any_danger = d_final > 0

                    for key, passed in decisions.items():
                        if passed:
                            acc[key]["passed"] += 1
                            acc[key]["passed_danger"] += d_final
                            acc[key]["passed_any_danger"] += int(any_danger)
                        else:
                            acc[key]["rejected"] += 1
                            if not any_danger:
                                acc[key]["rejected_safe"] += 1
                    for m in args.conv_checkpoints:
                        for key in ("nominal", "interval", "mc"):
                            if decisions[key]:
                                conv[m][key]["danger_sum"] += gt_danger[m][0]
                                conv[m][key]["passed"] += 1
                    rec = {
                        "seed": seed, "state": si, "tag": tag, "lambda": lam,
                        "disconnects": [int(x) for x in disconnects],
                        "nominal_ok": bool(nom_ok), "interval_ok": bool(int_ok),
                        "mc_ok": bool(mc_ok),
                        "gt_danger_rate": d_final, "gt_max_ratio": ratio_final,
                    }
                    case_records.append(rec)
                    if cases_fh is not None:
                        cases_fh.write(json.dumps(rec) + "\n")
                        cases_fh.flush()  # 增量落盘（会话中断不丢进度）

    if cases_fh is not None:
        cases_fh.close()
    # 断点续跑：合并已完成用例重建汇总（acc 计数器从全量记录推导）
    if args.resume and done_cases:
        with open(cases_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if (r["seed"], r.get("state", -1), r["tag"], r["lambda"],
                        tuple(r["disconnects"])) in done_cases:
                    case_records.append(r)
        acc = {k: {"passed": 0, "passed_danger": 0, "passed_any_danger": 0,
                   "rejected": 0, "rejected_safe": 0, "latency": []}
               for k in ("none", "nominal", "interval", "mc")}
        ok_key = {"none": None, "nominal": "nominal_ok",
                  "interval": "interval_ok", "mc": "mc_ok"}
        for r in case_records:
            danger = r["gt_danger_rate"] > 0
            for key, kk in ok_key.items():
                passed = True if kk is None else bool(r[kk])
                if passed:
                    acc[key]["passed"] += 1
                    acc[key]["passed_danger"] += r["gt_danger_rate"]
                    acc[key]["passed_any_danger"] += int(danger)
                else:
                    acc[key]["rejected"] += 1
                    acc[key]["rejected_safe"] += int(not danger)

    # 汇总
    summary = {}
    for key in acc:
        a = acc[key]
        summary[key] = {
            "passed": a["passed"], "rejected": a["rejected"],
            "false_pass_rate": float(a["passed_any_danger"] / max(a["passed"], 1)),
            "mean_danger_when_passed": float(a["passed_danger"] / max(a["passed"], 1)),
            "false_reject_rate": float(a["rejected_safe"] / max(a["rejected"], 1)),
            "latency_ms_mean": float(np.mean(a["latency"]) * 1000) if a["latency"] else 0.0,
        }
    conv_curves = {}
    for m in args.conv_checkpoints:
        conv_curves[str(m)] = {k: float(conv[m][k]["danger_sum"] / max(conv[m][k]["passed"], 1))
                               for k in ("nominal", "interval", "mc")}
    if args.resume:
        conv_curves["_partial"] = "恢复模式下收敛曲线仅覆盖本次新增用例"

    result = {"args": vars(args), "summary": summary,
              "conv_curves": conv_curves,
              "n_cases": len(case_records),
              "elapsed_seconds": round(time.time() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    if not args.resume:  # 非恢复模式全量写；恢复模式增量已写
        with open(cases_path, "w") as f:
            for r in case_records:
                f.write(json.dumps(r) + "\n")
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in summary.items()},
                     indent=2, ensure_ascii=False))
    print(f"收敛曲线: {json.dumps(conv_curves, ensure_ascii=False)}")
    print(f"耗时 {result['elapsed_seconds']}s（≥7200s: {result['elapsed_seconds'] >= 7200}）")
    print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
