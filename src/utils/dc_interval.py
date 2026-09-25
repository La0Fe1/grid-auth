"""区间直流潮流：不确定性鲁棒安全授权的数学核心（方向 B 重定义）。

原理：
- DC 潮流对节点注入是线性的：θ = B⁺ P，支路潮流 f_ℓ = (θ_or − θ_ex)/x_ℓ；
- 负荷不确定性箱 P ∈ [P_min, P_max] → 每个支路潮流的**精确区间**
  = 对注入箱的线性规划（min/max f_ℓ(P)，scipy.linprog）——精确而非估计；
- 安全授权判定：区间界对全部支路均 ≤ 限值−裕度 才放行
  → 对箱内所有模型构造性安全（标称验证做不到的性质）。

数据来源：Grid2Op grid.json（线路电抗、拓扑）+ 观测注入 + env.get_thermal_limit()。
"""
import json

import numpy as np
from scipy.optimize import linprog


def load_grid(grid_json_path):
    """从 grid.json 读线路电抗与节点拓扑。返回 (lines, n_sub)。

    lines: list of dict(or_sub, ex_sub, x)；substation 编号从 grid.json 的
    substations 顺序映射（0-based）。grid2op 的 substation 序号与 grid.json
    substations 数组顺序一致（grid2op 生成）。
    """
    with open(grid_json_path) as f:
        gj = json.load(f)
    n_sub = len(gj["substations"])
    lines = []
    for l in gj["lines"]:
        lines.append({
            "or": int(l["origin"]),
            "ex": int(l["extremity"]),
            "x": float(l["x"]),
        })
    return lines, n_sub


def build_susceptance(lines, n_sub):
    """DC 潮流 B 矩阵（去松弛节点前的满阵）。"""
    B = np.zeros((n_sub, n_sub))
    for l in lines:
        b = 1.0 / l["x"]
        B[l["or"], l["or"]] += b
        B[l["ex"], l["ex"]] += b
        B[l["or"], l["ex"]] -= b
        B[l["ex"], l["or"]] -= b
    return B


def dc_flows(lines, n_sub, P, slack=0):
    """标称 DC 潮流。P: 节点注入 (n_sub,)（负荷为负）；返回每条支路潮流数组。"""
    B = build_susceptance(lines, n_sub)
    P_red = np.delete(P, slack)
    B_red = np.delete(np.delete(B, slack, axis=0), slack, axis=1)
    theta_red = np.linalg.solve(B_red, P_red)
    theta = np.insert(theta_red, slack, 0.0)
    return np.array([(theta[l["or"]] - theta[l["ex"]]) / l["x"] for l in lines])


def injection_bounds(gen_p, load_p, gen_to_sub, load_to_sub, n_sub, load_eps=0.2):
    """负荷 ±load_eps 箱下的节点注入下/上界（发电视为已知）。"""
    P_base = np.zeros(n_sub)
    for g, s in enumerate(gen_to_sub):
        P_base[s] += gen_p[g]
    for ld, s in enumerate(load_to_sub):
        P_base[s] -= load_p[ld]
    load_by_sub = np.zeros(n_sub)
    for ld, s in enumerate(load_to_sub):
        load_by_sub[s] += load_p[ld]
    # 箱宽用负荷绝对值：负负荷（新能源反送）同样有 ±ε 量测不确定性
    width = load_eps * np.abs(load_by_sub)
    lo = P_base - width
    hi = P_base + width
    return lo, hi


def interval_flows_lp(lines, n_sub, lo, hi, slack=0):
    """每个支路潮流在注入箱上的精确区间（线性规划，逐支路 min/max）。"""
    n_b = len(lines)
    B = build_susceptance(lines, n_sub)
    keep = [s for s in range(n_sub) if s != slack]
    idx_of = {s: i for i, s in enumerate(keep)}  # 原始节点号 → 降阶系统索引
    B_red = np.delete(np.delete(B, slack, axis=0), slack, axis=1)
    B_inv = np.linalg.inv(B_red)
    # θ_red = B_red⁻¹ P_red；支路潮流 = Σ_s c_s P_s（线性）
    # c_ℓ_s: 支路 ℓ 对节点 s 注入的灵敏度
    n_var = len(keep)
    flow_lo, flow_hi = np.zeros(n_b), np.zeros(n_b)
    for li, l in enumerate(lines):
        c = np.zeros(n_var)
        for si, s in enumerate(keep):
            t_or = B_inv[si, idx_of[l["or"]]] if l["or"] != slack else 0.0
            t_ex = B_inv[si, idx_of[l["ex"]]] if l["ex"] != slack else 0.0
            c[si] = (t_or - t_ex) / l["x"]
        bounds = [(lo[s], hi[s]) for s in keep]
        r_min = linprog(c, bounds=bounds, method="highs")
        r_max = linprog(-c, bounds=bounds, method="highs")
        flow_lo[li] = r_min.fun if r_min.success else float("nan")
        flow_hi[li] = -r_max.fun if r_max.success else float("nan")
    return flow_lo, flow_hi


def interval_flows_closedform(lines, n_sub, lo, hi, slack=0):
    """每个支路潮流在注入箱上的精确区间（闭式角点公式，与 LP 等价）。

    线性映射在箱约束上的极值必在角点取得：
      f_ℓ^+ = Σ_s max(c_s lo_s, c_s hi_s)，f_ℓ^- = Σ_s min(c_s lo_s, c_s hi_s)，
    其中 c_s = (B_red⁻¹ 的行差)/x_ℓ 为支路 ℓ 对节点 s 注入的灵敏度。
    纯箱约束下与 interval_flows_lp 完全一致（单元测试验证），
    但免去 2|L| 次单纯形求解；LP 形式保留用于多面体/附加线性约束的扩展。
    2026-09-16 向量化：c 行 = (B_inv[or,:]−B_inv[ex,:])/x（每支路一次向量运算，
    免去逐元素标量索引的 O(|N|) 内层循环）。
    """
    n_b = len(lines)
    B = build_susceptance(lines, n_sub)
    keep = [s for s in range(n_sub) if s != slack]
    idx_of = {s: i for i, s in enumerate(keep)}
    B_red = np.delete(np.delete(B, slack, axis=0), slack, axis=1)
    B_inv = np.linalg.inv(B_red)
    lo_v = np.asarray([lo[s] for s in keep], dtype=float)
    hi_v = np.asarray([hi[s] for s in keep], dtype=float)
    flow_lo = np.zeros(n_b)
    flow_hi = np.zeros(n_b)
    for li, l in enumerate(lines):
        if not np.isfinite(l["x"]):
            continue  # 已断开支路：灵敏度为零，流量区间 = [0, 0]
        c = np.zeros(len(keep))
        if l["or"] != slack:
            c += B_inv[idx_of[l["or"]], :]
        if l["ex"] != slack:
            c -= B_inv[idx_of[l["ex"]], :]
        c /= l["x"]
        flow_lo[li] = float(np.sum(np.minimum(c * lo_v, c * hi_v)))
        flow_hi[li] = float(np.sum(np.maximum(c * lo_v, c * hi_v)))
    return flow_lo, flow_hi


def witness_closedform(lines, n_sub, lo, hi, thermal, margin=0.05, slack=0):
    """闭式角点版本的对抗反例：与 adversarial_witness 等价（纯箱约束）。

    对每条支路，正/负方向最坏角点由灵敏度符号直接给出
    （P*_s = hi_s 若 c_s≥0 否则 lo_s，负方向取反），无需 LP 求解。
    """
    B = build_susceptance(lines, n_sub)
    keep = [s for s in range(n_sub) if s != slack]
    idx_of = {s: i for i, s in enumerate(keep)}
    B_red = np.delete(np.delete(B, slack, axis=0), slack, axis=1)
    B_inv = np.linalg.inv(B_red)
    lo_v = np.asarray([lo[s] for s in keep], dtype=float)
    hi_v = np.asarray([hi[s] for s in keep], dtype=float)
    caps = np.asarray(thermal, dtype=float) * (1.0 - margin)
    worst_ratio, worst = -1.0, None
    for li, l in enumerate(lines):
        if not np.isfinite(l["x"]) or not np.isfinite(caps[li]) or caps[li] <= 0:
            continue
        c = np.zeros(len(keep))
        if l["or"] != slack:
            c += B_inv[idx_of[l["or"]], :]
        if l["ex"] != slack:
            c -= B_inv[idx_of[l["ex"]], :]
        c /= l["x"]
        for sign in (1.0, -1.0):
            P_red = np.where(sign * c >= 0, hi_v, lo_v)
            f_val = float(sign * np.sum(np.where(sign * c >= 0,
                                                 sign * c * hi_v, sign * c * lo_v)))
            ratio = abs(f_val) / caps[li]
            if ratio > worst_ratio:
                P_full = np.zeros(n_sub)
                P_full[keep] = P_red
                P_full[slack] = (lo[slack] + hi[slack]) / 2
                worst_ratio = ratio
                worst = (P_full, li, f_val, caps[li])
    if worst_ratio <= 1.0:
        return None
    return worst


def theta_interval_closedform(lines, n_sub, lo, hi, slack=0):
    """节点电压角 θ 在注入箱上的精确区间（闭式角点公式，与 theta_interval_lp 等价）。"""
    keep = [s for s in range(n_sub) if s != slack]
    idx_of = {s: i for i, s in enumerate(keep)}
    B = build_susceptance(lines, n_sub)
    B_red = np.delete(np.delete(B, slack, axis=0), slack, axis=1)
    B_inv = np.linalg.inv(B_red)
    lo_v = np.asarray([lo[s] for s in keep], dtype=float)
    hi_v = np.asarray([hi[s] for s in keep], dtype=float)
    th_lo = np.zeros(n_sub)
    th_hi = np.zeros(n_sub)
    for s in keep:
        c = B_inv[idx_of[s], :]
        th_lo[s] = float(np.sum(np.minimum(c * lo_v, c * hi_v)))
        th_hi[s] = float(np.sum(np.maximum(c * lo_v, c * hi_v)))
    return th_lo, th_hi


def adversarial_witness(lines, n_sub, lo, hi, thermal, margin=0.05, slack=0):
    """对不安全动作生成对抗反例：一个具体的箱内负荷向量 P*，
    使某条支路在 (1−margin)×限值 判据下越限。

    返回 (P_full, worst_line, worst_flow, worst_cap) 或 None（动作安全时）。
    P_full 为 (n_sub,) 完整注入向量（含松弛节点，松弛取箱中点）。
    """
    B = build_susceptance(lines, n_sub)
    keep = [s for s in range(n_sub) if s != slack]
    idx_of = {s: i for i, s in enumerate(keep)}
    B_red = np.delete(np.delete(B, slack, axis=0), slack, axis=1)
    B_inv = np.linalg.inv(B_red)
    caps = thermal * (1.0 - margin)
    worst_ratio, worst = -1.0, None
    for li, l in enumerate(lines):
        c = np.zeros(len(keep))
        for si, s in enumerate(keep):
            t_or = B_inv[si, idx_of[l["or"]]] if l["or"] != slack else 0.0
            t_ex = B_inv[si, idx_of[l["ex"]]] if l["ex"] != slack else 0.0
            c[si] = (t_or - t_ex) / l["x"]
        bounds = [(lo[s], hi[s]) for s in keep]
        # 正方向最坏情况（使 f 最大的注入）
        r_pos = linprog(-c, bounds=bounds, method="highs")
        if r_pos.success:
            f_pos = -r_pos.fun
            ratio = abs(f_pos) / caps[li]
            if ratio > worst_ratio:
                P_red = r_pos.x
                P_full = np.zeros(n_sub)
                P_full[keep] = P_red
                P_full[slack] = (lo[slack] + hi[slack]) / 2
                worst_ratio = ratio
                worst = (P_full, li, f_pos, caps[li])
        # 负方向最坏情况
        r_neg = linprog(c, bounds=bounds, method="highs")
        if r_neg.success:
            f_neg = r_neg.fun
            ratio = abs(f_neg) / caps[li]
            if ratio > worst_ratio:
                P_red = r_neg.x
                P_full = np.zeros(n_sub)
                P_full[keep] = P_red
                P_full[slack] = (lo[slack] + hi[slack]) / 2
                worst_ratio = ratio
                worst = (P_full, li, f_neg, caps[li])
    if worst_ratio <= 1.0:
        return None
    return worst


def theta_interval_lp(lines, n_sub, lo, hi, slack=0):
    """节点电压角 θ 在注入箱上的精确区间（DC 电压近似；线性规划逐节点 min/max）。"""
    keep = [s for s in range(n_sub) if s != slack]
    idx_of = {s: i for i, s in enumerate(keep)}
    B = build_susceptance(lines, n_sub)
    B_red = np.delete(np.delete(B, slack, axis=0), slack, axis=1)
    B_inv = np.linalg.inv(B_red)
    th_lo = np.zeros(n_sub)
    th_hi = np.zeros(n_sub)
    for s in keep:
        c = B_inv[idx_of[s], :]  # θ_s = Σ c_j P_j
        bounds = [(lo[ss], hi[ss]) for ss in keep]
        r_min = linprog(c, bounds=bounds, method="highs")
        r_max = linprog(-c, bounds=bounds, method="highs")
        th_lo[s] = r_min.fun if r_min.success else float("nan")
        th_hi[s] = -r_max.fun if r_max.success else float("nan")
    return th_lo, th_hi


def authorize(flow_lo, flow_hi, thermal_limits, margin=0.05):
    """区间安全授权：所有支路的 |流量区间| 上界 ≤ (1−margin)×限值 才放行。

    fail-closed（2026-09-14 修复）：任何 NaN/Inf（LP 失败、模型病态）→ 拒绝。
    NaN 比较恒为 False 会导致静默放行，是安全系统不可接受的缺陷。
    返回 (authorized, violations)：violations = 越限/异常支路索引列表。
    """
    flow_lo = np.asarray(flow_lo, dtype=float)
    flow_hi = np.asarray(flow_hi, dtype=float)
    caps = thermal_limits * (1.0 - margin)
    worst = np.maximum(np.abs(flow_lo), np.abs(flow_hi))
    abnormal = ~np.isfinite(worst)
    viol = np.where((worst > caps) | abnormal)[0]
    return len(viol) == 0, viol.tolist()
