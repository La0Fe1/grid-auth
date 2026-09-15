"""src/utils/dc_interval.py 单元测试（解析与构造性质）。"""
import numpy as np
import pytest

from src.utils.dc_interval import (
    authorize, build_susceptance, dc_flows, injection_bounds, interval_flows_lp,
)


def make_lines(n_sub=4):
    # 链式网络 0-1-2-3
    return [{"or": i, "ex": i + 1, "x": 0.1} for i in range(n_sub - 1)], n_sub


def test_build_susceptance_symmetry():
    lines, n = make_lines()
    B = build_susceptance(lines, n)
    assert np.allclose(B, B.T)
    assert np.allclose(B.sum(axis=1), 0)  # 行和为零


def test_dc_flows_energy_conservation():
    lines, n = make_lines()
    # 节点 3 注入 1.0（其余 0），slack=0 → 功率沿 3→2→1→0 流动，
    # 符号约定：正 = or→ex 方向 → 支路 0-1/1-2/2-3 上均为 -1
    P = np.zeros(n)
    P[3] = 1.0
    flows = dc_flows(lines, n, P, slack=0)
    assert np.allclose(flows, [-1.0, -1.0, -1.0])


def test_dc_flows_linearity():
    lines, n = make_lines()
    P = np.array([0.0, -0.5, 0.3, 0.9])
    f1 = dc_flows(lines, n, P)
    f2 = dc_flows(lines, n, 2 * P)
    assert np.allclose(f2, 2 * f1)


def test_injection_bounds_box():
    gen_p = np.array([2.0])
    load_p = np.array([0.5, 0.5])
    lo, hi = injection_bounds(gen_p, load_p, [0], [1, 2], n_sub=3, load_eps=0.2)
    assert np.all(hi >= lo)
    assert np.allclose(hi[0], 2.0)  # 发电节点无负荷不确定性
    assert np.allclose(lo[1], -0.6)  # 负荷 0.5 × 1.2


def test_injection_bounds_negative_load():
    """负负荷（新能源反送）下箱必须仍然有效（lo ≤ hi）——icaps 环境实测教训。"""
    gen_p = np.array([2.0])
    load_p = np.array([-0.5, 0.5])
    lo, hi = injection_bounds(gen_p, load_p, [0], [1, 2], n_sub=3, load_eps=0.2)
    assert np.all(hi >= lo)
    # 负负荷节点：箱宽 = ε×|负荷| = 0.1
    assert np.allclose(lo[1] - hi[1], -0.2)


def test_interval_contains_nominal():
    lines, n = make_lines()
    # 标称注入须在箱内：非 slack 节点全部为负荷（负注入）
    P = np.array([0.2, -0.7, -0.4, -0.5])
    lo, hi = injection_bounds(np.array([2.0]), np.abs(P[1:]), [0], [1, 2, 3],
                              n_sub=n, load_eps=0.3)
    # 松弛节点注入由平衡隐式决定，箱断言只覆盖非松弛节点
    assert np.all(lo[1:] <= P[1:]) and np.all(hi[1:] >= P[1:])
    f_lo, f_hi = interval_flows_lp(lines, n, lo, hi, slack=0)
    f_nom = dc_flows(lines, n, P, slack=0)
    assert np.all(f_lo <= f_nom + 1e-6)
    assert np.all(f_hi >= f_nom - 1e-6)


def test_interval_wider_than_nominal():
    lines, n = make_lines()
    lo = np.array([-0.5, -0.6, -0.3, -0.4])
    hi = np.array([0.5, -0.4, 0.3, 0.6])
    f_lo, f_hi = interval_flows_lp(lines, n, lo, hi, slack=0)
    assert np.all(f_hi - f_lo >= 0)


def test_authorize_rejects_overloaded():
    lines, n = make_lines()
    f_lo = np.array([-0.9, -0.9, -0.9])
    f_hi = np.array([1.2, 0.3, 0.3])
    caps = np.array([1.0, 1.0, 1.0])
    ok, viol = authorize(f_lo, f_hi, caps, margin=0.0)
    assert not ok and viol == [0]


def test_authorize_accepts_safe():
    lines, n = make_lines()
    f_lo = np.array([-0.8, -0.2, -0.2])
    f_hi = np.array([0.8, 0.2, 0.2])
    ok, viol = authorize(f_lo, f_hi, np.array([1.0, 1.0, 1.0]), margin=0.0)
    assert ok and viol == []
