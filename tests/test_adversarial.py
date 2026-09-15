"""src/utils/dc_interval.py 的对抗反例生成测试。"""
import numpy as np
import pytest

from src.utils.dc_interval import (
    adversarial_witness, dc_flows, injection_bounds,
)


def make_chain(n_sub=4):
    lines = [{"or": i, "ex": i + 1, "x": 0.1} for i in range(n_sub - 1)]
    return lines, n_sub


def test_witness_none_when_safe():
    lines, n = make_chain()
    lo = np.array([-0.5, -0.6, -0.3, -0.4])
    hi = np.array([0.5, -0.4, 0.3, 0.6])
    caps = np.array([10.0, 10.0, 10.0])
    assert adversarial_witness(lines, n, lo, hi, caps, margin=0.0) is None


def test_witness_is_in_box_and_violates():
    """反例必须在箱内、且确使某支路越限。"""
    lines, n = make_chain()
    lo = np.array([-0.1, -0.8, -0.6, -0.4])
    hi = np.array([0.9, -0.2, 0.4, 0.6])
    caps = np.array([0.5, 0.5, 0.5])
    w = adversarial_witness(lines, n, lo, hi, caps, margin=0.0)
    assert w is not None
    P, line, flow, cap = w
    # 箱内（非松弛节点）
    assert np.all(P[1:] >= lo[1:] - 1e-6) and np.all(P[1:] <= hi[1:] + 1e-6)
    # 越限
    assert abs(flow) > cap
    # 与 DC 潮流一致
    f = dc_flows(lines, n, P)
    assert np.isclose(f[line], flow, rtol=1e-4)


def test_witness_matches_authorize():
    """有反例 ⇔ authorize 拒绝。"""
    lines, n = make_chain()
    lo = np.array([-0.1, -0.8, -0.6, -0.4])
    hi = np.array([0.9, -0.2, 0.4, 0.6])
    caps = np.array([0.5, 0.5, 0.5])
    from src.utils.dc_interval import authorize, interval_flows_lp
    f_lo, f_hi = interval_flows_lp(lines, n, lo, hi)
    ok, _ = authorize(f_lo, f_hi, caps, margin=0.0)
    assert adversarial_witness(lines, n, lo, hi, caps, margin=0.0) is not None
    assert not ok


def test_negative_load_box_witness():
    """负负荷箱（icaps 教训）下反例生成不崩溃。"""
    gen_p = np.array([2.0])
    load_p = np.array([-0.5, 0.5])
    lo, hi = injection_bounds(gen_p, load_p, [0], [1, 2], n_sub=3, load_eps=0.2)
    lines = [{"or": 0, "ex": 1, "x": 0.1}, {"or": 1, "ex": 2, "x": 0.1}]
    caps = np.array([10.0, 10.0])
    w = adversarial_witness(lines, 3, lo, hi, caps, margin=0.0)
    assert w is None  # 低负载下无越限
