"""src/models/n1_gate.py 与 src/models/shield.py 单元测试（纯张量，不加载 grid2op）。"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.models.n1_gate import N1Gate
from src.models.shield import N1Shield


def make_edge_index(n_line, n_sub):
    or_sub = np.arange(n_line) % n_sub
    ex_sub = (np.arange(n_line) + 1) % n_sub
    return np.stack([or_sub, ex_sub])


def make_obs(n_sub, n_line, statuses=None):
    if statuses is None:
        statuses = np.ones(n_line)
    return {
        "node_feat": np.random.rand(1, n_sub, 3).astype(np.float32),
        "edge_feat": np.stack([np.random.rand(n_line), statuses], axis=1)[None].astype(np.float32),
    }


@pytest.fixture
def gate():
    n_line, n_sub = 20, 14
    return N1Gate(make_edge_index(n_line, n_sub), hidden=32), n_sub, n_line


class FixedGate(nn.Module):
    """返回固定概率的假门，用于屏蔽器逻辑测试。"""

    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x_node, x_edge):
        B, E = x_edge.shape[:2]
        return torch.full((B, E), self.value)


def test_gate_forward_shapes(gate):
    g, n_sub, n_line = gate
    node = torch.rand(2, n_sub, 3)
    edge = torch.rand(2, n_line, 2)
    p = g(node, edge)
    assert p.shape == (2, n_line)
    assert bool((p >= 0).all()) and bool((p <= 1).all())  # sigmoid 输出


def test_gate_param_invariance_to_n_line():
    g1 = N1Gate(make_edge_index(20, 14), hidden=32)
    g2 = N1Gate(make_edge_index(59, 36), hidden=32)
    assert g1.count_parameters() == g2.count_parameters()


def test_shield_blocks_low_prob():
    n_sub, n_line = 14, 20
    shield = N1Shield(FixedGate(0.0), alpha=0.0)
    obs = make_obs(n_sub, n_line, statuses=np.ones(n_line))
    proposal = np.zeros(n_line); proposal[3] = 1.0
    allowed, info = shield.filter(obs, proposal)
    assert allowed[3] == 0.0  # 被屏蔽
    assert 3 in info["blocked"]
    assert len(info["blocked_p"]) == 1


def test_shield_allows_high_prob():
    n_sub, n_line = 14, 20
    shield = N1Shield(FixedGate(1.0), alpha=0.0)
    obs = make_obs(n_sub, n_line, statuses=np.ones(n_line))
    proposal = np.zeros(n_line); proposal[3] = 1.0
    allowed, info = shield.filter(obs, proposal)
    assert allowed[3] == 1.0
    assert info["blocked"] == []


def test_shield_threshold_alpha():
    """alpha=0.1 时 p=0.95 ≥ 0.9 放行；alpha=0.0 时 p=0.95 < 1.0 屏蔽。"""
    n_sub, n_line = 14, 20
    obs = make_obs(n_sub, n_line, statuses=np.ones(n_line))
    proposal = np.zeros(n_line); proposal[3] = 1.0
    allowed_loose, _ = N1Shield(FixedGate(0.95), alpha=0.1).filter(obs, proposal)
    allowed_strict, _ = N1Shield(FixedGate(0.95), alpha=0.0).filter(obs, proposal)
    assert allowed_loose[3] == 1.0
    assert allowed_strict[3] == 0.0


def test_shield_skips_off_lines():
    """已断开线路的 toggle（重连请求）不做门控，并如实标注。"""
    n_sub, n_line = 14, 20
    shield = N1Shield(FixedGate(0.0), alpha=0.0)
    obs = make_obs(n_sub, n_line, statuses=np.ones(n_line))
    obs["edge_feat"][0, 3, 1] = 0.0  # 线 3 已断开
    proposal = np.zeros(n_line); proposal[3] = 1.0
    allowed, info = shield.filter(obs, proposal)
    assert allowed[3] == 1.0  # 不门控 → 放行（交给环境合法性）
    assert 3 in info["gated_off_lines"]


def test_shield_empty_proposal():
    n_sub, n_line = 14, 20
    shield = N1Shield(FixedGate(0.0), alpha=0.0)
    obs = make_obs(n_sub, n_line, statuses=np.ones(n_line))
    allowed, info = shield.filter(obs, np.zeros(n_line))
    assert allowed.sum() == 0.0
    assert info["blocked"] == []


def test_shield_attribution_info():
    """屏蔽归因信息完整：每条请求线可查到 p 与阈值（解释柱 H2 的数据基础）。"""
    n_sub, n_line = 14, 20
    shield = N1Shield(FixedGate(0.3), alpha=0.0)
    obs = make_obs(n_sub, n_line, statuses=np.ones(n_line))
    proposal = np.zeros(n_line); proposal[2] = 1.0; proposal[5] = 1.0
    allowed, info = shield.filter(obs, proposal)
    assert sorted(info["blocked"]) == [2, 5]
    assert info["p"].shape == (n_line,)
    assert info["threshold"] == 1.0
    assert np.allclose(info["p"][[2, 5]], 0.3)


def test_hybrid_shield_requires_env():
    """混合屏蔽器必须传入 grid2op 环境做精确终审。"""
    from src.models.shield import HybridShield

    shield = HybridShield(FixedGate(1.0), alpha=0.0)
    with pytest.raises(ValueError):
        shield.filter(make_obs(14, 20), np.zeros(20))
