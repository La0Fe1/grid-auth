"""src/models/gnn_policy.py 单元测试（纯张量，不加载 grid2op）。"""
import numpy as np
import pytest
import torch
from gymnasium import spaces

from src.models.gnn_policy import GNNGraphPolicy


def make_spaces(n_sub, n_line):
    obs = spaces.Dict({
        "node_feat": spaces.Box(-1e6, 1e6, (n_sub, 3), dtype=np.float32),
        "edge_feat": spaces.Box(-1e6, 1e6, (n_line, 2), dtype=np.float32),
    })
    act = spaces.MultiBinary(n_line)
    return obs, act


def make_dummy_obs(n_sub, n_line, B=2):
    return {
        "node_feat": torch.rand(B, n_sub, 3),
        "edge_feat": torch.rand(B, n_line, 2),
    }


def make_edge_index(n_line, n_sub):
    or_sub = np.arange(n_line) % n_sub
    ex_sub = (np.arange(n_line) + 1) % n_sub
    return np.stack([or_sub, ex_sub])


@pytest.fixture
def policy():
    n_sub, n_line = 14, 20
    obs_sp, act_sp = make_spaces(n_sub, n_line)
    pol = GNNGraphPolicy(obs_sp, act_sp, lambda _: 3e-4,
                         edge_index=make_edge_index(n_line, n_sub), hidden=32)
    return pol, n_sub, n_line


def test_forward_shapes(policy):
    pol, n_sub, n_line = policy
    obs = make_dummy_obs(n_sub, n_line)
    actions, values, log_prob = pol.forward(obs, deterministic=False)
    assert actions.shape == (2, n_line)
    assert values.shape == (2, 1)
    # SB3 Bernoulli 分布对动作维求和 → log_prob 为每样本一个标量
    assert log_prob.shape == (2,)


def test_evaluate_actions_shapes(policy):
    pol, n_sub, n_line = policy
    obs = make_dummy_obs(n_sub, n_line)
    actions = torch.randint(0, 2, (2, n_line)).float()
    value, log_prob, entropy = pol.evaluate_actions(obs, actions)
    assert value.shape == (2, 1)
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)


def test_param_count_invariance_to_n_line():
    """H3 机制核心性质：参数量与边数（电网规模）无关。"""
    obs1, act1 = make_spaces(14, 20)
    obs2, act2 = make_spaces(36, 59)
    p1 = GNNGraphPolicy(obs1, act1, lambda _: 3e-4,
                        edge_index=make_edge_index(20, 14), hidden=32)
    p2 = GNNGraphPolicy(obs2, act2, lambda _: 3e-4,
                        edge_index=make_edge_index(59, 36), hidden=32)
    assert p1.count_parameters() == p2.count_parameters()
    assert p1.count_parameters() < 100_000  # 远小于 100M 上限


def test_deterministic_same_input(policy):
    pol, n_sub, n_line = policy
    obs = make_dummy_obs(n_sub, n_line)
    pol.eval()
    with torch.no_grad():
        a1, v1, _ = pol.forward(obs, deterministic=True)
        a2, v2, _ = pol.forward(obs, deterministic=True)
    assert torch.equal(a1, a2) and torch.equal(v1, v2)


def test_encode_with_attention_shapes(policy):
    pol, n_sub, n_line = policy
    obs = make_dummy_obs(n_sub, n_line)
    logits, value, attns = pol.encode_with_attention(obs)
    assert logits.shape == (2, n_line)
    assert value.shape == (2, 1)
    assert len(attns) == 2
    assert attns[0].shape == (2, n_line, 2)  # 第 1 层 heads=2
    assert attns[1].shape == (2, n_line, 1)  # 第 2 层 heads=1
    # GAT 注意力来自 softmax，应非负
    assert bool((attns[0] >= 0).all()) and bool((attns[1] >= 0).all())


def test_attention_slice_layout(policy):
    """固化"自环边追加在原始边之后"的布局假设：注意力切片必须对应原始线路边。"""
    pol, n_sub, n_line = policy
    B = 2
    obs = make_dummy_obs(n_sub, n_line, B=B)
    x = obs["node_feat"].reshape(B * n_sub, -1)
    e = obs["edge_feat"].reshape(B * n_line, -1)
    dev = x.device
    node_offset = torch.arange(B, device=dev) * n_sub
    ei = pol.gnn_edge_index.to(dev).repeat(1, B)
    ei = ei + node_offset.repeat_interleave(n_line)[None, :]
    _, (attn_ei, alpha) = pol.conv1(x, ei, edge_attr=e, return_attention_weights=True)
    # 前 B*E 条注意力边必须与传入的原始边索引一致
    assert bool(torch.equal(attn_ei[:, : B * n_line], ei))
    # encode_with_attention 返回的第 1 层注意力 == alpha 原始边切片的 reshape（同一输入）
    _, _, attns = pol.encode_with_attention(obs)
    assert bool(torch.allclose(attns[0], alpha[: B * n_line].reshape(B, n_line, -1)))
