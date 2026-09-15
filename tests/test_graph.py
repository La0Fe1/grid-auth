"""src/utils/graph.py 单元测试。

普通测试用 FakeObs（不加载 grid2op，秒级）；integration 测试加载真实环境。
"""
import numpy as np
import pytest

from src.utils.graph import (
    EDGE_DIM, NODE_DIM, RICH_EDGE_DIM, RICH_NODE_DIM, build_graph_meta,
    obs_to_graph, obs_to_graph_v2,
)


class FakeObs:
    """最小假观测：只提供 obs_to_graph 需要的属性。"""

    def __init__(self, gen_p, load_p, v_or, v_ex, rho, line_status,
                 gen_to_sub=None, load_to_sub=None,
                 line_or_to_subid=None, line_ex_to_subid=None):
        self.n_sub = max(max(gen_to_sub) if gen_to_sub else 0,
                         max(load_to_sub) if load_to_sub else 0) + 1
        self.n_line = len(rho)
        self.gen_to_subid = np.asarray(gen_to_sub if gen_to_sub is not None else [])
        self.load_to_subid = np.asarray(load_to_sub if load_to_sub is not None else [])
        self.gen_p = np.asarray(gen_p, dtype=np.float32)
        self.load_p = np.asarray(load_p, dtype=np.float32)
        self.v_or = np.asarray(v_or, dtype=np.float32)
        self.v_ex = np.asarray(v_ex, dtype=np.float32)
        self.rho = np.asarray(rho, dtype=np.float32)
        self.line_status = np.asarray(line_status, dtype=np.float32)
        self.line_or_to_subid = np.asarray(
            line_or_to_subid if line_or_to_subid is not None else list(range(len(rho))))
        self.line_ex_to_subid = np.asarray(
            line_ex_to_subid if line_ex_to_subid is not None else [0] * len(rho))


def make_meta(n_sub, n_line):
    or_sub = np.arange(n_line) % n_sub
    ex_sub = (np.arange(n_line) + 1) % n_sub
    return dict(n_sub=n_sub, n_line=n_line,
                edge_index=np.stack([or_sub, ex_sub]).astype(np.int64),
                gen_to_sub=np.array([0, 1]), load_to_sub=np.array([0, 1]))


def make_obs(n_line=3):
    return FakeObs(
        gen_p=[80.0, 20.0], load_p=[50.0, 50.0],
        v_or=[1.0] * n_line, v_ex=[1.0] * n_line,
        rho=[0.5, 0.9, 1.2], line_status=[1.0, 1.0, 0.0],
        gen_to_sub=[0, 1], load_to_sub=[0, 1],
    )


def test_shapes():
    meta = make_meta(n_sub=3, n_line=3)
    nf, ef = obs_to_graph(make_obs(), meta)
    assert nf.shape == (3, NODE_DIM)
    assert ef.shape == (3, EDGE_DIM)
    assert nf.dtype == np.float32 and ef.dtype == np.float32


def test_gen_load_normalized():
    """发电/负荷按全网总量归一：节点特征两列各求和 ≈ 1。"""
    meta = make_meta(n_sub=3, n_line=3)
    nf, _ = obs_to_graph(make_obs(), meta)
    assert np.isclose(nf[:, 0].sum(), 1.0)
    assert np.isclose(nf[:, 1].sum(), 1.0)


def test_scale_invariance():
    """关键性质（H3 机制基础）：全网发/负荷等比放大，归一化特征不变。"""
    meta = make_meta(n_sub=3, n_line=3)
    nf1, _ = obs_to_graph(make_obs(), meta)
    obs2 = make_obs()
    obs2.gen_p *= 2.5
    obs2.load_p *= 2.5
    nf2, _ = obs_to_graph(obs2, meta)
    assert np.allclose(nf1[:, :2], nf2[:, :2], atol=1e-6)


def test_zero_generation_handled():
    """全零发电不除零、不崩溃。"""
    meta = make_meta(n_sub=3, n_line=3)
    obs = make_obs()
    obs.gen_p[:] = 0.0
    obs.load_p[:] = 0.0
    nf, _ = obs_to_graph(obs, meta)
    assert np.all(nf[:, 0] == 0.0) and np.all(nf[:, 1] == 0.0)


def test_voltage_averaged_per_substation():
    """电压列 = 连到该站的线路端电压均值（v_or 计 or 侧，v_ex 计 ex 侧）。"""
    meta = make_meta(n_sub=2, n_line=4)  # or: 0,1,0,1; ex: 1,0,1,0
    obs = make_obs(n_line=4)
    obs.v_or = [1.0, 2.0, 3.0, 4.0]
    obs.v_ex = [10.0, 20.0, 30.0, 40.0]
    nf, _ = obs_to_graph(obs, meta)
    # 站0: or 侧线 0,2 的 v_or + ex 侧线 1,3 的 v_ex → (1+3+20+40)/4
    assert np.isclose(nf[0, 2], (1 + 3 + 20 + 40) / 4)
    # 站1: or 侧线 1,3 的 v_or + ex 侧线 0,2 的 v_ex → (2+4+10+30)/4
    assert np.isclose(nf[1, 2], (2 + 4 +10 + 30) / 4)


def test_edge_features_match():
    meta = make_meta(n_sub=3, n_line=3)
    _, ef = obs_to_graph(make_obs(), meta)
    assert np.allclose(ef[:, 0], [0.5, 0.9, 1.2])
    assert np.allclose(ef[:, 1], [1.0, 1.0, 0.0])


def test_obs_to_graph_v2_shapes_and_margin():
    """丰富特征版：节点 5 维、边 5 维；边裕度列 = 1 - rho；冷却列归一到 [0,1]。"""
    meta = make_meta(n_sub=3, n_line=3)
    nf, ef = obs_to_graph_v2(make_obs(), meta)
    assert nf.shape == (3, RICH_NODE_DIM)
    assert ef.shape == (3, RICH_EDGE_DIM)
    assert np.allclose(ef[:, 2], 1.0 - ef[:, 0])  # margin = 1 - rho
    assert (nf[:, 3] >= 0).all()  # |v-1| 均值非负
    assert (ef[:, 4] >= 0).all() and (ef[:, 4] <= 1.0).all()  # 冷却归一化
    # 前 3 列与基础版一致
    nf_b, _ = obs_to_graph(make_obs(), meta)
    assert np.allclose(nf[:, :3], nf_b)


@pytest.mark.integration
def test_build_graph_meta_real_env():
    grid2op = pytest.importorskip("grid2op")
    from grid2op import make
    from lightsim2grid import LightSimBackend
    env = make("educ_case14_storage", test=True, backend=LightSimBackend())
    meta = build_graph_meta(env)
    assert meta["n_line"] == meta["edge_index"].shape[1]
    assert meta["edge_index"].min() >= 0
    assert meta["edge_index"].max() < meta["n_sub"]
