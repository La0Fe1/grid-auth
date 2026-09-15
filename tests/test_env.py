"""src/utils/env.py 单元测试（奖励函数）+ 环境集成测试（加载真实 grid2op）。"""
import numpy as np
import pytest
import torch
import torch.nn as nn

from src.utils.env import SurvivalReward


class FakeObs:
    def __init__(self, rho=None):
        self.rho = np.asarray(rho, dtype=np.float32) if rho is not None else None


class FakeGlopEnv:
    def __init__(self, obs):
        self._obs = obs

    def get_obs(self):
        return self._obs


def call_reward(obs, is_done=False, is_illegal=False):
    r = SurvivalReward()
    return r.__call__(None, FakeGlopEnv(obs), False, is_done, is_illegal, False)


def test_reward_game_over():
    assert call_reward(FakeObs([1.2]), is_done=True) == -10.0


def test_reward_illegal_action():
    """非法动作 -1（防'全 toggle'退化不动点，2026-09-13 修订）。"""
    assert call_reward(FakeObs([0.5]), is_illegal=True) == -1.0


def test_reward_safe_state():
    assert call_reward(FakeObs([0.5, 0.9, 0.99])) == 1.0


def test_reward_two_overflows():
    # 1.0 - 0.2*2 = 0.6
    assert np.isclose(call_reward(FakeObs([0.5, 1.1, 1.3])), 0.6)


def test_reward_no_obs():
    assert call_reward(None) == 1.0


def test_reward_no_rho():
    assert call_reward(FakeObs(rho=None)) == 1.0


@pytest.mark.integration
def test_graph_env_reset_step_shapes():
    from src.utils.env import Grid2OpGraphEnv

    env = Grid2OpGraphEnv(case="educ_case14_storage", test=True)
    obs, _ = env.reset()
    assert obs["node_feat"].shape == (env.meta["n_sub"], 3)
    assert obs["edge_feat"].shape == (env.meta["n_line"], 2)

    act = np.zeros(env.n_line, dtype=np.float32)
    obs2, rew, done, trunc, info = env.step(act)
    assert obs2["node_feat"].shape == obs["node_feat"].shape
    assert isinstance(rew, float)
    assert isinstance(done, bool) and trunc is False
    assert "is_illegal" in info


@pytest.mark.integration
def test_graph_env_toggle_semantics():
    """toggle 语义：动作合法则线路状态必翻转；非法则 info 标记且状态不变。"""
    from src.utils.env import Grid2OpGraphEnv

    env = Grid2OpGraphEnv(case="educ_case14_storage", test=True)
    obs0, _ = env.reset()
    act = np.zeros(env.n_line, dtype=np.float32)
    act[3] = 1.0
    _, _, _, _, info = env.step(act)
    status_now = env.env_glop.get_obs().line_status[3]
    status_before = obs0["edge_feat"][3, 1]
    if info.get("is_illegal"):
        assert status_now == status_before
    else:
        assert status_now != status_before


@pytest.mark.integration
def test_n1_env_step_bounds():
    from src.utils.env import N1OptEnv

    env = N1OptEnv(case="rte_case14_realistic")
    obs, _ = env.reset()
    act = np.zeros(env.n_line, dtype=np.float32)
    _, reward, done, trunc, info = env.step(act)
    assert done is True
    assert -env.n_line <= reward <= env.n_line
    assert 0 <= info["n1"] <= env.n_line


@pytest.mark.integration
def test_env_seed():
    from src.utils.env import Grid2OpGraphEnv

    env = Grid2OpGraphEnv(case="educ_case14_storage", test=True)
    assert env.seed(42) == [42]


@pytest.mark.integration
def test_env_set_id_selects_scenario():
    from src.utils.env import Grid2OpGraphEnv

    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    env.set_id(5)
    env.reset()
    assert str(env.env_glop.chronics_handler.get_id()).endswith("005")


@pytest.mark.integration
def test_env_seeded_sampling_deterministic():
    """同种子 → 同场景采样序列（训练可复现性的基础）。"""
    from src.utils.env import Grid2OpGraphEnv

    e1 = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    e2 = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    e1.seed(7)
    e2.seed(7)
    e1.reset()
    e2.reset()
    assert e1.env_glop.chronics_handler.get_id() == e2.env_glop.chronics_handler.get_id()


@pytest.mark.integration
def test_wrapper_seed_and_set_id_passthrough():
    """gymnasium 1.x 移除了 Wrapper.seed/set_id，须显式委托（SB3 训练依赖）。"""
    from src.models.shield import N1Shield
    from src.utils.env import FlattenGraphEnv, Grid2OpGraphEnv, ShieldedGraphEnv

    class ZeroGate(nn.Module):
        def forward(self, x_node, x_edge):
            return torch.zeros(x_node.shape[0], x_edge.shape[1])

    flat = FlattenGraphEnv(Grid2OpGraphEnv(case="educ_case14_storage", test=True))
    assert flat.seed(5) == [5]
    flat.set_id(0)  # 不应抛异常

    sh = ShieldedGraphEnv(Grid2OpGraphEnv(case="educ_case14_storage", test=True),
                          N1Shield(ZeroGate()))
    assert sh.seed(5) == [5]
    sh.set_id(0)


@pytest.mark.integration
def test_shielded_env_blocks_action():
    """屏蔽器在动作进入 grid2op 前生效：被屏蔽的 toggle 不发生，且归因入 info。"""
    from src.models.shield import N1Shield
    from src.utils.env import Grid2OpGraphEnv, ShieldedGraphEnv

    class ZeroGate(nn.Module):
        def forward(self, x_node, x_edge):
            return torch.zeros(x_node.shape[0], x_edge.shape[1])

    base = Grid2OpGraphEnv(case="educ_case14_storage", test=True)
    env = ShieldedGraphEnv(base, N1Shield(ZeroGate()))
    env.reset()
    act = np.zeros(env.n_line, dtype=np.float32)
    act[3] = 1.0
    _, _, _, _, info = env.step(act)
    assert info["shield"]["blocked"] == [3]
    assert base.env_glop.get_obs().line_status[3] == 1  # 屏蔽生效，线路未被 toggle

    # 对照：无屏蔽环境同一动作应翻转线路（若合法）
    base2 = Grid2OpGraphEnv(case="educ_case14_storage", test=True)
    base2.reset()
    act2 = np.zeros(base2.n_line, dtype=np.float32)
    act2[3] = 1.0
    _, _, _, _, info2 = base2.step(act2)
    if not info2.get("is_illegal"):
        assert base2.env_glop.get_obs().line_status[3] == 0
