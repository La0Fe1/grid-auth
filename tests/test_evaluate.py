"""src/evaluation/evaluate.py 与 agents.py 单元测试（假环境，不加载 grid2op）。"""
import numpy as np
import pytest

from src.evaluation.agents import SB3GraphAgent
from src.evaluation.evaluate import (
    EpisodeStats, compare_paired, run_agent, run_episode, summarize,
)
from src.baselines.do_nothing import DoNothingAgent


class FakeGraphEnv:
    """确定性假环境：每幕 10 步，前 2 步 rho=1.5（触发过载计数）。"""

    def __init__(self, n_line=4, n_sub=3):
        self.n_line = n_line
        self.meta = {"n_sub": n_sub, "n_line": n_line}
        self._seed = 0

    def seed(self, s):
        self._seed = s
        return [s]

    def set_id(self, cid):
        self._cid = cid

    def reset(self, seed=None, options=None):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        done = self.t >= 10
        return self._obs(), 1.0, done, False, {"is_illegal": False}

    def _obs(self):
        rho = 1.5 if self.t <= 2 else 0.5
        return {
            "node_feat": np.zeros((self.meta["n_sub"], 3), dtype=np.float32),
            "edge_feat": np.full((self.n_line, 2), rho, dtype=np.float32),
        }


def test_run_episode_metrics():
    env = FakeGraphEnv()
    agent = DoNothingAgent(env=env)
    st = run_episode(agent, env)
    assert st.survival == 10
    assert st.reward == 10.0
    assert st.overflow_steps == 2
    assert st.n_toggles == 0
    assert st.n_illegal == 0


def test_run_agent_seeds():
    def ef():
        return FakeGraphEnv()

    stats = run_agent(lambda env: DoNothingAgent(env=env), ef, seeds=[0, 1, 2])
    assert len(stats) == 3
    assert all(s.survival == 10 for s in stats)


def test_summarize_fields():
    stats = [EpisodeStats(10, 10.0, 2, 0, 0, 0), EpisodeStats(8, 8.0, 1, 0, 0, 0)]
    d = summarize(stats)
    assert set(d) == {"survival", "reward", "overflow_steps", "n_illegal",
                      "n_blocks", "n_toggles", "n_proposed"}
    assert d["survival"]["mean"] == 9.0


def test_compare_paired_significant():
    a = [EpisodeStats(s, 0, 0, 0, 0, 0) for s in (15, 16, 17, 18, 19, 20, 21, 22)]
    b = [EpisodeStats(s, 0, 0, 0, 0, 0) for s in (5, 6, 7, 8, 9, 10, 11, 12)]
    stat, p = compare_paired(a, b)
    assert p < 0.05


def test_sb3_graph_agent_wrapper():
    class FakeModel:
        def predict(self, obs, deterministic):
            return np.zeros(4, dtype=np.float32), None

    agent = SB3GraphAgent(FakeModel())
    out = agent.act({"node_feat": None, "edge_feat": None})
    assert out.shape == (4,)
    assert out.dtype == np.float32
