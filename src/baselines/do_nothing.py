"""基线 B0：do-nothing 智能体（永远空动作）。"""
import numpy as np


class DoNothingAgent:
    def __init__(self, env=None, n_line=None):
        self.n_line = n_line if n_line is not None else env.n_line

    def act(self, obs):
        return np.zeros(self.n_line, dtype=np.float32)
