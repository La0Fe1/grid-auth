"""智能体封装：把训练好的 SB3 模型/裸策略包装成 act(obs) 接口，供评估模块使用。"""
import numpy as np


class SB3GraphAgent:
    """SB3 PPO 模型（GNNGraphPolicy）的确定性动作接口。"""

    def __init__(self, model):
        self.model = model

    def act(self, obs):
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)


class PolicyAgent:
    """裸 ActorCriticPolicy 的确定性动作接口（跨算例迁移策略用，exp_005）。"""

    def __init__(self, policy):
        self.policy = policy

    def act(self, obs):
        action, _ = self.policy.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.float32)
