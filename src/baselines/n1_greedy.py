"""基线：N-1 贪心专家（对应 de Jong et al. 2024 的 N-1 感知专家思路，
用仿真搜索替代学习）。

每个决策步：当存在接近过载的线路时，枚举候选 toggle，
选择使后继状态 N-1 安全度提升最大（平局时最小化最大负载率）的单线动作；
否则不动。候选限制在 rho 最高的 max_candidates 条线以内控制仿真开销。

实现注意（2026-09-13）：全部仿真用单层组合动作（真实观测 → toggle ℓ + 断开 k），
禁止嵌套 simulate（grid2op 1.12.5 实测不可靠）。
"""
import numpy as np

from src.training.gen_n1_labels import (
    _outage_or_none, _simulate_or_none, _toggle_act, n1_score_of,
)


class N1GreedyAgent:
    """N-1 贪心专家：过载接近阈值时，选"安全准则下 N-1 改善最大"的单线 toggle。

    安全准则与屏蔽器一致：候选 post 必须无过载（ρ<0.95）且 N-1 安全度不恶化。
    act_every 控制干预频率（仿真开销控制）。
    """

    def __init__(self, env, rho_threshold=0.85, max_candidates=10, act_every=5):
        self.env = env
        self.rho_threshold = rho_threshold
        self.max_candidates = max_candidates
        self.act_every = act_every
        self.n_line = env.n_line
        self._step_counter = 0
        self._last_act = -act_every

    def _n1_score(self, obs_glop):
        return n1_score_of(obs_glop, self.env.env_glop)

    def act(self, obs):
        self._step_counter += 1
        action = np.zeros(self.n_line, dtype=np.float32)
        if self._step_counter - self._last_act < self.act_every:
            return action

        obs_glop = self.env.env_glop.get_obs()
        rho = obs_glop.rho
        if rho.max() < self.rho_threshold:
            return action
        self._last_act = self._step_counter

        glop = self.env.env_glop
        base = n1_score_of(obs_glop, glop)
        candidates = np.argsort(-rho)[: self.max_candidates]
        best_line, best_gain, best_rho = -1, 0, np.inf
        for lid in candidates:
            if obs_glop.line_status[lid] == 0:
                continue
            post = _simulate_or_none(obs_glop, _toggle_act(glop, lid))
            if post is None or not hasattr(post, "rho"):
                continue
            if post.rho.max() >= 0.95:
                continue  # 安全准则：不引入过载（与屏蔽器同一定义）
            gain = n1_score_of(obs_glop, glop, toggle_l=lid) - base
            post_rho = float(post.rho.max())
            if gain > best_gain or (gain == best_gain and post_rho < best_rho):
                best_line, best_gain, best_rho = int(lid), gain, post_rho
        if best_line >= 0 and best_gain > 0:
            action[best_line] = 1.0
        return action
