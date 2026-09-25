"""Grid2Op 图 gym 环境封装（由遗留 gnn_env.py / n1_env.py 迁移，动作语义经 grid2op 源码与实测双重确认）。

动作语义（重要）：
- 动作空间 = MultiBinary(n_line)：每条线路一个 toggle 请求。
- 底层 `act.change_line_status = [ids]` 的语义 = toggle（grid2op baseAction.py
  line_change_status 文档：True 表示"改变状态"——连着断、断着连）。
- 非法 toggle（例如某些线路禁止重连）由 grid2op 环境拒绝
  （info["is_illegal"]=True 时动作未生效），不额外报错。

数据目录（AGENTS.md 0.1 合规）：
grid2op 默认把数据集下载到用户主目录 ~/data_grid2op（违反项目隔离条款），
本模块统一使用项目内 data/grid2op_data/ 作为 dataset_path。
"""
import os

import gymnasium as gym
import numpy as np
from grid2op import make
from grid2op.Reward import BaseReward
from gymnasium import spaces
from lightsim2grid import LightSimBackend

from src.utils.graph import (
    EDGE_DIM, NODE_DIM, RICH_EDGE_DIM, RICH_NODE_DIM, build_graph_meta,
    obs_to_graph, obs_to_graph_v2,
)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_DATASET_PATH = os.path.join(_PROJECT_ROOT, "data", "grid2op_data")


def _resolve_dataset(case, dataset_path):
    """项目内已有解压数据集就用本地路径（0.1 合规）；否则退回环境名（内置测试环境）。

    注意：全新机器上必须先运行 data/get_data.py 把数据集解压到项目内，
    否则退回环境名时会触发 grid2op 默认下载（写入用户主目录，违反 0.1）。
    """
    local = os.path.join(dataset_path, case)
    return local if os.path.isdir(local) else case


class SurvivalReward(BaseReward):
    """存活 + 温和过载惩罚 + 非法动作惩罚：安全 +1，每条过载线路 -0.2，
    非法动作 -1，游戏结束 -10。

    设计理由：
    - 安全时保持稳定 +1（避免连续边际奖励噪声导致训练发散）；
    - 过载时温和离散惩罚，引导 agent 消除过载而不是躺着；
    - 非法动作 -1（2026-09-13 修订）：无此惩罚时策略会漂移到
      "全 toggle"不动点（MLP 基线实测：每步 7 条 toggle 全部非法被拒，
      行为退化为 do-nothing），grid2op 官方奖励同款做法。
    """

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if is_done:
            return -10.0
        if is_illegal:
            return -1.0
        obs = env.get_obs()
        if obs is None:
            return 1.0
        rho = getattr(obs, "rho", None)
        if rho is None:
            return 1.0
        n_overflow = float((rho > 1.0).sum())
        return 1.0 - 0.2 * n_overflow


class Grid2OpGraphEnv(gym.Env):
    """把 Grid2Op 包装成输出图结构的 gym 环境，动作 = 每条线路 toggle 请求。

    观测：{"node_feat": (n_sub, 3), "edge_feat": (n_line, 2)}（见 src/utils/graph.py）。
    """

    def __init__(self, case="rte_case14_realistic", test=False,
                 reward_class=SurvivalReward, dataset_path=DEFAULT_DATASET_PATH,
                 sample_chronics=True, features="basic", attack_enabled=True,
                 attack_budget_scale=1.0):
        kwargs = {}
        if not attack_enabled:  # 课程学习第一阶段的"无故障"环境
            kwargs = {"opponent_init_budget": 0, "opponent_budget_per_ts": 0}
        elif attack_budget_scale != 1.0:
            # 更强对手（FA-002 改进轴）：读默认预算后按倍率放大
            probe = make(dataset=_resolve_dataset(case, dataset_path), test=test,
                         backend=LightSimBackend(), reward_class=reward_class)
            init_b = float(getattr(probe, "opponent_init_budget",
                                   getattr(probe, "_opponent_init_budget", 283.0)))
            per_ts = float(getattr(probe, "opponent_budget_per_ts",
                                   getattr(probe, "_opponent_budget_per_ts", 0.0333)))
            kwargs = {"opponent_init_budget": init_b * attack_budget_scale,
                      "opponent_budget_per_ts": per_ts * attack_budget_scale}
        self.env_glop = make(dataset=_resolve_dataset(case, dataset_path), test=test,
                             backend=LightSimBackend(), reward_class=reward_class, **kwargs)
        self.meta = build_graph_meta(self.env_glop)
        self.features = features
        self._graph_fn = obs_to_graph_v2 if features == "rich" else obs_to_graph
        self.node_dim = RICH_NODE_DIM if features == "rich" else NODE_DIM
        self.edge_dim = RICH_EDGE_DIM if features == "rich" else EDGE_DIM
        # 场景采样：训练时随机采样（种子化 RNG）；评估时由 set_id 显式指定
        self._sample_chronics = sample_chronics
        self._rng = np.random.RandomState(0)
        self._explicit_id = None
        subpaths = getattr(self.env_glop.chronics_handler.real_data, "subpaths", [])
        self._n_chronics = len(subpaths)

        n_sub, n_line = self.meta["n_sub"], self.meta["n_line"]
        self.observation_space = spaces.Dict(
            {
                "node_feat": spaces.Box(low=-1e6, high=1e6, shape=(n_sub, self.node_dim),
                                        dtype=np.float32),
                "edge_feat": spaces.Box(low=-1e6, high=1e6, shape=(n_line, self.edge_dim),
                                        dtype=np.float32),
            }
        )
        # 每条线路一个 toggle 请求（0=不动，1=请求翻转通断状态）
        self.action_space = spaces.MultiBinary(n_line)

    @property
    def edge_index(self):
        return self.meta["edge_index"]

    @property
    def n_line(self):
        return self.meta["n_line"]

    def seed(self, seed=None):
        """设置内部场景采样 RNG（训练用）。评估场景由 set_id 显式指定。"""
        if seed is not None:
            self._rng = np.random.RandomState(seed)
        return [seed]

    def set_id(self, chronics_id):
        """显式指定下一幕的场景（评估口径：seed → 场景 id 由调用方决定）。

        优先级：显式 id > 随机采样；显式 id 在下次 reset 用完后清除，
        之后 reset 恢复随机采样（训练语义）。
        """
        self._explicit_id = chronics_id

    def reset(self, *, seed=None, options=None):
        if self._explicit_id is not None:
            self.env_glop.set_id(self._explicit_id)
            self._explicit_id = None
        elif self._sample_chronics and self._n_chronics > 0:
            self.env_glop.set_id(int(self._rng.randint(0, self._n_chronics)))
        obs = self.env_glop.reset()
        nf, ef = self._graph_fn(obs, self.meta)
        return {"node_feat": nf, "edge_feat": ef}, {}

    def step(self, action):
        # action: (n_line,) 0/1，请求 toggle 哪些线路；非法者由环境拒绝
        action = np.asarray(action)
        if action.ndim == 2:  # 兼容 VecEnv 传入的 (1, n_line)
            action = action[0]
        toggle_lines = [int(i) for i, v in enumerate(action) if v > 0.5]
        act = self.env_glop.action_space({})
        if toggle_lines:
            act.change_line_status = toggle_lines
        obs, reward, done, info = self.env_glop.step(act)
        nf, ef = self._graph_fn(obs, self.meta)
        return {"node_feat": nf, "edge_feat": ef}, float(reward), bool(done), False, info


class FlattenGraphEnv(gym.Wrapper):
    """把图观测拍平成向量（MLP 基线 B1 用）。

    观测维度 = n_sub*3 + n_line*2，与算例绑定 → 无跨算例能力
    （这正是三柱方法与之的核心差别，H3 对照）。
    """

    def __init__(self, env):
        super().__init__(env)
        n_sub, n_line = env.meta["n_sub"], env.meta["n_line"]
        self.observation_space = spaces.Box(
            -1e6, 1e6, (n_sub * 3 + n_line * 2,), dtype=np.float32)
        self.action_space = env.action_space

    def set_id(self, chronics_id):
        self.env.set_id(chronics_id)

    def seed(self, seed=None):
        # gymnasium 1.x 移除了 Wrapper.seed，须显式委托（SB3 训练时会调用）
        return self.env.seed(seed)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._flatten(obs), info

    def step(self, action):
        obs, r, d, t, info = self.env.step(action)
        return self._flatten(obs), r, d, t, info

    @staticmethod
    def _flatten(obs):
        return np.concatenate(
            [np.asarray(obs["node_feat"]).ravel(), np.asarray(obs["edge_feat"]).ravel()]
        ).astype(np.float32)


class MaskedGraphEnv(gym.Wrapper):
    """候选掩码环境（FA-002 第五改进轴）：动作空间约减。

    机制：动作进入前，从当前观测生成候选集 = {连通且 rho > rho_th 的线路，
    最多 max_candidates 条（按 rho 降序）}，把提案中非候选位置清零。
    依据：L2RPN 获胜方案共识（只在应力线路附近操作）；
    同时消除两种退化不动点——全 toggle（至多 max_candidates 条）与永不动作
    （无应力时不动作是合理行为，有应力时探索集中在小候选集上）。
    """

    def __init__(self, env, rho_th=0.9, max_candidates=5):
        super().__init__(env)
        self.rho_th = rho_th
        self.max_candidates = max_candidates
        self._last_obs = None

    @property
    def edge_index(self):
        return self.env.edge_index

    @property
    def n_line(self):
        return self.env.n_line

    @property
    def node_dim(self):
        return self.env.node_dim

    @property
    def edge_dim(self):
        return self.env.edge_dim

    def seed(self, seed=None):
        # gymnasium 1.x 移除了 Wrapper.seed，须显式委托（SB3 训练时会调用）
        return self.env.seed(seed)

    def set_id(self, chronics_id):
        self.env.set_id(chronics_id)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_obs = obs
        return obs, info

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32)
        orig = action_arr.copy()
        if action_arr.ndim == 2:
            action_arr = action_arr[0]
        ef = np.asarray(self._last_obs["edge_feat"])
        rho, status = ef[:, 0], ef[:, 1]
        cand = np.where((status > 0.5) & (rho > self.rho_th))[0]
        if len(cand) > self.max_candidates:
            cand = cand[np.argsort(-rho[cand])[: self.max_candidates]]
        mask = np.zeros_like(action_arr)
        mask[cand] = 1.0
        masked = action_arr * mask
        obs, r, d, t, info = self.env.step(masked)
        info = dict(info)
        info["mask"] = {"candidates": [int(c) for c in cand],
                        "n_masked": int((orig != masked).sum())}
        self._last_obs = obs
        return obs, r, d, t, info


class ShieldedGraphEnv(gym.Wrapper):
    """在动作进入 grid2op 前用 N1Shield 过滤（硬屏蔽训练/评估环境，exp_003 用）。

    step() 流程：当前图观测 → 屏蔽器 filter（硬删除不安全 toggle 请求）→ grid2op step。
    屏蔽归因信息并入 info["shield"]，供评估统计（被屏蔽率、归因案例）。
    """

    def __init__(self, env, shield):
        super().__init__(env)
        self.shield = shield
        self._last_graph_obs = None

    @property
    def n_line(self):
        return self.env.n_line

    @property
    def edge_index(self):
        return self.env.edge_index

    def set_id(self, chronics_id):
        self.env.set_id(chronics_id)

    def seed(self, seed=None):
        # gymnasium 1.x 移除了 Wrapper.seed，须显式委托（SB3 训练时会调用）
        return self.env.seed(seed)

    def reset(self, *, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        self._last_graph_obs = obs
        return obs, info

    def step(self, action):
        action_arr = np.asarray(action, dtype=np.float32)
        proposal = action_arr[0] if action_arr.ndim == 2 else action_arr
        try:
            allowed, shield_info = self.shield.filter(self._last_graph_obs, proposal,
                                                      env=self.env)
        except TypeError:
            # 兼容不需要环境参数的纯学习门屏蔽器
            allowed, shield_info = self.shield.filter(self._last_graph_obs, proposal)
        shield_info = dict(shield_info)
        shield_info["proposed"] = int(np.asarray(proposal).sum())  # 提案数（削减率分母）
        obs, reward, done, trunc, info = self.env.step(allowed)
        info = dict(info)
        info["shield"] = shield_info
        self._last_graph_obs = obs
        return obs, reward, done, trunc, info


class N1OptEnv(gym.Env):
    """单步 N-1 拓扑优化环境：观测图 → 一次 toggle → N-1 安全度改善 reward。

    N-1 安全度定义：对每条当前连通的线路，模拟它故障断开，若其余线路均不过载
    （rho < 1）则记 1 分。安全度 = 得分的线路数。
    reward = 动作后状态的安全度 − 动作前（reset 采样状态）的安全度。

    注意：reward 在动作后的时间步观测上计算（负荷随时间演进），
    与遗留实现保持一致；论文中该环境仅用于生成 N-1 门的训练标签与评估。
    """

    def __init__(self, case="rte_case14_realistic", seed=0, dataset_path=DEFAULT_DATASET_PATH):
        self.env_glop = make(dataset=_resolve_dataset(case, dataset_path),
                             backend=LightSimBackend())
        self.meta = build_graph_meta(self.env_glop)
        self.n_line = self.meta["n_line"]
        self.n_sub = self.meta["n_sub"]

        self.observation_space = spaces.Dict(
            {
                "node_feat": spaces.Box(-1e6, 1e6, (self.n_sub, 3), dtype=np.float32),
                "edge_feat": spaces.Box(-1e6, 1e6, (self.n_line, 2), dtype=np.float32),
            }
        )
        self.action_space = spaces.MultiBinary(self.n_line)
        self.rng = np.random.RandomState(seed)
        self._base_n1 = 0

    @property
    def edge_index(self):
        return self.meta["edge_index"]

    def _n1_score(self, obs):
        """N-1 安全度：多少条连通线路断开后其余线路均不过载。"""
        safe = 0
        for lid in range(self.n_line):
            if obs.line_status[lid] == 0:
                continue
            s = obs.simulate(self.env_glop.action_space({"set_line_status": [(lid, -1)]}))
            so = s[0] if s else None
            if so is not None and so.rho.max() < 1.0:
                safe += 1
        return safe

    def reset(self, *, seed=None, options=None):
        self.env_glop.reset()
        # 随机前进到某个时间步，采样不同负荷状态
        step = self.rng.randint(0, 500)
        self.env_glop.fast_forward_chronics(step)
        obs = self.env_glop.get_obs()
        self._base_n1 = self._n1_score(obs)
        nf, ef = obs_to_graph(obs, self.meta)
        return {"node_feat": nf, "edge_feat": ef}, {}

    def step(self, action):
        toggle_lines = [int(i) for i, v in enumerate(action) if v > 0.5]
        act = self.env_glop.action_space({})
        if toggle_lines:
            act.change_line_status = toggle_lines
        obs, _, _, _ = self.env_glop.step(act)
        n1_after = self._n1_score(obs)
        reward = n1_after - self._base_n1  # N-1 改善量（可正可负）
        nf, ef = obs_to_graph(obs, self.meta)
        return {"node_feat": nf, "edge_feat": ef}, float(reward), True, False, {"n1": n1_after}
