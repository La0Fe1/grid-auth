"""硬安全屏蔽器（安全柱的执行层）。

机制（对应创新性声明①"把不安全动作从动作集删除而非扣分"）：
- 对每条请求 toggle 且**当前连通**的线路 ℓ，N1Gate 估计"该 toggle 满足可操作安全准则
  （无过载留裕度 + N-1 安全度不恶化）"的概率 p_ℓ；
- p_ℓ < 1 − alpha（alpha = 误放预算，验证集校准）→ 屏蔽该请求（硬约束，动作集中删除）；
- 被屏蔽线路附归因信息（p_ℓ 与阈值），供解释柱（H2：屏蔽器回报"哪条支路不安全"）；
- 对当前**已断开**线路（toggle=重连）不做门控，交由 grid2op 合法性检查
  （v1 范围：门只学习"断开可行性"语义，见 AGENTS.md 决策记录）。

注意：门的训练标签只覆盖"断开"动作（src/training/gen_n1_labels.py），
因此 p_ℓ 对重连请求无定义——代码按"不门控"处理并在 info 中如实标注。
"""
import numpy as np
import torch


class N1Shield:
    def __init__(self, gate, alpha=0.0, device="cpu"):
        """gate: N1Gate（eval 状态）；alpha: 误放预算（p_ℓ ≥ 1−alpha 才放行）。"""
        self.gate = gate
        self.alpha = float(alpha)
        self.device = device
        self.threshold = 1.0 - self.alpha

    @torch.no_grad()
    def filter(self, obs, proposal):
        """obs: {"node_feat": (1,N,3), "edge_feat": (1,E,2)}；proposal: (E,) 0/1。

        Returns:
            allowed: (E,) float32 —— 与原 proposal 相同，被屏蔽位置 0
            info: dict —— {"blocked": [line_id...], "blocked_p": [...],
                           "p": (E,) 各线安全概率, "threshold": 1-alpha,
                           "gated_off_lines": [对重连请求未门控的线 id]}
        """
        proposal = np.asarray(proposal, dtype=np.float32).copy()
        if proposal.ndim == 2:  # 兼容 VecEnv 传入的 (1, E)
            proposal = proposal[0]
        allowed = proposal.copy()
        if proposal.sum() == 0:
            return allowed, {"blocked": [], "blocked_p": [], "p": None,
                             "threshold": self.threshold, "gated_off_lines": []}

        node = np.asarray(obs["node_feat"], dtype=np.float32)
        edge = np.asarray(obs["edge_feat"], dtype=np.float32)
        if node.ndim == 2:  # 兼容无 batch 维的观测（env 直接输出）
            node = node[None]
        if edge.ndim == 2:
            edge = edge[None]
        node_t = torch.as_tensor(node, device=self.device)
        edge_t = torch.as_tensor(edge, device=self.device)
        p = self.gate(node_t, edge_t)[0].cpu().numpy()  # (E,)

        blocked, blocked_p, off_lines = [], [], []
        for l in np.nonzero(proposal > 0.5)[0]:
            line_on = bool(edge[0, int(l), 1] > 0.5)  # 当前是否连通
            if line_on and p[int(l)] < self.threshold:
                allowed[l] = 0.0
                blocked.append(int(l))
                blocked_p.append(float(p[int(l)]))
            elif not line_on:
                off_lines.append(int(l))
        return allowed, {"blocked": blocked, "blocked_p": blocked_p, "p": p,
                         "threshold": self.threshold, "gated_off_lines": off_lines}


class HybridShield:
    """混合屏蔽器（FA-001 决策产物）：学习门初筛 + 精确仿真终审。

    - 第一层 N1Shield：学习门挡住大概率不安全 toggle（削减精确仿真开销）；
    - 第二层精确校验：对门放行的每个 toggle，用单层组合仿真（与标签同一实现，
      src/training/gen_n1_labels.py）复算安全准则；
    - **误放率恒为 0**：精确层是最终裁决，学习门误差只影响误杀率（性能代价）。

    精确校验需要 grid2op 环境（当前真实观测），由 ShieldedGraphEnv 在调用时传入。
    """

    def __init__(self, gate, alpha=0.1, device="cpu"):
        self.inner = N1Shield(gate, alpha=alpha, device=device)
        self._check_counter = 0

    def filter(self, obs, proposal, env=None):
        if env is None:
            raise ValueError("HybridShield 需要 grid2op 环境（env）做精确终审")
        from src.training.gen_n1_labels import (
            _simulate_or_none, _toggle_act, n1_score_of,
        )

        first, info = self.inner.filter(obs, proposal)
        glop = env.env_glop
        obs_glop = glop.get_obs()
        base = n1_score_of(obs_glop, glop)

        allowed = first.copy()
        exact_blocked, n_checks = [], 0
        for l in np.nonzero(first > 0.5)[0]:
            line_on = bool(obs_glop.line_status[int(l)] == 1)
            if not line_on:
                continue  # 重连请求：门不覆盖，交环境合法性
            n_checks += 1
            post = _simulate_or_none(obs_glop, _toggle_act(glop, l))
            safe = (post is not None and hasattr(post, "rho")
                    and post.rho.max() < 0.95
                    and n1_score_of(obs_glop, glop, toggle_l=l) >= base)
            if not safe:
                allowed[l] = 0.0
                exact_blocked.append(int(l))

        self._check_counter += n_checks
        info["exact_blocked"] = exact_blocked
        info["n_exact_checks"] = n_checks
        info["exact_base_n1"] = base
        return allowed, info

    @property
    def total_exact_checks(self):
        return self._check_counter
