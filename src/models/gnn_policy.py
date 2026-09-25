"""GNN 策略（SB3 ActorCriticPolicy 接口，由遗留 train_gnn_ppo.py 迁移）。

核心设计（跨算例泛化的机制基础，主假设 H3）：
- 2 层 GATConv（带边特征）：节点=变电站、边=线路；
- per-edge readout：参数共享的同一个 MLP 对每条边输出 toggle logit，
  边数 n_line 不进入参数量 → 14-bus 训练的权重可零样本部署到 36/118-bus；
- value head：全局平均池化后单层线性；
- encode_with_attention()：额外返回各层 GAT 注意力权重（每边 × 每头），
  供解释柱（H2：注意力归因与 N-1 越限支路一致性）评估使用。
"""
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from torch_geometric.nn import GATConv, global_mean_pool


class GNNGraphPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 edge_index, node_dim=3, edge_dim=2, hidden=64, **kwargs):
        self.gnn_edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden = hidden
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule):
        # GNN 层（参数只依赖 node_dim/edge_dim/hidden，与 n_line 无关）
        self.conv1 = GATConv(self.node_dim, self.hidden, heads=2, edge_dim=self.edge_dim)
        self.conv2 = GATConv(self.hidden * 2, self.hidden, heads=1, edge_dim=self.edge_dim)
        # 参数共享的 edge readout：同一个 MLP 处理每条边
        self.edge_readout = nn.Sequential(
            nn.Linear(self.hidden, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 1)
        )
        self.value_head = nn.Linear(self.hidden, 1)
        self.act = nn.ReLU()

        optimizer_kwargs = getattr(self, "optimizer_kwargs", {})
        self.optimizer = self.optimizer_class(
            self.parameters(), lr=lr_schedule(1), **optimizer_kwargs
        )

    def _encode(self, obs, return_attention=False):
        """图 → (per-line logits, value[, 各层注意力权重])。"""
        x = obs["node_feat"]  # (B, N, Fn)
        e = obs["edge_feat"]  # (B, E, Fe)
        B, N, _ = x.shape
        E = e.shape[1]
        dev = x.device

        x = x.reshape(B * N, -1)
        e = e.reshape(B * E, -1)
        node_offset = torch.arange(B, device=dev) * N
        ei = self.gnn_edge_index.to(dev).repeat(1, B)  # (2, B*E)
        ei = ei + node_offset.repeat_interleave(E)[None, :]
        batch = torch.arange(B, device=dev).repeat_interleave(N)

        attns = []
        x, a1 = self.conv1(x, ei, edge_attr=e, return_attention_weights=True)
        x = self.act(x)
        if return_attention:
            # PyG GATConv 返回 (x, (attn_ei, alpha))；add_self_loops=True（默认）时
            # 自环边追加在原始边之后（tests/test_gnn_policy.py 固化该布局假设），
            # 取前 B*E 条 = 原始线路边注意力：(B*E, heads) → (B, E, heads)
            attns.append(a1[1][: B * E].reshape(B, E, -1))  # (B, E, 2)
        x, a2 = self.conv2(x, ei, edge_attr=e, return_attention_weights=True)
        x = self.act(x)
        if return_attention:
            attns.append(a2[1][: B * E].reshape(B, E, -1))  # (B, E, 1)

        # 边 embedding = 两端节点 embedding 平均
        edge_emb = (x[ei[0]] + x[ei[1]]) / 2  # (B*E, hidden)
        line_logits = self.edge_readout(edge_emb).squeeze(-1).reshape(B, E)

        glob = global_mean_pool(x, batch)  # (B, hidden)
        value = self.value_head(glob)  # (B, 1)

        if return_attention:
            return line_logits, value, attns
        return line_logits, value

    def forward(self, obs, deterministic=False):
        line_logits, value = self._encode(obs)
        dist = self.action_dist.proba_distribution(action_logits=line_logits)
        actions = dist.get_actions(deterministic=deterministic)
        log_prob = dist.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, value, log_prob

    def evaluate_actions(self, obs, actions):
        line_logits, value = self._encode(obs)
        dist = self.action_dist.proba_distribution(action_logits=line_logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return value, log_prob, entropy

    def _predict(self, observation, deterministic=False):
        line_logits, _ = self._encode(observation)
        dist = self.action_dist.proba_distribution(action_logits=line_logits)
        return dist.get_actions(deterministic=deterministic)

    def predict_values(self, obs):
        _, value = self._encode(obs)
        return value

    def encode_with_attention(self, obs):
        """推理用：返回 (per-line logits, value, 各层注意力权重列表)。

        注意力权重形状：第 1 层 (B, E, 2)，第 2 层 (B, E, 1)；均来自 softmax，取值 [0,1]。
        """
        with torch.no_grad():
            line_logits, value, attns = self._encode(obs, return_attention=True)
        return line_logits, value, attns

    def count_parameters(self):
        """参数量（exp_008 计算成本报告用）。"""
        return sum(p.numel() for p in self.parameters())
