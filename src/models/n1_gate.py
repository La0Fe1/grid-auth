"""边级 N-1 可行性门（安全柱的核心，创新性声明机制①）。

与遗留 train_n1_supervised.py 的区别：遗留分类器预测"最优单线断开"（推荐器），
本模块预测"toggle 线路 ℓ 是否满足可操作安全准则"（可行性判定器），是屏蔽器的判定组件。

标签定义（2026-09-13 修订，生成脚本见 src/training/gen_n1_labels.py）：
"toggle 后完美 N-1 安全"在本算例几乎不可满足（标签全零、门退化），已弃用。
现行定义：toggle ℓ 后的状态无过载（max rho < 0.95 留裕度）且
N-1 安全度不恶化（n1_score(post) ≥ n1_score(current)）→ y_ℓ = 1，否则 0。

模型：与 GNN 策略同构的 GAT 编码器 + 参数共享 per-edge readout，
输出 (B, E) logits → sigmoid 概率。参数量与边数无关（14/36/118-bus 共用同一门）。
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class N1Gate(nn.Module):
    def __init__(self, edge_index, node_dim=3, edge_dim=2, hidden=64):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden = hidden
        self.conv1 = GATConv(node_dim, hidden, heads=2, edge_dim=edge_dim)
        self.conv2 = GATConv(hidden * 2, hidden, heads=1, edge_dim=edge_dim)
        self.edge_readout = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )
        self.act = nn.ReLU()
        self.register_buffer("edge_index", torch.as_tensor(edge_index, dtype=torch.long))

    def forward(self, x_node, x_edge):
        """(B,N,node_dim),(B,E,edge_dim) → (B,E) 安全概率。"""
        logits = self._encode(x_node, x_edge)
        return torch.sigmoid(logits)

    def _encode(self, x_node, x_edge, return_attention=False):
        B, N, _ = x_node.shape
        E = x_edge.shape[1]
        dev = x_node.device
        x = x_node.reshape(B * N, -1)
        e = x_edge.reshape(B * E, -1)
        node_offset = torch.arange(B, device=dev) * N
        ei = self.edge_index.repeat(1, B) + node_offset.repeat_interleave(E)[None, :]
        attns = []
        x, a1 = self.conv1(x, ei, edge_attr=e, return_attention_weights=True)
        x = self.act(x)
        if return_attention:
            attns.append(a1[1][: B * E].reshape(B, E, -1))
        x, a2 = self.conv2(x, ei, edge_attr=e, return_attention_weights=True)
        x = self.act(x)
        if return_attention:
            attns.append(a2[1][: B * E].reshape(B, E, -1))
        edge_emb = (x[ei[0]] + x[ei[1]]) / 2
        logits = self.edge_readout(edge_emb).squeeze(-1).reshape(B, E)
        if return_attention:
            return logits, attns
        return logits

    def encode_with_attention(self, x_node, x_edge):
        """推理用：返回 (安全概率, 各层注意力权重)。自环边布局假设与策略同源
        （tests/test_gnn_policy.py 固化）。"""
        with torch.no_grad():
            logits, attns = self._encode(x_node, x_edge, return_attention=True)
        return torch.sigmoid(logits), attns

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())
