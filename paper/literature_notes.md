# 选题查重记录（方向 B：Agentic AI for safe & explainable smart grid control）

> 按 AGENTS.md 1.2：检索 2025-01 至今（18 个月窗口，即 2025-03-13 之后）直接相同题目的论文。
> 反幻觉铁律：仅收录经检索工具验证的条目；未验证条目单独列出，**禁止引用**。

## 已核实的相关论文（验证方式见括号）

| # | 论文 | 作者 | 年份/来源 | 三柱覆盖 | 与我们关系 |
|---|---|---|---|---|---|
| 1 | LLM-Guided Safe Reinforcement Learning for Energy System Topology Reconfiguration | Zongyan Zhang, Chao Shen, Xu Wan, Jie Song, Mingyang Sun | 2026, Applied Energy, arXiv:2603.14018（Semantic Scholar API 核实，paperId 7049730e） | 安全=拉格朗日软惩罚（无硬保证）；GNN=无；解释=无（LLM 只做轨迹修正，无动作归因） | **最接近**：安全 RL+拓扑重构，但三柱只占其一 |
| 2 | Power Grid Control with Graph-Based Distributed Reinforcement Learning | Carlo Fabrizio, Gianvito Losapio, Marco Mussi, Alberto Maria Metelli, Marcello Restelli | 2025, arXiv:2509.02861（Semantic Scholar API 核实） | GNN=有（编码观测）；安全=无（标准 PPO 奖励）；解释=无；"泛化"=计算效率而非零样本跨电网 | **最接近**：GNN+RL+Grid2Op，但无安全无解释 |
| 3 | Centrally Coordinated Multi-Agent RL for Power Grid Topology Control | Barbera de Mol, Davide Barbieri, Jan Viebahn, Davide Grossi | 2025, ACM e-Energy 2025, arXiv:2502.08681（Semantic Scholar API 核实） | GNN=无；安全=无；解释=无（贡献在动作空间分解） | 中：同任务不同方法 |
| 4 | Robust Defense Against Extreme Grid Events Using Dual-Policy RL Agents | Benjamin M. Peter, Mert Korkali | 2025, IEEE TPEC, arXiv:2411.11180（S2 页面与 ar5iv 交叉核实） | GNN=有（GCN）；安全=无控制屏蔽（对抗用于 N-k 筛选）；解释=无 | 中高：PPO+GNN 应对极端事件，无屏蔽无解释 |
| 5 | Imitation Learning for Intra-Day Power Grid Operation through Topology Actions | M. D. de Jong, Jan Viebahn, Y. Shapovalova | 2024, PKDD/ECML Workshops, arXiv:2407.19865（Semantic Scholar API 核实） | N-1 感知专家（ρN−1 奖励）+IL；GNN=无；解释=无 | 窗口外（2024-07），可引用其 N-1 定义与"IL 略逊于专家"结论 |
| 6 | AI challenge for safe and low carbon power grid operation | Adrien Pavão, Antoine Marot, Jules Sintes, Viktor Eriksson Möllerstedt, Laure Crochepierre, Karim Chaouache, Benjamin Donnot, Van Tuan Dang, Isabelle Guyon | 2025, Energy and AI, **DOI: 10.1016/j.egyai.2025.100564**（DOAJ API 核实） | L2RPN 官方赛后分析 | **缺口证据**：官方结论"无一参赛方案满足严格安全/可靠性标准" |
| 7 | An Explainable Graph RL Copilot (ResiliGraph-STGAT) | Zishan Ali Khan 等 | 2025, Zenodo 预印本, DOI: 10.5281/zenodo.20320542（OpenAIRE API 确认存在，未同行评审） | 解释=有（图注意力归因）；GNN=有；任务=配电故障/拥塞预测+RL 干预建议（IEEE 33-bus OpenDSS），非 Grid2Op 拓扑控制，无硬安全约束 | 邻域证据：注意力解释+RL 在电网已有先例，但任务不同 |

## 否决评估（AGENTS.md 1.2）

- **三柱合一（GNN 策略 + 硬 N-1 安全屏蔽 + 策略内建注意力解释 + 零样本跨算例）**：18 个月窗口内直接相同者 = **0 篇**。
- **高度相似（覆盖其中一柱或两柱）**：2 篇（#1 安全柱、#2 GNN 柱）→ 落在 1–2 篇档 → 按 1.2 必须指明具体技术缺口（见 paper/novelty_statement.md）。
- **原 PLAN.md 主创新"GNN 策略"单独看**：≥3 篇相似（#2、#4 及搜索摘要中若干未验证条目）→ **该表述已被否决，不得作为主创新**。主创新重新定义为本项目三柱组合。
- 技术可行性（1.2 最后一款）：Grid2Op 公开（MPL-2.0，待核实）、rte_case14_realistic 与 IEEE 14/36/118-bus 环境已在本机跑通、官方 PPO 基线可复现 → 可行。

## 未验证条目（禁止写入论文）

- TechRxiv "Intelligent Energy Management: RL and AI for Optimized Smart Grids"（PPO+GNN L2RPN，2025-02）：其 DOI 10.36227/techrxiv.174059992.22498810 在 Crossref **查无记录** → 疑似虚构，不引用。
- 搜索摘要中提及但未逐一核实的其他条目（H-MARL4PowerGridTopo GitHub 仓库、Anguiano Batanero et al. 2025、Graph-based soft-label IL 2025 等）：状态=未验证，如需引用必须先核实 DOI/arXiv ID。
