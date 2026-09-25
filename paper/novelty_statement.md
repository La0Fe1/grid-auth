# 创新性声明（AGENTS.md 1.3，≤500 字）

**题目（方向 B）**：带硬 N-1 安全屏蔽与注意力解释的图神经网络策略：电网拓扑控制的零样本跨算例部署

**现有工作的具体局限（均为已核实文献）**

1. Zhang et al., "LLM-Guided Safe Reinforcement Learning for Energy System Topology Reconfiguration", *Applied Energy*, 2026（arXiv:2603.14018）：Safety-SAC 用拉格朗日软惩罚近似安全，约束满足无硬保证；LLM 模块只向回放池注入修正轨迹，不解释策略为何选择某动作；策略为 MLP，无图结构，不具跨算例迁移能力。
2. Fabrizio et al., "Power Grid Control with Graph-Based Distributed Reinforcement Learning", arXiv:2509.02861, 2025：GNN 仅编码观测，安全仍依赖标准 PPO 奖励软惩罚；无任何决策解释；其"泛化"指分布式框架的计算效率，不是零样本跨电网部署。
3. Peter & Korkali, "Robust Defense Against Extreme Grid Events Using Dual-Policy Reinforcement Learning Agents", *IEEE TPEC* 2025（arXiv:2411.11180）：PPO+GNN 双策略应对极端事件，但策略无安全屏蔽层，越限只能事后体现在奖励里；无动作级解释。
4. Pavão et al., "AI challenge for safe and low carbon power grid operation", *Energy and AI*, 2025（DOI: 10.1016/j.egyai.2025.100564）：L2RPN 官方赛后分析明确写道，所有参赛方案无一满足严格安全/可靠性标准——这是缺口存在的社区官方证据。

**技术缺口（一句话，28 字）**：硬 N-1 屏蔽 + 注意力可解释的 GNN 策略与零样本跨算例泛化。

**为什么能填补（机制层面）**

① 屏蔽器分两层：学习式边级门（预测"toggle 后无过载且 N-1 安全度不恶化"的概率）做快速初筛，削减精确仿真开销（削减率列为待验证假设）；初筛放行的动作再经单层组合仿真的精确终审——**误放率恒为 0**（硬安全由构造成立），学习门的误差只影响误杀率（性能代价，可测可控）。两层对母线数均不变，随策略一起零样本迁移。② GAT 注意力是策略自身的决策依据，屏蔽器阻挡某动作时可回报"哪条支路不安全"——安全与解释共用同一图表示，而非事后 SHAP。③ 图消息传递与边级门的表示对母线数不变，14-bus 训练可零样本部署 36/118-bus——MLP 策略与 LLM 引导框架均被输入维度锁死，无法迁移。

**对 Q1 审稿人"足够新"**：社区官方总结（Pavão 2025）刚宣布"无一方案满足严格安全标准"；窗口内两篇最接近工作（Zhang 2026；Fabrizio 2025）各只占三柱之一。三柱合一的架构在 18 个月窗口内无直接相同者（逐篇核对表见 paper/literature_notes.md）。
