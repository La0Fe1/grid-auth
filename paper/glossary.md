# 术语表（AGENTS.md 4.2.3：全文统一表述）

| 术语（中文） | 术语（英文，论文用） | 定义 |
|---|---|---|
| 授权层 | authorization layer | 在动作执行前对其进行物理安全验证的运行时组件 |
| 区间验证 | interval verification | 对不确定性集内所有模型同时做最坏情况验证（本文：LP 精确区间潮流） |
| 标称验证 | nominal verification | 仅对单一标称模型验证（TwinGridShield/SafeVolt 的做法） |
| 蒙特卡洛验证 | Monte Carlo verification | 在不确定性集内随机采样的经验最坏情况验证（统计近似，无构造保证） |
| 误放率 | false-pass rate | P(动作被放行 且 真值存在失配下危险) |
| 误拒率 | false-reject rate | P(动作被拒绝 且 真值全安全) |
| 失配 | model mismatch | 验证器所用模型参数/量测与真实系统的偏差 |
| 不确定性箱 | uncertainty box | 负荷量测的对称区间 ±ε（本文的基础不确定性模型） |
| 遥测识别 | telemetry-based identification | 从运行遥测（相角差/有功潮流）回归支路电抗，无需离线电网文件 |
| 构造性保证 | constructive guarantee | 由算法构造直接成立的安全性质（非统计近似）：箱内全模型安全 |
| 安全裕度 | safety margin | 授权判定的保守系数：\|f\| ≤ (1−m)×热限 |
| 真值 | ground truth (GT) | 失配箱内 M 点采样的危险率（实验的判定基准，ε_ref/m_ref 固定） |
| 提案源 | proposer | 提出候选拓扑动作的组件（LLM/贪心/随机） |
| 验证者感知提示词 | verifier-aware prompt | 告知 LLM 提案将由外部安全层验证的提示词（解冻行为） |
| 自证安全提示词 | self-verify prompt | 要求 LLM 自行保证安全的提示词（冻结行为） |
| DC 模型 | DC network model | 忽略无功与损耗的线性潮流模型（Bθ=P） |
| 区间直流潮流 | interval DC power flow | 注入箱 → 支路潮流精确区间的线性规划计算 |

## 数学符号约定（全文统一）
- 向量：小写粗体（$\mathbf{f}$）；矩阵：大写粗体（$\mathbf{B}$）；标量：斜体（$H, R$）
- 支路潮流区间：$[f_\ell^-, f_\ell^+]$；负荷箱：$[P_i^-, P_i^+]$；不确定性宽度：$\varepsilon$；安全裕度：$m$
- 线路集合 $\mathcal{L}$，节点集合 $\mathcal{N}$，松弛节点 $s_0$
