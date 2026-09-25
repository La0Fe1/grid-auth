# Uncertainty-Robust Runtime Authorization for Grid AI Agent Actions

电网 AI 智能体动作的**不确定性鲁棒运行时安全授权**：区间潮流（LP 精确最坏情况，箱式不确定集
下退化为闭式角点公式）验证，在负荷量测/参数有界失配下实现构造性零危险放行。

**核心结果**（rte_case14_realistic，12,500 用例，±20% 失配箱，精确角点真值）：
标称验证（TwinGridShield 式）误放率 18.98%；蒙特卡洛验证（K = 100）3.89%；
**区间验证误放 0.00%、误拒 0.00%，0.18 ms/动作**（闭式角点；等价 LP 形式 27.9 ms）。
放行率 49.3% / 41.6% / 40.0%（标称 / 蒙特卡洛 / 区间）。

**其余主要结果**：

- **跨系统**（各 1,500 用例）：36-bus 标称 85.19%、蒙特卡洛 77.59%、区间 0.00%（放行 5.3%）；
  118-bus 85.00% / 82.72% / 0.00%（放行 5.0%）。结构化不确定集把放行率提高到 34.0% / 33.3%
  （与箱式语义不同，须先在真实量测对上校准）。
- **DC-to-AC 覆盖**（4.94M 支路点）：AC 潮流落入 DC 区间的比例 72.8%
  （λ = 2/3/4 分别为 76.4% / 73.1% / 68.5%，DC 最坏角点处 64.1%）；
  DC 安全用例在 DC 最坏角点上 AC 越限 19.7%（λ = 2）；被拒侧 witness 在 AC 下确认率 96.25%。
- **两级部署门**：tier 2 在应力触发时于 DC 最坏角点做 AC 复核（early exit）。τ = 0.9 触发
  60/341、捕获 44/67 个角点越限（65.7%），放行 297/341（87.1%），1.2 ms/触发动作（22.3 次 AC 求解）；
  τ = 0（穷举）清除全部 67 个角点越限，放行 274/341，1.3 ms。
- **遥测辨识**：仅用角度差与有功潮流，复现 AC 潮流 r = 0.9997、RMSE 0.66 MW（slope 0.98）；
  5 个快照即足够，50–80% 采样丢失与 ±0.10 rad 母线相关角度偏差下决策不变。
- **LLM 提案研究**：自证提示词下 100 个状态全部冻结（0/100）；验证者感知提示词下 46/100 行动，
  其提案失配危险率 60.00%，经标称 / 蒙特卡洛 / 区间分别降至 16.36% / 0.65% / 0.00%。

## 复现（从零）

```bash
bash reproduce.sh
```

手动步骤：

1. **环境**：conda `gridgnn`（Python 3.12 + grid2op 1.12.5 + lightsim2grid 1.0.0 + torch + PyG + scipy）。
   运行前 `source setenv.sh`（缓存/临时文件重定向到项目内，见 AGENTS.md 0.1）。
2. **测试**：`python -m pytest tests/ -q`（75 项）。
3. **数据**：`python data/get_data.py`（rte_case14_realistic；icaps / idf 用内置 dev 模式）。
4. **主实验（表 1 / 图 3）**：
   `python -m experiments.exp_016.full_rerun rte_case14_realistic` 重放全部 12,500 用例
   （5 种子 × 100 状态 × 5 候选 × 5 负荷档；独立采样 RandomState 20260916），
   再由 `python -m experiments.exp_016.exact_gt` 打**精确角点真值**标签，产出
   `experiments/exp_016/results/exact_gt.json`。
   注：同批用例在采样真值口径下的汇总为 `experiments/exp_012/results/auth_eval_*_full.json`，
   两种口径的差异见论文 2.7 节。
5. **跨系统（表 2 / 表 3）**：同 4，case 换 `l2rpn_icaps_2021` / `l2rpn_idf_2023 --test-mode`
   （各 1,500 用例）；结构化集合用
   `python -m experiments.exp_019_review_ext.run_structured --case l2rpn_icaps_2021`。
6. **消融与鲁棒性（图 6 / 图 7）**：`python -m experiments.exp_016.exact_ablation_rerun`
   （精确真值重跑，7,500 用例/配置）。
7. **DC-to-AC 覆盖与两级门（3.6 节 / 表 4）**：
   `python -m experiments.exp_016.dc_ac_containment`、
   `python -m experiments.exp_019_review_ext.run_twotier`。
8. **参与因子（表 5）**：`python -m experiments.exp_019_review_ext.run_genconst`。
9. **LLM 提案（图 5）**：评估用随库冻结提案集 `data/processed/llm_proposals_100.json`
   （无需 API 即可复现评估结果）；重新生成提案需 DeepSeek API，提示词协议见
   `src/evaluation/gen_llm_prompts.py` 与 `experiments/exp_019_review_ext/gen_rich_prompts.py`。
10. **图表**：`python -m experiments.exp_016.figs_exact`（图 3/6/7）等；配色样式模块
    `_pubstyle.py` 随仓库分发，输出到 `paper/figures/`。
11. **论文**：`cd paper && pdflatex main.tex && bibtex main && pdflatex ×2`。

## 目录结构（AGENTS.md 3.1）

- `src/models`（gnn_policy/n1_gate/shield——B 方向遗留资产）、`src/utils`（dc_interval 区间潮流核心、env、graph、freq_dyn）、`src/evaluation`（auth_eval 授权评估、run_ablation/run_robustness）、`src/baselines`、`src/training`
- `experiments/exp_NNN/`：每个实验的代码、配置、日志、结果（数值只引自这里）
- `paper/`：main.tex、main.pdf、figures/、references.bib（全部经检索工具核实）、glossary.md、outline.md
- 状态文件：`experiment_plan.md` / `experiment_log.md` / `decisions.md` / `failure_analysis.md`

## 关键决策与失败记录（诚实披露）

- **方向 B 原方案（RL 三柱）已否决**：15 配置 × 5 种子全部退化为 do-nothing；注意力解释无信号；
  根因 = rte_case14 生存任务中 do-nothing 近最优（死因是自然负荷峰值过载而非对手攻击）。
  完整证据链见 failure_analysis.md FA-001~003（作为负结果保留）。
- **当前方案 = 方向 B 重定义**：选题声明 paper/novelty_statement_v2.md；缺口 = TwinGridShield/SafeVolt
  只做标称验证（两者均已 API 核实）。
- **安全缺陷记录**：NaN 静默放行（LP 失败 → authorize 误放行，icaps 上触发）→ 已修复为 fail-closed；
  负负荷箱倒置（icaps 新能源反送）→ 箱宽改绝对值。两例均有回归测试。
- **评估口径**：种子级 mean±std + 场景级配对检验（5 对样本 Wilcoxon 最小 p=0.0625 的解决，decisions.md）。
- **真值口径**：主实验真值为**精确角点判据**而非采样（采样真值会漏掉箱体角点，低估泄漏、虚增误拒；
  两种口径的对照见 `experiments/exp_016/results/exact_gt.json` 与 exp_012 的采样汇总）。

## 引用与许可

- Grid2Op：MPL-2.0；数据集来自官方 releases（datasets-v0.1.0）。
- LLM 提案数据集 `data/processed/llm_proposals_100.json` 由 DeepSeek（deepseek-chat）生成，
  提示词协议全文见 `src/evaluation/gen_llm_prompts.py`。
- 论文 27 条参考文献全部经 arXiv API / Crossref / OpenAlex 核实（references.bib 内注明）。
