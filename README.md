# Uncertainty-Robust Runtime Authorization for Grid AI Agent Actions

电网 AI 智能体动作的**不确定性鲁棒运行时安全授权**：区间潮流（LP 精确最坏情况）验证，
在负荷量测/参数有界失配下实现构造性零危险放行。

**核心结果**（rte_case14_realistic，11,101 用例，±20% 失配箱）：
标称验证（TwinGridShield 式）误放率 17.96%；蒙特卡洛验证 2.55%；**区间验证 0.00%（误拒仅 1.02%，28.8ms/动作）**。

## 复现（从零）

```bash
bash reproduce.sh
```

手动步骤：

1. **环境**：conda `gridgnn`（Python 3.12 + grid2op 1.12.5 + lightsim2grid + torch + PyG + scipy）。
   运行前 `source setenv.sh`（缓存/临时文件重定向到项目内，见 AGENTS.md 0.1）。
2. **测试**：`python -m pytest tests/ -q`（70 项）。
3. **数据**：`python data/get_data.py`（rte_case14_realistic；icaps/idf 用内置 dev 模式）。
4. **主实验**：`python -u -m src.evaluation.auth_eval --resume --out experiments/exp_012/results/auth_eval.json`
   （大仿真 ≥2h 档；--resume 断点续跑，每用例增量落盘 *_cases.jsonl）。
5. **消融/鲁棒性/跨系统/LLM**：见 reproduce.sh 第 5-8 步。
6. **图表**：每个 `experiments/exp_NNN/fig_*.py` 生成一张图（输出到 paper/figures/）。
7. **论文**：`cd paper && pdflatex main.tex && bibtex main && pdflatex ×2`。

## 目录结构（AGENTS.md 3.1）

- `src/models`（gnn_policy/n1_gate/shield——B 方向遗留资产）、`src/utils`（dc_interval 区间潮流核心、env、graph、freq_dyn）、`src/evaluation`（auth_eval 授权评估、run_ablation/run_robustness、lang 门控脚本在 paper/）、`src/baselines`、`src/training`
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

## 引用与许可

- Grid2Op：MPL-2.0；数据集来自官方 releases（datasets-v0.1.0）。
- LLM 提案数据集 data/processed/llm_proposals.json 由 DeepSeek（deepseek-chat）生成，
  提示词协议全文见 src/evaluation/gen_llm_prompts.py。
- 论文 9 条参考文献全部经 Semantic Scholar/Crossref/DOAJ API 核实（references.bib 内注明）。
