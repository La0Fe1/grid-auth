#!/usr/bin/env bash
# 一键复现脚本（AGENTS.md 7）：从零复现论文全部实验与图表。
# 用法（Git Bash / WSL）：bash reproduce.sh
# 前提：conda 环境 gridgnn（Python 3.12 + grid2op 1.12.5 + lightsim2grid 1.0.0 + torch + PyG + scipy）
set -e
cd "$(dirname "$0")"

echo "== 1. 环境变量重定向（0.1 条款）=="
source setenv.sh

PY="E:/forsci/minicinda/envs/gridgnn/python.exe"
if [ ! -f "$PY" ]; then
  PY=$(command -v python)
fi
echo "   python: $PY"

echo "== 2. 单元测试（75 项）=="
"$PY" -m pytest tests/ -q

echo "== 3. 数据集获取（rte_case14_realistic；icaps/idf 用内置 dev 模式）=="
"$PY" data/get_data.py

echo "== 4. 主实验（表 1 / 图 3）：12,500 用例重放 + 精确角点真值 =="
"$PY" -u -m experiments.exp_016.full_rerun rte_case14_realistic
"$PY" -u -m experiments.exp_016.exact_gt

echo "== 5. 跨系统（表 2）：36-bus / 118-bus，各 1,500 用例 =="
"$PY" -u -m experiments.exp_016.full_rerun l2rpn_icaps_2021 --test-mode
"$PY" -u -m experiments.exp_016.full_rerun l2rpn_idf_2023 --test-mode
"$PY" -u -m experiments.exp_016.exact_gt

echo "== 6. 结构化不确定集（表 3）=="
"$PY" -u -m experiments.exp_019_review_ext.run_structured --case l2rpn_icaps_2021
"$PY" -u -m experiments.exp_019_review_ext.run_structured --case l2rpn_idf_2023

echo "== 7. 消融与鲁棒性（图 6 / 图 7）：精确真值重跑 =="
"$PY" -u -m experiments.exp_016.exact_ablation_rerun

echo "== 8. DC-to-AC 覆盖（3.6 节）与两级部署门（表 4）=="
"$PY" -u -m experiments.exp_016.dc_ac_containment
"$PY" -u -m experiments.exp_019_review_ext.run_twotier

echo "== 9. 参与因子（表 5）与遥测缺失鲁棒性 =="
"$PY" -u -m experiments.exp_019_review_ext.run_genconst
"$PY" -u -m experiments.exp_019_review_ext.run_telemetry_missing

echo "== 10. LLM 提案评估（图 5）：使用随库冻结提案集，无需 API =="
echo "    提案集: data/processed/llm_proposals_100.json（自证提示词集: llm_proposals.json）"
"$PY" -u -m experiments.exp_017.fig_llm

echo "== 11. 图表生成 =="
for m in experiments.exp_016.figs_exact \
         experiments.exp_011_interval_auth.fig_framework \
         experiments.exp_011_interval_auth.fig_telemetry \
         experiments.exp_017.fig_llm \
         experiments.exp_014.fig_eps \
         experiments.exp_013.fig_ablation; do
  "$PY" -u -m $m
done

echo "== 12. 论文编译 =="
cd paper
pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
cd ..

echo "== 完成：paper/main.pdf + experiments/*/results/* =="
