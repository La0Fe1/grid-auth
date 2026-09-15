#!/usr/bin/env bash
# 一键复现脚本（AGENTS.md 7）：从零复现论文全部实验与图表。
# 用法（Git Bash / WSL）：bash reproduce.sh
# 前提：conda 环境 gridgnn（Python 3.12 + grid2op 1.12.5 + lightsim2grid + torch + PyG + scipy）
set -e
cd "$(dirname "$0")"

echo "== 1. 环境变量重定向（0.1 条款）=="
source setenv.sh

PY="E:/forsci/minicinda/envs/gridgnn/python.exe"
if [ ! -f "$PY" ]; then
  PY=$(command -v python)
fi

echo "== 2. 单元测试（70 项）=="
"$PY" -m pytest tests/ -q

echo "== 3. 数据集获取（rte_case14_realistic；icaps/idf 用内置 dev 模式）=="
"$PY" data/get_data.py

echo "== 4. 主实验 exp_012（大仿真 ≥2h 档；--resume 断点续跑）=="
"$PY" -u -m src.evaluation.auth_eval --resume \
  --out experiments/exp_012/results/auth_eval.json

echo "== 5. 消融 exp_013 =="
"$PY" -u -m src.evaluation.run_ablation

echo "== 6. 鲁棒性 exp_014 =="
"$PY" -u -m src.evaluation.run_robustness

echo "== 7. 跨系统 exp_015 =="
for case in l2rpn_icaps_2021 l2rpn_idf_2023; do
  "$PY" -u -m src.evaluation.auth_eval --case $case --test-mode \
    --n-states 20 --lamdas 1.2 1.5 2.0 --gt-m 1000 \
    --conv-checkpoints 100 300 1000 --resume \
    --out experiments/exp_015/results/auth_eval_${case}_fine.json
done

echo "== 8. LLM 提案（需 DeepSeek API via llm-chat MCP；无密钥时使用随库数据集）=="
echo "   随库提案集: data/processed/llm_proposals.json"
"$PY" -u -m src.evaluation.auth_eval --proposals data/processed/llm_proposals.json \
  --out experiments/exp_017/results/llm_auth_eval.json

echo "== 9. 图表生成 =="
for m in experiments.exp_011_interval_auth.fig_framework \
         experiments.exp_011_interval_auth.fig_telemetry \
         experiments.exp_012.fig_main \
         experiments.exp_017.fig_llm \
         experiments.exp_014.fig_eps \
         experiments.exp_013.fig_ablation; do
  "$PY" -u -m $m
done

echo "== 10. 论文编译 =="
cd paper
pdflatex -interaction=nonstopmode main.tex > /dev/null
bibtex main > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
pdflatex -interaction=nonstopmode main.tex > /dev/null
cd ..

echo "== 完成：paper/main.pdf + experiments/*/results/* =="
