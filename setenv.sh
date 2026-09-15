#!/usr/bin/env bash
# AGENTS.md 0.1：运行任何脚本前先 `source setenv.sh`，
# 把所有缓存/临时文件重定向到项目内目录，禁止散落到用户主目录/系统临时目录。
ROOT="C:/Users/佬肥/PycharmProjects/project-grid-gnn"

mkdir -p "$ROOT/.tmp" "$ROOT/.cache" "$ROOT/.cache/matplotlib" "$ROOT/.cache/cuda" "$ROOT/.cache/pycache" "$ROOT/.cache/pip" "$ROOT/.logs"

export TMPDIR="$ROOT/.tmp"
export TEMP="$ROOT/.tmp"
export TMP="$ROOT/.tmp"
export XDG_CACHE_HOME="$ROOT/.cache"
export HF_HOME="$ROOT/.cache"
export TRANSFORMERS_CACHE="$ROOT/.cache"
export TORCH_HOME="$ROOT/.cache"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export CUDA_CACHE_PATH="$ROOT/.cache/cuda"
export PYTHONPYCACHEPREFIX="$ROOT/.cache/pycache"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export PYTHONIOENCODING=utf-8
