#!/usr/bin/env bash
# 缓存/临时文件重定向到仓库内（AGENTS.md 0.1）。
# 仓库根目录自动检测（不依赖绝对路径）。
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$ROOT/.tmp" "$ROOT/.cache" "$ROOT/.cache/matplotlib" "$ROOT/.cache/cuda" "$ROOT/.cache/pycache" "$ROOT/.logs"

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
export PYTHONIOENCODING=utf-8
export PIP_CACHE_DIR="$ROOT/.cache/pip"
