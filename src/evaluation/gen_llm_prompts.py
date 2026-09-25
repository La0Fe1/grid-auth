"""exp_017 LLM 提案源：状态 → 标准化提示词（DeepSeek 调用协议）。

协议（论文方法节将全文引用）：
- 系统提示：电网运行助手角色 + 任务（缓解过载的拓扑动作）+ 输出格式（JSON）；
- 用户提示：状态摘要 = 最应力 8 条线路的 (线路id, 负载率, 通断状态) +
  可用动作说明（断开线路）+ 安全约束说明（断开后不得造成其他线路过载）。
- 输出：{"actions": [line_id, ...]}，空列表 = 不动。

提示词保存为 data/processed/llm_prompts.json，供代理调用 DeepSeek（llm-chat MCP）；
回复经 assemble_llm_proposals 组装为提案数据集。

用法：python -m src.evaluation.gen_llm_prompts \
    --states data/processed/auth_states.pkl --n 20 --out data/processed/llm_prompts.json
"""
import argparse
import json
import os
import pickle

import numpy as np

from src.utils.env import Grid2OpGraphEnv

SYSTEM_PROMPT = (
    "You are a power grid operator assistant. Given a summary of the current grid "
    "state, propose topology actions to relieve line overloads. Actions are line "
    "disconnections only. You MUST NOT disconnect a line if doing so would overload "
    "any other line. Reply with JSON only: {\"actions\": [line_id, ...]} where "
    "line_id is an integer; empty list means no action."
)


def format_state_prompt(st, thermal, top_k=8):
    """状态摘要：最应力 top_k 条线路的 (id, 负载率 rho, 通断)。"""
    order = np.argsort(-st["rho"])[:top_k]
    lines_desc = ", ".join(
        f"line {int(i)} (rho={float(st['rho'][i]):.2f}, "
        f"{'connected' if st['line_status'][i] else 'disconnected'})"
        for i in order)
    return (f"Grid state summary — most loaded lines: {lines_desc}. "
            f"Propose disconnections to relieve overloads.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", default="data/processed/auth_states.pkl")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="data/processed/llm_prompts.json")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.states):
        raise FileNotFoundError(f"状态缓存不存在: {args.states}（先跑 auth_eval 采样）")
    with open(args.states, "rb") as f:
        states = pickle.load(f)
    states = states[: args.n]

    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    thermal = env.env_glop.get_thermal_limit().astype(float)

    prompts = []
    for idx, st in enumerate(states):
        prompts.append({"state_id": idx, "system": SYSTEM_PROMPT,
                        "user": format_state_prompt(st, thermal)})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"生成 {len(prompts)} 条提示词 → {args.out}")


if __name__ == "__main__":
    main()
