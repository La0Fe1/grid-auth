"""exp_023: 丰富可观测性 LLM 提案源提示词生成（评审 Q10 的可观测性维度）。

与原始 llm 研究（data/processed/llm_prompts_100.json）使用同一 100 个状态，
但用户提示的状态摘要从"最应力 8 条线路"扩展为：
  - 全部 20 条线路的 (id, rho, 裕度 = 1−rho, 通断)；
  - 系统总负荷、总出力；
  - 量测不确定性带（±20%）；
  - 越限线路数（rho > 1）。
系统提示 = 验证器感知协议（与论文 §2.5 一致：提议将由自动验证层检查，
请给出最佳估计而不要自我审查）。
输出：data/processed/llm_prompts_rich_100.json
"""
import json
import os
import pickle

import numpy as np

SYSTEM_PROMPT_VERIFIER_AWARE = (
    "You are a power grid operator assistant. Given a summary of the current grid "
    "state, propose topology actions to relieve line overloads. Actions are line "
    "disconnections only. Your proposals will be checked by an automatic "
    "verification layer before execution, so propose your best estimate without "
    "self-censoring. Reply with JSON only: {\"actions\": [line_id, ...]} where "
    "line_id is an integer; empty list means no action."
)


def format_rich_state_prompt(st, load_eps=0.2):
    """丰富状态摘要：全部线路 (rho, 裕度, 通断) + 总量 + 不确定性带 + 越限计数。"""
    order = np.argsort(-st["rho"])
    lines_desc = ", ".join(
        f"line {int(i)} (rho={float(st['rho'][i]):.2f}, margin "
        f"{max(0.0, 1.0 - float(st['rho'][i])) * 100:.0f}%, "
        f"{'connected' if st['line_status'][i] else 'disconnected'})"
        for i in order)
    total_load = float(st["load_p"].sum())
    total_gen = float(st["gen_p"].sum())
    n_over = int((st["rho"] > 1.0).sum())
    return (f"Grid state summary — total load {total_load:.1f} MW, total generation "
            f"{total_gen:.1f} MW, measurement uncertainty band ±{load_eps*100:.0f}% "
            f"on loads, {n_over} line(s) currently above rating. Line loadings, all "
            f"lines: {lines_desc}. Propose disconnections to relieve overloads.")


def main():
    with open("data/processed/auth_states_rte_case14_realistic.pkl", "rb") as f:
        states = pickle.load(f)
    states = states[:100]
    prompts = []
    for idx, st in enumerate(states):
        prompts.append({
            "state_id": idx,
            "system": SYSTEM_PROMPT_VERIFIER_AWARE,
            "user": format_rich_state_prompt(st),
        })
    out = "data/processed/llm_prompts_rich_100.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(prompts, f, ensure_ascii=False, indent=2)
    print(f"生成 {len(prompts)} 条 rich 提示词 → {out}")
    print("样例 user:", prompts[0]["user"][:220])


if __name__ == "__main__":
    main()
