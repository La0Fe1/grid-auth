"""exp_003 门校准：在验证集上选择屏蔽阈值 alpha（误放预算）。

目标（失败判据 F4）：学习式门与精确仿真判定的总体不一致率 ≤5%。
统计口径（写入 JSON）：
- 条件误放率 = P(门判安全 | 真不安全)，屏蔽器放行危险动作的风险；
- 条件误杀率 = P(门判不安全 | 真安全)，屏蔽器误伤正常动作的比例；
- 不一致率 = P(判定 ≠ 标签)。

选定策略：在"条件误放率 ≤ max_false_pass"约束下选不一致率最小的 alpha。

用法：python -m src.evaluation.calibrate_gate \
    --gate experiments/exp_003/checkpoints/n1_gate.pt \
    --labels data/processed/n1_gate_labels.pkl \
    --out experiments/exp_003/results/calibration.json
"""
import argparse
import json
import os
import pickle

import numpy as np
import torch

from src.models.n1_gate import N1Gate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--labels", default="data/processed/n1_gate_labels.pkl")
    ap.add_argument("--max-false-pass", type=float, default=0.05,
                    help="允许的最大条件误放率")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/exp_003/results/calibration.json")
    args = ap.parse_args()

    with open(args.labels, "rb") as f:
        data = pickle.load(f)
    meta, samples = data["meta"], data["samples"]
    X_node = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
    X_edge = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)
    Y = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.float32)

    # 与训练一致的验证集划分（train_n1_gate.py 同款）
    np.random.seed(args.seed)
    idx = np.random.permutation(len(samples))
    te_idx = idx[int(len(samples) * 0.8):]

    gate = N1Gate(meta["edge_index"])
    gate.load_state_dict(torch.load(args.gate, map_location="cpu"))
    gate.eval()
    with torch.no_grad():
        probs = gate(X_node[te_idx], X_edge[te_idx]).ravel().numpy()
    y = Y[te_idx].ravel().numpy().astype(bool)

    # 温度缩放（logit 域）：缓解门过度保守/自信导致的决策边界失配。
    # T 在验证集上按 Brier 分数网格搜索（0.5–5.0）。
    def temp_scale(p, T):
        p = np.clip(p, 1e-7, 1 - 1e-7)
        logit = np.log(p / (1 - p))
        return 1.0 / (1.0 + np.exp(-logit / T))

    best_T, best_brier = 1.0, float("inf")
    for T in [0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]:
        brier = float(np.mean((temp_scale(probs, T) - y) ** 2))
        if brier < best_brier:
            best_T, best_brier = T, brier
    probs = temp_scale(probs, best_T)

    rows = []
    for alpha in [0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50]:
        pred = probs >= (1.0 - alpha)
        fp_cond = float(((pred == 1) & (y == 0)).sum() / max((y == 0).sum(), 1))
        fb_cond = float(((pred == 0) & (y == 1)).sum() / max((y == 1).sum(), 1))
        incon = float((pred != y).mean())
        rows.append({"alpha": alpha, "false_pass_cond": fp_cond,
                     "false_block_cond": fb_cond, "inconsistency": incon})

    feasible = [r for r in rows if r["false_pass_cond"] <= args.max_false_pass]
    chosen = min(feasible, key=lambda r: r["inconsistency"]) if feasible else rows[0]
    f4_pass = chosen["inconsistency"] <= 0.05

    result = {"args": vars(args), "alpha_grid": rows,
              "chosen_alpha": chosen["alpha"], "f4_pass": f4_pass,
              "chosen_stats": chosen,
              "temperature": best_T, "brier": best_brier}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"选择 alpha={chosen['alpha']}（条件误放率 {chosen['false_pass_cond']:.3f}，"
          f"不一致率 {chosen['inconsistency']:.3f}），F4 判据通过={f4_pass}")
    for r in rows:
        print(f"  alpha={r['alpha']:.2f}: 误放={r['false_pass_cond']:.3f} "
              f"误杀={r['false_block_cond']:.3f} 不一致={r['inconsistency']:.3f}")


if __name__ == "__main__":
    main()
