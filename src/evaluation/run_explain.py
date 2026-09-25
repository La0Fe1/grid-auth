"""exp_004 解释柱评估（H2）：GAT 注意力归因与 N-1 越限支路的一致性。

输出 explain.json：scan_states 的逐状态一致性 + 与随机排序基线的配对 Wilcoxon p 值。

用法：python -m src.evaluation.run_explain \
    --model experiments/exp_002/checkpoints/gnn_seed42/best_model.zip \
    --out experiments/exp_004/results/explain.json
"""
import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO

from src.evaluation.explain import scan_states
from src.utils.env import Grid2OpGraphEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--case", default="rte_case14_realistic")
    ap.add_argument("--n-states", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="experiments/exp_004/results/explain.json")
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    env = Grid2OpGraphEnv(case=args.case, test=False)
    t0 = time.time()
    res = scan_states(model.policy, env, n_states=args.n_states,
                      top_k=args.top_k, rng_seed=42)
    elapsed = time.time() - t0

    jac, rk = np.array(res["jaccard"]), np.array(res["rank"])
    jac_r, rk_r = np.array(res["jaccard_random"]), np.array(res["rank_random"])
    summary = {
        "n_states": args.n_states,
        "jaccard_mean": float(np.nanmean(jac)),
        "jaccard_random_mean": float(np.nanmean(jac_r)),
        "jaccard_p": res["jaccard_p"],
        "rank_mean": float(np.nanmean(rk)),
        "rank_random_mean": float(np.nanmean(rk_r)),
        "rank_p": res["rank_p"],
        "elapsed_seconds": round(elapsed, 1),
    }
    out = {"args": vars(args), "summary": summary,
           "jaccard": res["jaccard"], "rank": res["rank"],
           "jaccard_random": res["jaccard_random"], "rank_random": res["rank_random"]}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Jaccard(Top-{args.top_k}): {summary['jaccard_mean']:.3f} vs "
          f"随机 {summary['jaccard_random_mean']:.3f} (p={summary['jaccard_p']:.4f})")
    print(f"越限线注意力秩: {summary['rank_mean']:.3f} vs "
          f"随机 {summary['rank_random_mean']:.3f} (p={summary['rank_p']:.4f})；"
          f"耗时 {elapsed:.0f}s")
    print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
