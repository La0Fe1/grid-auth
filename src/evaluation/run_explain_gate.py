"""H2 检验（学习门版）：N1Gate 的 GAT 注意力归因与 N-1 越限支路的一致性。

说明（FA-003 后）：策略网络的注意力因 RL 塌缩而无意义；H2 用学习门的注意力检验——
门专为 N-1 安全判据训练，其注意力最可能有解释力（"安全与解释共用同一图表示"）。

用法：python -m src.evaluation.run_explain_gate \
    --gate experiments/exp_003/checkpoints/n1_gate.pt \
    --out experiments/exp_004/results/explain_gate.json
"""
import argparse
import json
import os

import numpy as np
import torch

from src.evaluation.explain import (
    combined_attention, rank_metric, topk_jaccard, violated_lines,
)
from src.models.n1_gate import N1Gate
from src.utils.env import Grid2OpGraphEnv
from src.utils.graph import obs_to_graph
from src.utils.metrics import paired_wilcoxon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True)
    ap.add_argument("--case", default="rte_case14_realistic")
    ap.add_argument("--n-states", type=int, default=50)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--out", default="experiments/exp_004/results/explain_gate.json")
    args = ap.parse_args()

    env = Grid2OpGraphEnv(case=args.case, test=False)
    gate = N1Gate(env.meta["edge_index"])
    gate.load_state_dict(torch.load(args.gate, map_location="cpu"))
    gate.eval()

    rng = np.random.RandomState(42)
    jac, rk, jac_r, rk_r = [], [], [], []
    for i in range(args.n_states):
        env.env_glop.set_id(i % 50)
        env.env_glop.reset()
        env.env_glop.fast_forward_chronics((i % 5) * 100)
        obs_glop = env.env_glop.get_obs()
        nf, ef = obs_to_graph(obs_glop, env.meta)
        prob, attns = gate.encode_with_attention(
            torch.as_tensor(nf[None], dtype=torch.float32),
            torch.as_tensor(ef[None], dtype=torch.float32))
        attn = combined_attention(attns)[0]
        viol = violated_lines(env, obs_glop)
        jac.append(topk_jaccard(attn, viol, args.top_k))
        rk.append(rank_metric(attn, viol))
        jac_r.append(topk_jaccard(rng.permutation(attn), viol, args.top_k))
        rk_r.append(rank_metric(rng.permutation(attn), viol))

    _, jac_p = paired_wilcoxon(jac, jac_r)
    _, rk_p = paired_wilcoxon(rk, rk_r)
    res = {
        "args": vars(args),
        "jaccard_mean": float(np.nanmean(jac)),
        "jaccard_random_mean": float(np.nanmean(jac_r)),
        "jaccard_p": jac_p,
        "rank_mean": float(np.nanmean(rk)),
        "rank_random_mean": float(np.nanmean(rk_r)),
        "rank_p": rk_p,
        "jaccard": jac, "rank": rk, "jaccard_random": jac_r, "rank_random": rk_r,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Jaccard(Top-{args.top_k}): {res['jaccard_mean']:.3f} vs "
          f"随机 {res['jaccard_random_mean']:.3f} (p={jac_p:.4f})")
    print(f"越限线注意力秩: {res['rank_mean']:.3f} vs "
          f"随机 {res['rank_random_mean']:.3f} (p={rk_p:.4f})")
    print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
