"""exp_005 跨算例零样本泛化评估（H3）。

输入：14-bus（rte_case14_realistic）训练的模型；
输出：14（训练算例）/36（l2rpn_icaps_2021）/118（l2rpn_idf_2023）三个系统上的
逐环境对比（迁移策略 vs do-nothing，配对 Wilcoxon p 值），JSON 落盘。

跨算例部署机制（H3 核心）：用目标环境的 edge_index 重建 GNNGraphPolicy
并加载 14-bus 权重——GNN 层与 per-edge readout 的参数只依赖特征维度，
与边数无关（单元测试 test_param_count_invariance_to_n_line 已固化该性质）。
MLP 基线因输入维度绑定（501/1363/4460 维）无法部署，表内如实标注。

注意：36/118-bus 使用内置开发模式（test=True，见 data/get_data.py 说明）。
"""
import argparse
import json
import os
import time

import numpy as np
from stable_baselines3 import PPO

from src.baselines.do_nothing import DoNothingAgent
from src.evaluation.agents import PolicyAgent, SB3GraphAgent
from src.evaluation.evaluate import compare_paired, run_agent, summarize
from src.models.gnn_policy import GNNGraphPolicy
from src.utils.env import Grid2OpGraphEnv

CASES = {
    "14-bus(train)": ("rte_case14_realistic", False),
    "36-bus(zero-shot)": ("l2rpn_icaps_2021", True),
    "118-bus(zero-shot)": ("l2rpn_idf_2023", True),
}


def transfer_policy(model, env):
    """用目标环境的 edge_index 重建策略并加载训练权重（H3 部署机制）。"""
    policy = GNNGraphPolicy(env.observation_space, env.action_space,
                            lambda _: 3e-4, edge_index=env.edge_index)
    policy.load_state_dict(model.policy.state_dict(), strict=False)
    policy.eval()
    return policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="14-bus 训练模型 zip")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--scenarios-per-seed", type=int, default=4)
    ap.add_argument("--out", default="experiments/exp_005/results/generalize.json")
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    results = {}
    for name, (case, test) in CASES.items():
        env_factory = lambda case=case, test=test: Grid2OpGraphEnv(case=case, test=test)
        t0 = time.time()
        if name.endswith("(train)"):
            gnn_stats = run_agent(lambda env: SB3GraphAgent(model), env_factory,
                                  args.seeds, args.scenarios_per_seed)
        else:
            gnn_stats = run_agent(
                lambda env: PolicyAgent(transfer_policy(model, env)), env_factory,
                args.seeds, args.scenarios_per_seed)
        dn_stats = run_agent(lambda env: DoNothingAgent(env=env), env_factory,
                             args.seeds, args.scenarios_per_seed)
        stat, p = compare_paired(gnn_stats, dn_stats, field="survival")
        results[name] = {
            "gnn": {k: v["mean"] for k, v in summarize(gnn_stats).items()},
            "do_nothing": {k: v["mean"] for k, v in summarize(dn_stats).items()},
            "wilcoxon_p_survival": p,
        }
        print(f"{name}: GNN {results[name]['gnn']['survival']:.1f} vs "
              f"do-nothing {results[name]['do_nothing']['survival']:.1f} "
              f"(p={p:.4f}) [{time.time()-t0:.0f}s]")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=2)
    print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
