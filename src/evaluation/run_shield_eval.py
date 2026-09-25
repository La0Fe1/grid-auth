"""exp_003 混合屏蔽器评估（H1）：危险动作率、学习门削减率、性能代价。

协议：
- 加载训练好的策略 + HybridShield（学习门初筛 + 精确仿真终审）；
- 5 种子 × 4 场景跑幕，统计：
  * proposed: 策略请求的 toggle 总数
  * blocked_gate: 学习门初筛挡掉的 toggle 数
  * n_exact_checks: 精确终审执行的仿真校验次数（= 门放行数）
  * blocked_exact: 精确层挡掉的 toggle 数
  * n_illegal: 最终动作被 grid2op 拒绝的次数（应与危险动作率=0 相配合）
  * 削减率 = 1 - n_exact_checks / proposed（H1 新判据：≥50%）
- 对比：无屏蔽版本（shield=None）在同样种子/场景下的存活时间与 toggle 数 → 性能代价。

用法：
  python -m src.evaluation.run_shield_eval --model PATH --gate PATH --alpha 0.1 \
      --out experiments/exp_003/results/shield_eval.json
"""
import argparse
import json
import os

import numpy as np
from stable_baselines3 import PPO

from src.evaluation.agents import SB3GraphAgent
from src.evaluation.evaluate import run_agent_scenarios
from src.models.n1_gate import N1Gate
from src.models.shield import HybridShield
from src.utils.env import Grid2OpGraphEnv, ShieldedGraphEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--case", default="rte_case14_realistic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--scenarios-per-seed", type=int, default=4)
    ap.add_argument("--out", default="experiments/exp_003/results/shield_eval.json")
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")

    # 混合屏蔽环境
    gate_env = Grid2OpGraphEnv(case=args.case, test=False)
    gate = N1Gate(gate_env.meta["edge_index"])
    gate.load_state_dict(torch_load(args.gate))
    gate.eval()
    shield = HybridShield(gate, alpha=args.alpha, device="cpu")

    def shielded_factory():
        return ShieldedGraphEnv(Grid2OpGraphEnv(case=args.case, test=False), shield)

    def plain_factory():
        return Grid2OpGraphEnv(case=args.case, test=False)

    scen_sh = run_agent_scenarios(lambda env: SB3GraphAgent(model), shielded_factory,
                                  args.seeds, args.scenarios_per_seed)
    # 累积屏蔽统计（info 里没有返回，从盾对象取）
    n_exact_checks = shield.total_exact_checks

    # 对照：无屏蔽版本（同一模型、同一场景）
    scen_pl = run_agent_scenarios(lambda env: SB3GraphAgent(model), plain_factory,
                                  args.seeds, args.scenarios_per_seed)

    def agg(scen, key):
        return float(np.mean([getattr(s, key) for s in scen.values()]))

    result = {
        "args": vars(args),
        "shielded": {
            "survival_mean": agg(scen_sh, "survival"),
            "reward_mean": agg(scen_sh, "reward"),
            "n_illegal_mean": agg(scen_sh, "n_illegal"),
            "n_toggles_mean": agg(scen_sh, "n_toggles"),   # 屏蔽后实际 toggle
            "n_blocks_mean": agg(scen_sh, "n_blocks"),
        },
        "plain": {
            "survival_mean": agg(scen_pl, "survival"),
            "reward_mean": agg(scen_pl, "reward"),
            "n_illegal_mean": agg(scen_pl, "n_illegal"),
            "n_toggles_mean": agg(scen_pl, "n_toggles"),
        },
        "shield_stats": {
            "total_exact_checks": int(n_exact_checks),
            "reduction_rate": None,
        },
    }
    proposed = agg(scen_sh, "n_proposed") * len(scen_sh)
    if proposed > 0:
        result["shield_stats"]["reduction_rate"] = float(1.0 - n_exact_checks / proposed)
    result["perf_cost_survival"] = (1 - result["shielded"]["survival_mean"]
                                    / max(result["plain"]["survival_mean"], 1e-9))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2, ensure_ascii=False)[:1200])
    print(f"结果已保存: {args.out}")


def torch_load(path):
    import torch
    return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    main()
