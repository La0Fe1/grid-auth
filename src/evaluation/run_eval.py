"""评估入口 CLI：把智能体跑 N 幕（每幕固定种子），保存/打印指标（exp_001/002/005/006/007 共用）。

所有数值来自环境返回值，并原样写入 JSON 结果文件（反幻觉铁律 2：论文数值只许引自该文件）。

用法（项目根目录，模块方式运行）：
  python -m src.evaluation.run_eval --agent do_nothing --seeds 0 1 2 3 4 \
      --out experiments/exp_001/results/do_nothing.json
  python -m src.evaluation.run_eval --agent n1_greedy --rho-threshold 0.9 --seeds 0 1
  python -m src.evaluation.run_eval --agent sb3 --model PATH/final_model.zip --seeds 0 1 2 3 4
"""
import argparse
import json
import os
import time

from src.evaluation.agents import SB3GraphAgent
from src.evaluation.evaluate import aggregate_per_seed, run_agent_scenarios, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True, choices=["do_nothing", "n1_greedy", "sb3"])
    ap.add_argument("--model", default=None, help="sb3 智能体：模型 zip 路径")
    ap.add_argument("--case", default="rte_case14_realistic")
    ap.add_argument("--test-mode", action="store_true")
    ap.add_argument("--flatten", action="store_true",
                    help="评估 MLP 基线时使用拍平观测（FlattenGraphEnv）")
    ap.add_argument("--mask", action="store_true",
                    help="评估掩码动作空间模型（与训练时一致的候选掩码）")
    ap.add_argument("--mask-rho", type=float, default=0.9)
    ap.add_argument("--mask-max", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--scenarios-per-seed", type=int, default=4,
                    help="每种子评估的场景数（场景 id = seed*N + k）")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default=None, help="结果 JSON 路径（缺省不落盘，仅打印）")
    # n1_greedy 参数
    ap.add_argument("--rho-threshold", type=float, default=0.9)
    ap.add_argument("--max-candidates", type=int, default=10)
    args = ap.parse_args()

    from stable_baselines3 import PPO

    from src.baselines.do_nothing import DoNothingAgent
    from src.baselines.n1_greedy import N1GreedyAgent
    from src.utils.env import FlattenGraphEnv, Grid2OpGraphEnv

    def env_factory():
        env = Grid2OpGraphEnv(case=args.case, test=args.test_mode)
        if args.mask:
            from src.utils.env import MaskedGraphEnv
            env = MaskedGraphEnv(env, rho_th=args.mask_rho, max_candidates=args.mask_max)
        return FlattenGraphEnv(env) if args.flatten else env

    if args.agent == "do_nothing":
        agent_factory = lambda env: DoNothingAgent(env=env)
    elif args.agent == "n1_greedy":
        agent_factory = lambda env: N1GreedyAgent(env, rho_threshold=args.rho_threshold,
                                                  max_candidates=args.max_candidates)
    else:
        model = PPO.load(args.model)
        agent_factory = lambda env: SB3GraphAgent(model)

    t0 = time.time()
    scen = run_agent_scenarios(agent_factory, env_factory, seeds=args.seeds,
                               scenarios_per_seed=args.scenarios_per_seed,
                               max_steps=args.max_steps)
    stats = aggregate_per_seed(scen, args.seeds, args.scenarios_per_seed)
    elapsed = time.time() - t0

    summary = summarize(stats)
    per_seed = {str(s): vars(st) for s, st in zip(args.seeds, stats)}
    per_scenario = {str(k): vars(st) for k, st in scen.items()}
    print(f"[{args.agent}] {len(args.seeds)} 种子 × {args.scenarios_per_seed} 场景，"
          f"耗时 {elapsed:.0f}s")
    for k, v in summary.items():
        print(f"  {k}: {v['mean']:.1f} ± {v['std']:.1f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"args": vars(args), "summary": summary, "per_seed": per_seed,
                       "per_scenario": per_scenario,
                       "elapsed_seconds": round(elapsed, 1)}, f, indent=2)
        print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
