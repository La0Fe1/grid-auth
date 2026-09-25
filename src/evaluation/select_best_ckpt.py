"""FA-002 最后一搏：检查点选择（验证集选最优，测试集报告）。

背景：PPO 后期塌缩，好策略可能只存在于训练中段（证据：seed44 在 100k 步达
3192 分后塌缩）。标准模型选择实践：每 25k 步保存带编号 checkpoint
（train_ppo --step-checkpoints），在验证集（场景 100-119）上逐个评估选最优，
再在测试集（场景 0-19）上报告——选择与报告严格分离。

用法：
  python -m src.evaluation.select_best_ckpt \
      --ckpt-dir experiments/exp_002/checkpoints/sel_base_seed42 \
      --out experiments/exp_002/results/sel_base_seed42.json
"""
import argparse
import glob
import json
import os

import numpy as np
from stable_baselines3 import PPO

from src.evaluation.agents import SB3GraphAgent
from src.evaluation.evaluate import run_agent_scenarios
from src.utils.env import Grid2OpGraphEnv

VAL_OFFSET = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--case", default="rte_case14_realistic")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--scenarios-per-seed", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    all_ckpts = glob.glob(os.path.join(args.ckpt_dir, "ckpt_*.zip"))
    # 排除时间型 ckpt_latest（每 10 分钟覆盖保存），只留带步数编号的选择型 checkpoint
    ckpts = sorted([p for p in all_ckpts if os.path.basename(p) != "ckpt_latest.zip"],
                   key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    if not ckpts:
        raise FileNotFoundError(f"{args.ckpt_dir} 中没有带步数的 ckpt_*.zip"
                                f"（训练时需 --step-checkpoints）")

    def env_factory():
        return Grid2OpGraphEnv(case=args.case, test=False)

    def val_survival(model):
        scen = run_agent_scenarios(lambda env: SB3GraphAgent(model), env_factory,
                                   args.seeds, args.scenarios_per_seed,
                                   id_offset=VAL_OFFSET)
        return float(np.mean([s.survival for s in scen.values()]))

    results = []
    for p in ckpts:
        model = PPO.load(p, device="cpu")
        vs = val_survival(model)
        results.append({"ckpt": os.path.basename(p), "val_survival": vs})
        print(f"{os.path.basename(p)}: val survival {vs:.1f}")

    best = max(results, key=lambda r: r["val_survival"])
    print(f"最优 checkpoint: {best['ckpt']}（验证集存活 {best['val_survival']:.1f}）")
    best_model = PPO.load(os.path.join(args.ckpt_dir, best["ckpt"]), device="cpu")
    test_scen = run_agent_scenarios(lambda env: SB3GraphAgent(best_model), env_factory,
                                    args.seeds, args.scenarios_per_seed, id_offset=0)
    test_per_seed = {}
    for s in args.seeds:
        vals = [test_scen[s * args.scenarios_per_seed + k].survival
                for k in range(args.scenarios_per_seed)]
        test_per_seed[str(s)] = {"survival_mean": float(np.mean(vals))}
    test_survival = float(np.mean([s.survival for s in test_scen.values()]))

    out = {"args": vars(args), "ckpt_results": results,
           "best_ckpt": best["ckpt"], "best_val_survival": best["val_survival"],
           "test_survival_mean": test_survival,
           "test_per_seed": test_per_seed,
           "test_per_scenario": {str(k): vars(s) for k, s in test_scen.items()}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"测试集存活均值: {test_survival:.1f}（do-nothing 基准 1100.7）")
    print(f"结果已保存: {args.out}")


if __name__ == "__main__":
    main()
