"""基线 B1 训练：MLP 策略（拍平图观测），无屏蔽。

与 train_ppo 共用 TimeCheckpointCallback（10 分钟 checkpoint 铁律）。
MLP 观测维度与算例绑定（n_sub*3 + n_line*2），无法跨算例迁移——
这是三柱方法与之的核心差别（H3 对照实验）。

用法（项目根目录，模块方式运行）：
  python -m src.baselines.train_mlp_ppo --case rte_case14_realistic \
      --train-steps 500000 --seed 42 --out experiments/exp_002/checkpoints/mlp_seed42
  冒烟测试：--smoke
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from src.training.train_ppo import TimeCheckpointCallback
from src.utils.env import FlattenGraphEnv, Grid2OpGraphEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="rte_case14_realistic")
    ap.add_argument("--test-mode", action="store_true")
    ap.add_argument("--train-steps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/exp_002/checkpoints/mlp_baseline")
    ap.add_argument("--eval-freq", type=int, default=20_000)
    ap.add_argument("--n-eval-episodes", type=int, default=5)
    ap.add_argument("--resume", default=None,
                    help="从 checkpoint zip 继续训练（中断恢复：--train-steps 为总目标步数）")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.train_steps, args.eval_freq, args.n_eval_episodes = 64, 32, 1

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_env = FlattenGraphEnv(Grid2OpGraphEnv(case=args.case, test=args.test_mode))
    eval_env = FlattenGraphEnv(Grid2OpGraphEnv(case=args.case, test=args.test_mode))
    eval_env.seed(args.seed)  # 评估场景采样按种子固定（否则所有种子的 eval 相同）

    if args.resume:
        model = PPO.load(args.resume, env=train_env, device=args.device)
        print(f"从 {args.resume} 恢复（已训练 {model.num_timesteps} 步，"
              f"目标 {args.train_steps} 步）")
    else:
        model = PPO("MlpPolicy", train_env, seed=args.seed, verbose=1, device=args.device)
    t_start = time.time()
    callbacks = [
        TimeCheckpointCallback(args.out, save_interval_sec=600),
        EvalCallback(eval_env, best_model_save_path=args.out, log_path=args.out,
                     eval_freq=args.eval_freq, n_eval_episodes=args.n_eval_episodes,
                     deterministic=True),
    ]
    model.learn(total_timesteps=args.train_steps, callback=callbacks, progress_bar=True)
    elapsed = time.time() - t_start

    model.save(os.path.join(args.out, "final_model"))
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump({"args": vars(args), "elapsed_seconds": round(elapsed, 1),
                   "resumed_from": args.resume,
                   "total_timesteps": int(model.num_timesteps),
                   "start_unix": t_start}, f, indent=2)
    print(f"MLP 基线训练完成：{elapsed:.0f}s（≥4h={elapsed >= 14400}）")


if __name__ == "__main__":
    main()
