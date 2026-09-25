"""GNN-PPO 训练（exp_002 主实验训练器）。

铁律（AGENTS.md 2.1）：深度强化学习训练 ≥4 小时连续运行，
每 10 分钟保存 checkpoint（TimeCheckpointCallback）+ EvalCallback 周期评估写 CSV。
实验数值只能取自本脚本产出的日志文件（experiment_log / EvalCallback CSV）。

可选硬屏蔽训练（exp_003 消融"硬屏蔽"变体）：
  --shield PATH --alpha 0.02 → ShieldedGraphEnv 包装环境。

用法（在项目根目录下以模块方式运行）：
  python -m src.training.train_ppo --case rte_case14_realistic \
      --train-steps 500000 --seed 42 --out experiments/exp_002/checkpoints
  冒烟测试：--smoke（极小步数验证 pipeline 无语法/维度错误）
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback

from src.models.gnn_policy import GNNGraphPolicy
from src.models.n1_gate import N1Gate
from src.models.shield import N1Shield
from src.utils.env import Grid2OpGraphEnv, ShieldedGraphEnv


class TimeCheckpointCallback(BaseCallback):
    """每 save_interval_sec 秒保存一次模型（'每 10 分钟 checkpoint'铁律）。"""

    def __init__(self, save_dir, save_interval_sec=600, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.interval = save_interval_sec
        self._last = None

    def _on_step(self):
        now = time.time()
        if self._last is None:
            self._last = now
            return True
        if now - self._last >= self.interval:
            self.model.save(os.path.join(self.save_dir, "ckpt_latest"))
            self._last = now
        return True


class StepCheckpointCallback(BaseCallback):
    """每 save_freq_steps 步保存一个带步数编号的 checkpoint（不覆盖）。

    用途（FA-002 最后一搏）：PPO 后期塌缩时，好策略可能只存在于训练中段；
    用验证集协议评估在各 step checkpoint 中选择最优模型（标准模型选择实践）。
    """

    def __init__(self, save_dir, save_freq_steps=25_000, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.freq = save_freq_steps
        self._next = save_freq_steps

    def _on_step(self):
        if self.num_timesteps >= self._next:
            self.model.save(os.path.join(self.save_dir, f"ckpt_{int(self.num_timesteps)}"))
            self._next += self.freq
        return True


def make_env(case, test, shield=None, features="basic", attack_enabled=True,
             attack_budget_scale=1.0, mask=False, mask_rho=0.9, mask_max=5):
    env = Grid2OpGraphEnv(case=case, test=test, features=features,
                          attack_enabled=attack_enabled,
                          attack_budget_scale=attack_budget_scale)
    if mask:
        from src.utils.env import MaskedGraphEnv
        env = MaskedGraphEnv(env, rho_th=mask_rho, max_candidates=mask_max)
    if shield is not None:
        env = ShieldedGraphEnv(env, shield)
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="rte_case14_realistic")
    ap.add_argument("--test-mode", action="store_true", help="使用 test 环境（仅调试用）")
    ap.add_argument("--train-steps", type=int, default=500_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--shield", default=None, help="N1Gate checkpoint 路径（启用硬屏蔽训练）")
    ap.add_argument("--alpha", type=float, default=0.0, help="屏蔽门误放预算")
    ap.add_argument("--features", choices=["basic", "rich"], default="basic",
                    help="图特征版本（exp_006 消融维度 2）")
    ap.add_argument("--curriculum", type=int, default=0,
                    help="课程学习：前 N 步用无攻击环境训练（0=关闭）")
    ap.add_argument("--ent-coef", type=float, default=0.0,
                    help="PPO 熵正则系数（FA-002 改进轴：防确定性塌缩）")
    ap.add_argument("--attack-budget-scale", type=float, default=1.0,
                    help="对手攻击预算倍率（FA-002 改进轴：>1 削弱 do-nothing 基线）")
    ap.add_argument("--mask", action="store_true",
                    help="候选掩码动作空间（FA-002 第五轴：rho>0.9 至多 5 条线路）")
    ap.add_argument("--mask-rho", type=float, default=0.9)
    ap.add_argument("--mask-max", type=int, default=5)
    ap.add_argument("--resume", default=None,
                    help="从 checkpoint zip 继续训练（中断恢复：--train-steps 为总目标步数）")
    ap.add_argument("--pretrain", default=None,
                    help="BC 预热权重 .pt（IL 轴：加载后开始 RL 微调）")
    ap.add_argument("--step-checkpoints", action="store_true",
                    help="每 25k 步存带编号 checkpoint（不覆盖，供模型选择）")
    ap.add_argument("--out", default="experiments/exp_002/checkpoints")
    ap.add_argument("--eval-freq", type=int, default=20_000)
    ap.add_argument("--n-eval-episodes", type=int, default=5)
    ap.add_argument("--tb", action="store_true", help="启用 tensorboard 日志（需安装 tensorboard）")
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：64 步训练验证 pipeline")
    args = ap.parse_args()

    if args.smoke:
        args.train_steps, args.eval_freq, args.n_eval_episodes = 64, 32, 1

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    shield = None
    if args.shield is not None:
        from src.utils.env import DEFAULT_DATASET_PATH, _resolve_dataset
        from grid2op import make as g2op_make
        from lightsim2grid import LightSimBackend
        from src.utils.graph import build_graph_meta
        probe_env = g2op_make(dataset=_resolve_dataset(args.case, DEFAULT_DATASET_PATH),
                              backend=LightSimBackend())
        gate = N1Gate(build_graph_meta(probe_env)["edge_index"], hidden=args.hidden)
        gate.load_state_dict(torch.load(args.shield, map_location=args.device))
        gate.eval()
        shield = N1Shield(gate, alpha=args.alpha, device=args.device)
        print(f"启用硬屏蔽训练：gate={args.shield}, alpha={args.alpha}")

    train_env = make_env(args.case, args.test_mode, shield=shield, features=args.features,
                         attack_budget_scale=args.attack_budget_scale,
                         mask=args.mask, mask_rho=args.mask_rho, mask_max=args.mask_max)
    eval_env = make_env(args.case, args.test_mode, shield=shield, features=args.features,
                        attack_budget_scale=args.attack_budget_scale,
                        mask=args.mask, mask_rho=args.mask_rho, mask_max=args.mask_max)
    eval_env.seed(args.seed)  # 评估场景采样按种子固定（否则所有种子的 eval 相同）

    if args.resume:
        # 中断恢复：SB3 加载后 num_timesteps 从保存值继续，learn(总目标) 只补剩余步数
        model = PPO.load(args.resume, env=train_env, device=args.device)
        # 防覆盖缺陷（FA-002）：恢复后 EvalCallback 计数器重置，可能用退化模型
        # 覆盖历史最优——先把现有 best_model.zip 备份。
        best_path = os.path.join(args.out, "best_model.zip")
        if os.path.exists(best_path):
            import shutil
            shutil.copy(best_path, os.path.join(args.out, "best_model_pre_resume.zip"))
        print(f"从 {args.resume} 恢复（已训练 {model.num_timesteps} 步，"
              f"目标 {args.train_steps} 步）")
    else:
        model = PPO(
            GNNGraphPolicy,
            train_env,
            policy_kwargs={"edge_index": train_env.edge_index, "hidden": args.hidden,
                           "node_dim": train_env.node_dim, "edge_dim": train_env.edge_dim},
            learning_rate=args.lr,
            ent_coef=args.ent_coef,
            seed=args.seed,
            verbose=1,
            device=args.device,
            tensorboard_log=os.path.join(args.out, "tb") if args.tb else None,
        )
        if args.pretrain:
            model.policy.load_state_dict(torch.load(args.pretrain, map_location="cpu"))
            print(f"已加载 BC 预热权重: {args.pretrain}")

    t_start = time.time()
    callbacks = [
        TimeCheckpointCallback(args.out, save_interval_sec=600),
        EvalCallback(
            eval_env,
            best_model_save_path=args.out,
            log_path=args.out,
            eval_freq=args.eval_freq,
            n_eval_episodes=args.n_eval_episodes,
            deterministic=True,
        ),
    ]
    if args.step_checkpoints:
        callbacks.append(StepCheckpointCallback(args.out, save_freq_steps=25_000))

    if args.curriculum > 0:
        # 课程学习：先无攻击环境（学拓扑操作），再切带攻击环境
        cur_env = make_env(args.case, args.test_mode, shield=shield,
                           features=args.features, attack_enabled=False)
        cur_steps = min(args.curriculum, args.train_steps)
        model.learn(total_timesteps=cur_steps, progress_bar=True)
        model.set_env(train_env)
        print(f"课程阶段完成（{cur_steps} 步无攻击），切换带攻击环境")
        model.learn(total_timesteps=args.train_steps - cur_steps,
                    callback=callbacks, progress_bar=True)
    else:
        model.learn(total_timesteps=args.train_steps, callback=callbacks, progress_bar=True)
    elapsed = time.time() - t_start

    model.save(os.path.join(args.out, "final_model"))
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump({
            "args": vars(args),
            "elapsed_seconds": round(elapsed, 1),
            "resumed_from": args.resume,
            "total_timesteps": int(model.num_timesteps),
            "n_params": model.policy.count_parameters(),
            "start_unix": t_start,
        }, f, indent=2)
    print(f"训练完成：{elapsed:.0f}s（≥4h={elapsed >= 14400}），"
          f"参数量={model.policy.count_parameters()}")


if __name__ == "__main__":
    main()
