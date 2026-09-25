"""采集 N-1 贪心专家示范（FA-002 IL 轴数据源）。

产出 {"meta": 图元信息, "demos": [(node_feat, edge_feat, action)]}：
正样本（专家 toggle）全保留；负样本（专家不动）按 --neg-ratio 欠采样，
避免全零样本淹没（否则 BC 只会学到"永不动作"）。

用法（项目根目录）：
  python -m src.training.gen_expert_demos --out data/processed/expert_demos.pkl
"""
import argparse
import pickle
import time

import numpy as np

from src.baselines.n1_greedy import N1GreedyAgent
from src.utils.env import Grid2OpGraphEnv
from src.utils.graph import obs_to_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/processed/expert_demos.pkl")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--neg-ratio", type=float, default=5.0,
                    help="负样本/正样本保留比例（欠采样）")
    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        args.n_episodes = 2

    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    agent = N1GreedyAgent(env, rho_threshold=0.85, max_candidates=10, act_every=5)

    pos, neg = [], []
    t0 = time.time()
    for ep in range(args.n_episodes):
        env.set_id(ep % 100)
        env.reset()
        obs_glop = env.env_glop.get_obs()
        nf, ef = obs_to_graph(obs_glop, env.meta)
        steps, done = 0, False
        while not done and steps < args.max_steps:
            action = agent.act({"node_feat": nf, "edge_feat": ef})
            (pos if action.sum() > 0 else neg).append((nf.copy(), ef.copy(), action.copy()))
            # raw grid2op step 返回 4 元组（obs, reward, done, info）
            obs_glop, r, done, info = env.env_glop.step(
                env.env_glop.action_space({"change_line_status":
                                           [int(i) for i, v in enumerate(action) if v > 0.5]}))
            nf, ef = obs_to_graph(obs_glop, env.meta)
            steps += 1
        print(f"ep{ep}: {steps} 步, 累计正样本 {len(pos)}")

    # 负样本欠采样
    keep_neg = min(len(neg), int(len(pos) * args.neg_ratio))
    if keep_neg < len(neg):
        rng = np.random.RandomState(0)
        neg = [neg[i] for i in rng.choice(len(neg), keep_neg, replace=False)]
    demos = pos + neg
    rng = np.random.RandomState(0)
    rng.shuffle(demos)

    with open(args.out, "wb") as f:
        pickle.dump({"meta": env.meta, "demos": demos}, f)
    print(f"保存 {len(demos)} 条示范（正 {len(pos)} / 负 {len(neg)}）到 {args.out}，"
          f"耗时 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
