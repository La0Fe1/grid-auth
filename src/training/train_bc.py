"""IL 预热（FA-002 第六轴）：行为克隆专家示范 → GNNGraphPolicy 权重。

用法（项目根目录）：
  python -m src.training.train_bc --demos data/processed/expert_demos.pkl \
      --out experiments/exp_002/checkpoints/bc_pretrain.pt --epochs 50
之后用 train_ppo --pretrain experiments/exp_002/checkpoints/bc_pretrain.pt 做 RL 微调。
"""
import argparse
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from gymnasium import spaces

from src.models.gnn_policy import GNNGraphPolicy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", default="data/processed/expert_demos.pkl")
    ap.add_argument("--out", default="experiments/exp_002/checkpoints/bc_pretrain.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    with open(args.demos, "rb") as f:
        data = pickle.load(f)
    meta, demos = data["meta"], data["demos"]
    if args.smoke:
        demos = demos[:64]
    n_sub, n_line = meta["n_sub"], meta["n_line"]

    obs_sp = spaces.Dict({
        "node_feat": spaces.Box(-1e6, 1e6, (n_sub, 3), dtype=np.float32),
        "edge_feat": spaces.Box(-1e6, 1e6, (n_line, 2), dtype=np.float32),
    })
    act_sp = spaces.MultiBinary(n_line)
    policy = GNNGraphPolicy(obs_sp, act_sp, lambda _: args.lr,
                            edge_index=meta["edge_index"], hidden=args.hidden)

    Xn = torch.tensor(np.array([d[0] for d in demos]), dtype=torch.float32)
    Xe = torch.tensor(np.array([d[1] for d in demos]), dtype=torch.float32)
    Y = torch.tensor(np.array([d[2] for d in demos]), dtype=torch.float32)

    # 正样本权重（缓解类不平衡）
    pos_rate = float(Y.mean())
    pos_w = torch.tensor([max((1 - pos_rate) / max(pos_rate, 1e-6), 1.0)])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    n = len(demos)
    idx = np.random.RandomState(0).permutation(n)
    n_tr = int(n * 0.9)
    tr_idx, te_idx = idx[:n_tr], idx[n_tr:]

    for epoch in range(2 if args.smoke else args.epochs):
        policy.train()
        order = np.random.permutation(n_tr)
        for i in range(0, n_tr, args.batch):
            b = order[i:i + args.batch]
            obs = {"node_feat": Xn[tr_idx[b]], "edge_feat": Xe[tr_idx[b]]}
            logits, _ = policy._encode(obs)
            loss = loss_fn(logits, Y[tr_idx[b]])
            opt.zero_grad()
            loss.backward()
            opt.step()
        policy.eval()
        with torch.no_grad():
            obs_te = {"node_feat": Xn[te_idx], "edge_feat": Xe[te_idx]}
            logits_te, _ = policy._encode(obs_te)
            te_loss = float(loss_fn(logits_te, Y[te_idx]))
            te_acc = float(((torch.sigmoid(logits_te) > 0.5).float() == Y[te_idx]).float().mean())
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch}: loss={float(loss):.4f} te_loss={te_loss:.4f} te_acc={te_acc:.4f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(policy.state_dict(), args.out)
    print(f"BC 预热权重已保存: {args.out}（正样本率 {pos_rate:.3f}）")


if __name__ == "__main__":
    main()
