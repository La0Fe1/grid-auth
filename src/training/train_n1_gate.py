"""N1Gate 监督训练（exp_003 安全柱：学习式 N-1 可行性门）。

训练数据：data/processed/n1_gate_labels.pkl（gen_n1_labels.py 产出）。
输出：out/n1_gate.pt（val AUC 最优权重）+ out/train_log.csv（loss/acc/balanced-acc/AUC）。

用法（在项目根目录下以模块方式运行）：
  python -m src.training.train_n1_gate --labels data/processed/n1_gate_labels.pkl \
      --out experiments/exp_003/checkpoints --epochs 300 --seed 42
  冒烟测试：--smoke
"""
import argparse
import csv
import json
import os
import pickle

import numpy as np
import torch
import torch.nn as nn

from src.models.n1_gate import N1Gate
from src.utils.metrics import roc_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/processed/n1_gate_labels.pkl")
    ap.add_argument("--out", default="experiments/exp_003/checkpoints")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：2 epoch 极小数据")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    with open(args.labels, "rb") as f:
        data = pickle.load(f)
    meta, samples = data["meta"], data["samples"]
    if args.smoke:
        samples = samples[:8]
    X_node = torch.tensor(np.array([s[0] for s in samples]), dtype=torch.float32)
    X_edge = torch.tensor(np.array([s[1] for s in samples]), dtype=torch.float32)
    Y = torch.tensor(np.array([s[2] for s in samples]), dtype=torch.float32)

    n = len(samples)
    idx = np.random.permutation(n)
    n_tr = int(n * 0.8)
    tr_idx, te_idx = idx[:n_tr], idx[n_tr:]
    Y_tr, Y_te = Y[tr_idx], Y[te_idx]

    model = N1Gate(meta["edge_index"], hidden=args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    # N1Gate.forward 输出 sigmoid 概率，直接 BCELoss（torch 内部有数值截断保护）
    loss_fn = nn.BCELoss()

    epochs = 2 if args.smoke else args.epochs
    log_path = os.path.join(args.out, "train_log.csv")
    best_auc, best_state = -1.0, None

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss", "val_acc", "val_balanced_acc", "val_auc"])
        for epoch in range(epochs):
            model.train()
            prob_tr = model(X_node[tr_idx], X_edge[tr_idx])
            loss = loss_fn(prob_tr, Y_tr)
            opt.zero_grad()
            loss.backward()
            opt.step()

            model.eval()
            with torch.no_grad():
                prob_te = model(X_node[te_idx], X_edge[te_idx])
                val_loss = float(loss_fn(prob_te, Y_te))
                pred = (prob_te > 0.5).float()
                acc = float((pred == Y_te).float().mean())
                pos_te = (Y_te == 1).float().mean()
                neg_te = 1 - pos_te
                bal = float(((pred[Y_te == 1] == 1).float().mean() +
                             (pred[Y_te == 0] == 0).float().mean()) / 2)
                auc = roc_auc(Y_te.numpy().astype(bool).ravel(), prob_te.numpy().ravel())
            if auc > best_auc:
                best_auc, best_state = auc, {k: v.clone() for k, v in model.state_dict().items()}
            w.writerow([epoch, float(loss), val_loss, acc, bal, auc])
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"epoch {epoch}: loss={float(loss):.4f} val_acc={acc:.3f} "
                      f"val_bal={bal:.3f} val_auc={auc:.3f}")

    torch.save(best_state, os.path.join(args.out, "n1_gate.pt"))
    with open(os.path.join(args.out, "train_meta.json"), "w") as f:
        json.dump({"args": vars(args), "n_samples": n, "best_val_auc": best_auc}, f, indent=2)
    print(f"保存最优门（val AUC={best_auc:.3f}）到 {os.path.join(args.out, 'n1_gate.pt')}")


if __name__ == "__main__":
    main()
