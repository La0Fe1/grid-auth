"""方向 E 可行性尖刺（exp_010）：PINN 调频轨迹代理 + 归因对照冒烟。

问题定义：输入 (t, H, R, P_step) → 三状态轨迹 [Δf, ΔPm, ΔPv]；
- PINN：数据损失（轨迹拟合）+ 物理损失（三方程 ODE 残差，t 的自动微分）；
- 数据驱动基线：同架构 MLP，无物理项；
- 归因：∂Δf(t*)/∂H 经梯度×输入 vs 模拟器有限差分真值，测 Pearson 相关（t*=0.5s 近 nadir）。

冒烟判据：①PINN 测试 RMSE ≤ 基线；②归因 Pearson ≥ 0.8；③loss 收敛。
全部通过 → 方向 E 技术可行（记录到 decisions.md）。

用法：python -m experiments.exp_010_freq_e.spike_pinn
"""
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

from src.utils.freq_dyn import DEFAULT_PARAMS, f as ode_f, simulate

T_HORIZON, DT = 20.0, 0.02
ATTRIB_T = 0.5


def gen_data(n=400, ts=None, seed=0):
    """H∈[3,8], R∈[0.03,0.08], P_step∈[0.05,0.2] 均匀网格采样轨迹。"""
    rng = np.random.RandomState(seed)
    if ts is None:
        ts = np.arange(0.0, T_HORIZON, DT)
    X, Y = [], []
    for _ in range(n):
        p = dict(DEFAULT_PARAMS, H=float(rng.uniform(3, 8)),
                 R=float(rng.uniform(0.03, 0.08)),
                 P_step=float(rng.uniform(0.05, 0.2)))
        t, traj = simulate(p, T=T_HORIZON, dt=DT)
        for i in range(0, len(ts), 5):  # 时间下采样控制规模
            X.append([ts[i], p["H"], p["R"], p["P_step"]])
            Y.append(traj[i, :])
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    return X, Y, ts


class PINN(nn.Module):
    """4 输入 (t,H,R,P_step) → 3 状态；物理损失用 t 的自动微分。"""

    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(),
                                 nn.Linear(hidden, 3))

    def forward(self, x):
        return self.net(x)

    def physics_residual(self, x, out):
        """三方程 ODE 残差（x 含 t 列，要求 x.requires_grad）。"""
        x.requires_grad_(True)
        out = self.net(x)
        df, dpm, dpv = out[:, 0], out[:, 1], out[:, 2]
        t_col = x[:, 0:1]
        df_t = torch.autograd.grad(df.sum(), x, create_graph=True)[0][:, 0]
        dpm_t = torch.autograd.grad(dpm.sum(), x, create_graph=True)[0][:, 0]
        dpv_t = torch.autograd.grad(dpv.sum(), x, create_graph=True)[0][:, 0]
        H, R = x[:, 1], x[:, 2]
        P_step = x[:, 3]
        r1 = 2 * H * df_t - (dpm - P_step - DEFAULT_PARAMS["D"] * df)
        r2 = DEFAULT_PARAMS["Tg"] * dpv_t - (DEFAULT_PARAMS["Pc"] - dpv - df / R)
        r3 = DEFAULT_PARAMS["Tt"] * dpm_t - (dpv - dpm)
        return torch.stack([r1, r2, r3], dim=1)


def _train_sub(pinn, Xs, Ys, lam, epochs=800, lr=1e-3):
    """在子数据集上训练（低数据实验用）。"""
    opt = torch.optim.Adam(pinn.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        pinn.train()
        out = pinn(Xs)
        data_loss = nn.functional.mse_loss(out, Ys)
        if lam > 0:
            res = pinn.physics_residual(Xs, out)
            phys_loss = (res ** 2).mean()
            g_d = torch.autograd.grad(data_loss, pinn.parameters(),
                                      retain_graph=True, create_graph=False)
            g_p = torch.autograd.grad(phys_loss, pinn.parameters(),
                                      retain_graph=True, create_graph=False)
            nd = sum(g.norm() for g in g_d if g is not None) + 1e-8
            np_ = sum(g.norm() for g in g_p if g is not None) + 1e-8
            w = (nd / np_).detach().clamp(0.01, 100.0)
            loss = data_loss + lam * w * phys_loss
        else:
            loss = data_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()


def attribution(model, x_test, param_col, xs_mean, xs_std):
    """梯度×输入归因：∂Δf(t*)/∂param，与有限差分真值对比。"""
    model.eval()
    x = torch.tensor(x_test, dtype=torch.float32)
    x.requires_grad_(True)
    out = model(x)
    t_star_idx = np.argmin(np.abs(x_test[:, 0] - ATTRIB_T))
    # 对每条样本：用 (t=t*) 的输出对 param 求导
    mask = (x_test[:, 0] == x_test[t_star_idx, 0])
    df_sel = out[mask][:, 0].sum()
    grad = torch.autograd.grad(df_sel, x, create_graph=False)[0][mask, param_col]
    return grad.detach().numpy() / xs_std[param_col]  # 去归一化缩放


def main():
    rng = np.random.RandomState(0)
    ts = np.arange(0.0, T_HORIZON, DT)
    X_all, Y_all, _ = gen_data(n=400, ts=ts)
    # 归一化
    xs_mean = X_all.mean(0)
    xs_std = X_all.std(0)
    xs_std[xs_std == 0] = 1.0
    Xn = (X_all - xs_mean) / xs_std
    Y_mean, Y_std = Y_all.mean(0), Y_all.std(0)
    Yn = (Y_all - Y_mean) / Y_std

    n = len(Xn)
    idx = rng.permutation(n)
    tr, te = idx[: int(n * 0.8)], idx[int(n * 0.8):]
    Xtr, Ytr = torch.tensor(Xn[tr]), torch.tensor(Yn[tr])
    Xte, Yte = torch.tensor(Xn[te]), torch.tensor(Yn[te])

    def train_model(pinn, epochs, lr, lam, balance=True):
        opt = torch.optim.Adam(pinn.parameters(), lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        hist = []
        for ep in range(epochs):
            pinn.train()
            out = pinn(Xtr)
            data_loss = nn.functional.mse_loss(out, Ytr)
            if lam > 0:
                res = pinn.physics_residual(Xtr, out)
                phys_loss = (res ** 2).mean()
                if balance:
                    # 梯度范数配平（PINN 经典做法）：动态权重按数据/物理梯度比
                    g_d = torch.autograd.grad(data_loss, pinn.parameters(),
                                              retain_graph=True, create_graph=False)
                    g_p = torch.autograd.grad(phys_loss, pinn.parameters(),
                                              retain_graph=True, create_graph=False)
                    nd = sum(g.norm() for g in g_d if g is not None) + 1e-8
                    np_ = sum(g.norm() for g in g_p if g is not None) + 1e-8
                    w = (nd / np_).detach().clamp(0.01, 100.0)
                    loss = data_loss + lam * w * phys_loss
                else:
                    loss = data_loss + lam * phys_loss
            else:
                loss = data_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            hist.append(float(loss))
        return hist

    t0 = time.time()
    # 全数据（400 轨迹）
    pinn = PINN()
    hist = train_model(pinn, epochs=800, lr=1e-3, lam=1.0, balance=True)
    pinn.eval()
    with torch.no_grad():
        rmse_pinn = float(torch.sqrt(nn.functional.mse_loss(pinn(Xte), Yte)))

    mlp = PINN()
    hist_mlp = train_model(mlp, epochs=800, lr=1e-3, lam=0.0)
    mlp.eval()
    with torch.no_grad():
        rmse_mlp = float(torch.sqrt(nn.functional.mse_loss(mlp(Xte), Yte)))

    # 低数据（50 轨迹 ≈ 1/8 训练集）——PINN 的经典优势区
    n_low = int(0.125 * len(tr))
    Xtr_low, Ytr_low = Xtr[:n_low], Ytr[:n_low]
    pinn_low = PINN()
    train_model_low = lambda m, lam: _train_sub(m, Xtr_low, Ytr_low, lam)
    train_model_low(pinn_low, 1.0)
    pinn_low.eval()
    with torch.no_grad():
        rmse_pinn_low = float(torch.sqrt(nn.functional.mse_loss(pinn_low(Xte), Yte)))
    mlp_low = PINN()
    train_model_low(mlp_low, 0.0)
    mlp_low.eval()
    with torch.no_grad():
        rmse_mlp_low = float(torch.sqrt(nn.functional.mse_loss(mlp_low(Xte), Yte)))

    # 归因对照（测试集上 t=t* 的样本）
    t_star_val = ts[int(np.round(ATTRIB_T / DT))]
    te_tstar = X_all[te][X_all[te][:, 0] == t_star_val]
    att_pinn = attribution(pinn, (te_tstar - xs_mean) / xs_std, 1, xs_mean, xs_std)
    # 真值：FD 灵敏度
    fds = []
    for row in te_tstar:
        p = dict(DEFAULT_PARAMS, H=float(row[1]))
        eps = p["H"] * 1e-3
        hi = simulate(dict(p, H=p["H"] + eps), T=T_HORIZON, dt=DT)
        lo = simulate(dict(p, H=p["H"] - eps), T=T_HORIZON, dt=DT)
        i = int(np.round(ATTRIB_T / DT))
        fds.append((hi[1][i, 0] - lo[1][i, 0]) / (2 * eps))
    fds = np.array(fds)
    pearson = float(np.corrcoef(att_pinn, fds)[0, 1]) if len(fds) > 2 else float("nan")

    result = {
        "rmse_pinn_norm": rmse_pinn, "rmse_mlp_norm": rmse_mlp,
        "rmse_pinn_lowdata_norm": rmse_pinn_low, "rmse_mlp_lowdata_norm": rmse_mlp_low,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "attrib_pearson_H": pearson, "n_attrib_points": int(len(fds)),
        "loss_first_last": [hist[0], hist[-1]],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    os.makedirs(os.path.dirname(os.path.abspath(__file__)) + "/results", exist_ok=True)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "spike_pinn.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"判据: RMSE_PINN≤MLP: {rmse_pinn <= rmse_mlp} | 归因 Pearson≥0.8: {pearson >= 0.8}")
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    main()
