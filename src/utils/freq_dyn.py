"""频率调节动力学（方向 E 可行性模块）：经典单机调频模型。

状态 x = [Δf, ΔPm, ΔPv]（频率偏差、机械功率偏差、汽门开度偏差）：
  2H dΔf/dt  = ΔPm − ΔPe − D·Δf     （转子运动方程）
  Tg dΔPv/dt = ΔPc − ΔPv − Δf/R     （调速器）
  Tt dΔPm/dt = ΔPv − ΔPm            （汽轮机）
ΔPe = 阶跃扰动（负荷阶跃），ΔPc = 0（一次调频，无二次调度）。

解析结果：
- 稳态：Δf_ss = −ΔPe / (D + 1/R)
- 初始 ROCOF：−ΔPe / (2H)

用途：方向 E（PINN 调频代理 + XAI 归因）的动力学真值与归因基准来源。
与用户既有暂态稳定项目（功角轨迹/CCT）不同：本模块是频率调节动力学。
"""
import numpy as np

DEFAULT_PARAMS = dict(H=5.0, D=1.0, R=0.05, Tg=0.2, Tt=0.5, P_step=0.1, Pc=0.0)


def f(x, t, p):
    """ODE 右端。x=[df, dPm, dPv]；返回顺序与状态一致 [ddf, ddPm, ddPv]。"""
    df, dpm, dpv = x
    H, D, R, Tg, Tt = p["H"], p["D"], p["R"], p["Tg"], p["Tt"]
    ddf = (dpm - p["P_step"] - D * df) / (2.0 * H)
    ddpv = (p["Pc"] - dpv - df / R) / Tg
    ddpm = (dpv - dpm) / Tt
    return np.array([ddf, ddpm, ddpv])


def rk4_step(x, t, dt, p):
    k1 = f(x, t, p)
    k2 = f(x + dt / 2 * k1, t + dt / 2, p)
    k3 = f(x + dt / 2 * k2, t + dt / 2, p)
    k4 = f(x + dt * k3, t + dt, p)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(p=None, x0=(0.0, 0.0, 0.0), T=30.0, dt=0.01):
    """返回 (t, X)；X 列 = [Δf, ΔPm, ΔPv]。"""
    p = dict(DEFAULT_PARAMS, **(p or {}))
    n = int(T / dt)
    t = np.arange(n) * dt
    X = np.zeros((n, 3))
    X[0] = x0
    for i in range(1, n):
        X[i] = rk4_step(X[i - 1], t[i - 1], dt, p)
    return t, X


def nadir(t, X):
    """频率最低点（nadir）与到达时间。"""
    df = X[:, 0]
    i = int(np.argmin(df))
    return float(df[i]), float(t[i])


def initial_rocof(p=None):
    """解析初始 ROCOF = −ΔPe/(2H)。"""
    p = dict(DEFAULT_PARAMS, **(p or {}))
    return -p["P_step"] / (2.0 * p["H"])


def steady_state(p=None):
    """解析稳态频率偏差。"""
    p = dict(DEFAULT_PARAMS, **(p or {}))
    return -p["P_step"] / (p["D"] + 1.0 / p["R"])


def nadir_sensitivity_fd(p=None, param="H", eps_ratio=1e-3):
    """nadir 对参数的数值灵敏度（中心差分，归因对照的'真值'）。"""
    p = dict(DEFAULT_PARAMS, **(p or {}))
    base = p[param]
    eps = base * eps_ratio
    p_hi, p_lo = dict(p, **{param: base + eps}), dict(p, **{param: base - eps})
    n_hi, _ = nadir(*simulate(p_hi, T=15.0, dt=0.005))
    n_lo, _ = nadir(*simulate(p_lo, T=15.0, dt=0.005))
    return (n_hi - n_lo) / (2 * eps)
