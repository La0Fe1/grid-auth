"""src/utils/freq_dyn.py 单元测试（解析结果对照）。"""
import numpy as np
import pytest

from src.utils.freq_dyn import (
    initial_rocof, nadir, nadir_sensitivity_fd, simulate, steady_state,
)


def test_steady_state_convergence():
    t, X = simulate(T=60.0, dt=0.01)
    assert np.isclose(X[-1, 0], steady_state(), rtol=1e-3)


def test_initial_rocof_matches_analytical():
    t, X = simulate(dt=0.002)
    num_rocof = (X[1, 0] - X[0, 0]) / (t[1] - t[0])
    assert np.isclose(num_rocof, initial_rocof(), rtol=0.05)


def test_nadir_negative_and_bounded():
    n, t_n = nadir(*simulate())
    assert n < 0
    assert n < steady_state()  # nadir 是最低点，比稳态更负
    assert t_n > 0


def test_nadir_improves_with_inertia():
    """惯性越大，nadir 越浅（归因方向的物理先验）。"""
    n_low, _ = nadir(*simulate({"H": 3.0}))
    n_high, _ = nadir(*simulate({"H": 8.0}))
    assert n_high > n_low


def test_sensitivity_sign():
    """∂nadir/∂H > 0（H 增大 nadir 变浅）；∂nadir/∂R < 0（R 增大=调速响应更弱=nadir 更深）。"""
    assert nadir_sensitivity_fd(param="H") > 0
    assert nadir_sensitivity_fd(param="R") < 0
