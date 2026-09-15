"""src/utils/metrics.py 单元测试。"""
import numpy as np
import pytest

from src.utils.metrics import mean_std, paired_wilcoxon, roc_auc


def test_wilcoxon_significant():
    """明显更优的成对样本应给出 p<0.05（8 对同向，p=2×0.5^8≈0.0078）。"""
    x = np.array([1052, 1100, 1010, 1080, 1040, 1060, 1090, 1030])  # 方法 A
    y = np.array([1500, 1580, 1490, 1600, 1520, 1550, 1480, 1530])  # 方法 B
    stat, p = paired_wilcoxon(x, y)
    assert p < 0.05


def test_wilcoxon_no_difference():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])
    stat, p = paired_wilcoxon(x, y)
    assert p == 1.0
    assert np.isnan(stat)


def test_wilcoxon_shape_mismatch():
    with pytest.raises(ValueError):
        paired_wilcoxon([1, 2], [1])


def test_roc_auc_perfect():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    assert roc_auc(y, s) == 1.0


def test_roc_auc_reversed():
    y = np.array([0, 0, 0, 1, 1, 1])
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    assert roc_auc(y, s) == 0.0


def test_roc_auc_single_class():
    y = np.array([1, 1, 1])
    s = np.array([0.5, 0.6, 0.7])
    assert np.isnan(roc_auc(y, s))


def test_roc_auc_ties_handled():
    y = np.array([0, 1, 0, 1])
    s = np.array([0.5, 0.5, 0.5, 0.5])
    assert roc_auc(y, s) == 0.5


def test_mean_std_format():
    assert mean_std([1.0, 1.0, 1.0]) == "1.0±0.0"
