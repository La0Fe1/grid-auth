"""统计与指标工具（评估模块共用）。

- paired_wilcoxon(x, y): 配对 Wilcoxon 符号秩检验（scipy），论文表格 p 值来源。
- roc_auc(y_true, y_score): 精确秩和 AUC（不依赖 sklearn）。
- mean_std(x): "均值 ± 标准差" 格式化（论文表格用）。
"""
import numpy as np
from scipy import stats


def paired_wilcoxon(x, y, alternative="two-sided"):
    """配对 Wilcoxon 符号秩检验（AGENTS.md 2.2 要求报告 p 值）。

    Args:
        x, y: 同一批种子/场景下成对观测的指标值。
    Returns:
        (statistic, pvalue)；全部样本持平时返回 (nan, 1.0)。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError(f"配对样本形状不一致: {x.shape} vs {y.shape}")
    mask = ~np.isclose(x, y)
    if mask.sum() == 0:
        return float("nan"), 1.0
    res = stats.wilcoxon(x[mask], y[mask], alternative=alternative)
    return float(res.statistic), float(res.pvalue)


def roc_auc(y_true, y_score):
    """二分类 AUC（Mann-Whitney 秩和公式，精确、无依赖）。

    AUC = P(正样本得分 > 负样本得分)，并列用平均秩处理。
    """
    y_true = np.asarray(y_true, dtype=bool)
    y_score = np.asarray(y_score, dtype=float)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    pos = y_score[y_true]
    neg = y_score[~y_true]
    n_pos, n_neg = len(pos), len(neg)
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[:n_pos].sum()
    return float((r_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def mean_std(x, decimals=1):
    """'均值±标准差' 字符串（论文表格格式）。"""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return f"{x.mean():.{decimals}f}±nan"
    return f"{x.mean():.{decimals}f}±{x.std(ddof=1):.{decimals}f}"
