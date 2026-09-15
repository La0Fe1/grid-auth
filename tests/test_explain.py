"""src/evaluation/explain.py 纯函数单元测试。"""
import numpy as np
import pytest

from src.evaluation.explain import combined_attention, rank_metric, topk_jaccard


def test_topk_jaccard_exact():
    attn = np.array([0.9, 0.1, 0.8, 0.2])
    violated = np.array([0, 2])
    # top-2 = {0, 2} == violated → Jaccard 1.0
    assert topk_jaccard(attn, violated, k=2) == 1.0
    # top-2 vs violated {1} → 交集空 → 0
    assert topk_jaccard(attn, np.array([1]), k=2) == 0.0


def test_topk_jaccard_partial():
    attn = np.array([0.9, 0.1, 0.8, 0.2])
    violated = np.array([0, 1])
    # top-2 = {0, 2}, 交 {0}，并 {0,1,2} → 1/3
    assert abs(topk_jaccard(attn, violated, k=2) - 1 / 3) < 1e-9


def test_topk_jaccard_empty():
    assert topk_jaccard(np.array([0.5, 0.5]), np.array([], dtype=int), k=1) == 0.0


def test_rank_metric_low_for_high_attn():
    attn = np.array([0.9, 0.1, 0.5])
    violated = np.array([0])  # 注意力最高 → 归一化秩 0
    assert rank_metric(attn, violated) == 0.0
    violated_low = np.array([1])  # 注意力最低 → 归一化秩 1
    assert rank_metric(attn, violated_low) == 1.0


def test_rank_metric_empty_violated_nan():
    assert np.isnan(rank_metric(np.array([0.5, 0.5]), np.array([], dtype=int)))


def test_rank_metric_single_line_nan():
    assert np.isnan(rank_metric(np.array([0.5]), np.array([0])))


def test_combined_attention_shape():
    attns = [np.random.rand(2, 20, 2), np.random.rand(2, 20, 1)]
    out = combined_attention(attns)
    assert out.shape == (2, 20)
