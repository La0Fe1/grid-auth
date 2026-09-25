"""解释柱（H2）评估：GAT 注意力归因与 N-1 越限支路的一致性。

结构：
- 纯函数 topk_jaccard / rank_metric（单测覆盖，不依赖 grid2op）；
- violated_lines(env, obs_glop)：当前状态 N-1 越限支路集合（故障后引起过载的连通线路）；
- scan_states(policy, env, ...)：采样状态，统计注意力与越限集的一致性，
  并与随机排序基线比较（配对 Wilcoxon，见 exp_004 用法）。
"""
import numpy as np

from src.utils.metrics import paired_wilcoxon


def violated_lines(env, obs_glop):
    """N-1 越限支路：断开后使剩余线路过载的连通线路 id 列表。"""
    viol = []
    for lid in np.nonzero(obs_glop.line_status == 1)[0]:
        try:
            res = obs_glop.simulate(env.env_glop.action_space({"set_line_status": [(int(lid), -1)]}))
        except Exception:
            viol.append(int(lid))
            continue
        so = res[0] if isinstance(res, tuple) else res
        if so is None or not hasattr(so, "rho") or so.rho.max() >= 1.0:
            viol.append(int(lid))
    return np.array(viol, dtype=int)


def topk_jaccard(attn, violated, k):
    """注意力 Top-k 线路集合与越限线路集合的 Jaccard 相似度。"""
    top = set(np.argsort(-np.asarray(attn, dtype=float))[:k].tolist())
    viol = set(np.asarray(violated, dtype=int).tolist())
    union = top | viol
    if not union:
        return 0.0
    return len(top & viol) / len(union)


def rank_metric(attn, violated):
    """越限线路在注意力排序中的平均归一化秩（0=注意力最高；随机期望 0.5，越低越好）。"""
    attn = np.asarray(attn, dtype=float)
    violated = np.asarray(violated, dtype=int)
    if len(violated) == 0 or len(attn) < 2:
        return float("nan")
    order = np.argsort(np.argsort(-attn))  # 秩，0=注意力最高
    return float(order[violated].mean() / (len(attn) - 1))


def combined_attention(attns):
    """各层/各头注意力合并为单一每线得分：层内多头平均后，层间平均。"""
    layer_scores = [np.asarray(a).mean(axis=-1) for a in attns]  # 每层 (B, E)
    return np.mean(np.stack(layer_scores, axis=0), axis=0)  # (B, E)


def scan_states(policy, env, n_states=50, top_k=5, rng_seed=42, step_stride=100):
    """采样 n_states 个状态，返回一致性统计 + 随机基线配对比较。

    Returns:
        dict: {"jaccard": [...], "rank": [...], "jaccard_random": [...],
               "rank_random": [...], "jaccard_p": p 值, "rank_p": p 值}
    """
    import torch

    from src.utils.graph import obs_to_graph

    rng = np.random.RandomState(rng_seed)
    jac, rk, jac_rnd, rk_rnd = [], [], [], []
    n_line = env.n_line

    for i in range(n_states):
        env.env_glop.set_id(i % 50)
        env.env_glop.reset()
        env.env_glop.fast_forward_chronics((i % 5) * step_stride)
        obs_glop = env.env_glop.get_obs()
        nf, ef = obs_to_graph(obs_glop, env.meta)
        obs = {"node_feat": torch.as_tensor(nf[None], dtype=torch.float32),
               "edge_feat": torch.as_tensor(ef[None], dtype=torch.float32)}

        _, _, attns = policy.encode_with_attention(obs)
        attn = combined_attention(attns)[0]  # (E,)
        viol = violated_lines(env, obs_glop)

        jac.append(topk_jaccard(attn, viol, top_k))
        rk.append(rank_metric(attn, viol))
        jac_rnd.append(topk_jaccard(rng.permutation(attn), viol, top_k))
        rk_rnd.append(rank_metric(rng.permutation(attn), viol))

    _, jac_p = paired_wilcoxon(jac, jac_rnd)
    _, rk_p = paired_wilcoxon(rk, rk_rnd)
    return {
        "jaccard": jac, "rank": rk,
        "jaccard_random": jac_rnd, "rank_random": rk_rnd,
        "jaccard_p": jac_p, "rank_p": rk_p,
    }
