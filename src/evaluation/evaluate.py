"""评估模块：逐幕运行智能体、汇总指标、成对显著性检验（exp_002/003/005/006/007 共用）。

指标定义（全部来自环境返回值，禁止估算；AGENTS.md 反幻觉铁律 2）：
- survival: 存活步数（episode 长度）
- reward: 累计奖励
- overflow_steps: 存在过载线路（rho>1）的步数
- n_illegal: 被 grid2op 拒绝的非法动作数
- n_blocks: 屏蔽器拦截的 toggle 线路数（无屏蔽时恒 0）
- n_toggles: 实际请求 toggle 的线路数

智能体接口：agent = agent_factory(env) → 有 act(obs)->np.ndarray 方法的对象。
每幕使用全新环境实例（env_factory），并用 env.seed(seed) 固定幕内随机性。
"""
import dataclasses

import numpy as np

from src.utils.metrics import mean_std, paired_wilcoxon


@dataclasses.dataclass
class EpisodeStats:
    survival: int
    reward: float
    overflow_steps: int
    n_illegal: int
    n_blocks: int
    n_toggles: int
    n_proposed: int = 0  # 屏蔽环境下策略提案的 toggle 总数（削减率分母）


def run_episode(agent, env, max_steps=None):
    """跑一幕（场景须由调用方在 reset 前通过 set_id 指定）。"""
    obs, _ = env.reset()
    total_r, overflows = 0.0, 0
    n_illegal, n_blocks, n_toggles, n_proposed = 0, 0, 0, 0
    steps, done = 0, False
    while not done and (max_steps is None or steps < max_steps):
        action = np.asarray(agent.act(obs), dtype=np.float32)
        obs, r, done, trunc, info = env.step(action)
        steps += 1
        total_r += float(r)
        rho = info.get("rho")
        if rho is None and isinstance(obs, dict) and "edge_feat" in obs:
            rho = np.asarray(obs["edge_feat"])[:, 0]  # 图观测回退
        if rho is not None and (np.asarray(rho) > 1.0).any():
            overflows += 1
        n_illegal += int(bool(info.get("is_illegal")))
        sh = info.get("shield")
        if sh:
            n_blocks += len(sh.get("blocked", []))
            n_proposed += int(sh.get("proposed", 0))
        n_toggles += int(action.sum())
        if trunc:
            break
    return EpisodeStats(steps, total_r, overflows, n_illegal, n_blocks, n_toggles, n_proposed)


def _mean_stats(stats):
    """多幕 EpisodeStats 的逐字段均值（保持 EpisodeStats 结构）。"""
    vals = {}
    for f in ("survival", "reward", "overflow_steps", "n_illegal", "n_blocks",
              "n_toggles", "n_proposed"):
        vals[f] = float(np.mean([getattr(s, f) for s in stats]))
    return EpisodeStats(**vals)


def aggregate_per_seed(scen, seeds, scenarios_per_seed):
    """场景级明细 → 种子级均值列表（与 seeds 对齐）。"""
    stats = []
    for s in seeds:
        seed_stats = [scen[s * scenarios_per_seed + k] for k in range(scenarios_per_seed)]
        stats.append(_mean_stats(seed_stats))
    return stats


def run_agent(agent_factory, env_factory, seeds, scenarios_per_seed=4, max_steps=None,
              id_offset=0):
    """评估口径（AGENTS.md 2.2）：种子 s → 场景 id = id_offset + s*scenarios_per_seed + k。

    每个种子的指标 = 其 scenarios_per_seed 个场景结果的均值；
    返回与 seeds 一一对应的 EpisodeStats 列表（供种子级均值±标准差）。
    单遍评估（内部复用 run_agent_scenarios，无重复开销）。
    """
    scen = run_agent_scenarios(agent_factory, env_factory, seeds,
                               scenarios_per_seed=scenarios_per_seed,
                               max_steps=max_steps, id_offset=id_offset)
    return aggregate_per_seed(scen, seeds, scenarios_per_seed)


def run_agent_scenarios(agent_factory, env_factory, seeds, scenarios_per_seed=4,
                        max_steps=None, id_offset=0):
    """场景级明细评估：返回 {scenario_id: EpisodeStats}。

    用途：5 种子配对 Wilcoxon 最小 p 为 0.0625（永远达不到 p<0.05），
    因此显著性检验在场景级成对样本上做（5 种子 × 4 场景 = 20 对观测），
    种子级均值±标准差仍按铁律报告（2026-09-13 决策，见 decisions.md）。

    id_offset：场景 id 偏移（0=标准测试集；100=验证集，模型选择用）。
    """
    flat = {}
    for s in seeds:
        for k in range(scenarios_per_seed):
            env = env_factory()
            env.set_id(id_offset + s * scenarios_per_seed + k)
            flat[id_offset + s * scenarios_per_seed + k] = run_episode(
                agent_factory(env), env, max_steps=max_steps)
    return flat


def compare_paired_scenarios(scen_a, scen_b, field="survival", alternative="two-sided"):
    """场景级成对检验：两个 {scenario_id: EpisodeStats} 按场景对齐后配对 Wilcoxon。"""
    common = sorted(set(scen_a) & set(scen_b))
    a = np.array([getattr(scen_a[c], field) for c in common], dtype=float)
    b = np.array([getattr(scen_b[c], field) for c in common], dtype=float)
    return paired_wilcoxon(a, b, alternative=alternative)


def summarize(stats):
    """EpisodeStats 列表 → 各字段 mean/std 字典。"""
    d = {}
    for f in ("survival", "reward", "overflow_steps", "n_illegal", "n_blocks",
              "n_toggles", "n_proposed"):
        vals = np.array([getattr(s, f) for s in stats], dtype=float)
        d[f] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else float("nan")}
    return d


def compare_paired(stats_a, stats_b, field="survival", alternative="two-sided"):
    """同种子成对比较两智能体某指标，返回 (statistic, pvalue)。"""
    a = np.array([getattr(s, field) for s in stats_a], dtype=float)
    b = np.array([getattr(s, field) for s in stats_b], dtype=float)
    return paired_wilcoxon(a, b, alternative=alternative)


def format_comparison(stats_a, stats_b, field="survival", name_a="A", name_b="B"):
    a = np.array([getattr(s, field) for s in stats_a], dtype=float)
    b = np.array([getattr(s, field) for s in stats_b], dtype=float)
    stat, p = paired_wilcoxon(a, b)
    return f"{name_a} {mean_std(a)} vs {name_b} {mean_std(b)}: p={p:.4f}"
