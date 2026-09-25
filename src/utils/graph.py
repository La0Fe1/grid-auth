"""图构建纯函数：grid2op 观测 → 图（节点特征 + 边特征）。

图设计（由遗留 gnn_env.py 迁移为唯一权威实现）：
- 节点 = 变电站（substation）
- 边 = 线路（line），有向（or -> ex），E = n_line
- 节点特征 = [该站总发电（全网归一化）, 该站总负荷（全网归一化）, 该站平均电压]
- 边特征 = [负载率 rho, 通断 line_status]

归一化设计（跨算例泛化的关键，AGENTS.md 主假设 H3 的机制基础）：
发电/负荷按全网总量归一，绝对值随电网规模变化不影响特征尺度，
14-bus 训练的特征分布与 36/118-bus 可比，避免负迁移。
"""
import numpy as np

NODE_DIM = 3
EDGE_DIM = 2


def build_graph_meta(env):
    """从 grid2op 环境读一次，得到固定不动的图结构。"""
    obs = env.reset()
    return dict(
        n_sub=int(obs.n_sub),
        n_line=int(obs.n_line),
        edge_index=np.stack([obs.line_or_to_subid, obs.line_ex_to_subid]).astype(np.int64),
        gen_to_sub=obs.gen_to_subid.astype(np.int64),
        load_to_sub=obs.load_to_subid.astype(np.int64),
    )


def obs_to_graph(obs, meta):
    """把一次观测转成图（节点特征 + 边特征）。

    Returns:
        node_feat: (n_sub, 3) float32
        edge_feat: (n_line, 2) float32
    """
    n_sub = meta["n_sub"]
    node_feat = np.zeros((n_sub, NODE_DIM), dtype=np.float32)

    # 发电聚合到变电站
    for g, s in enumerate(meta["gen_to_sub"]):
        node_feat[s, 0] += obs.gen_p[g]
    # 负荷聚合到变电站
    for l, s in enumerate(meta["load_to_sub"]):
        node_feat[s, 1] += obs.load_p[l]

    # 归一化成相对总量（0-1），跨规模可比
    total_gen = node_feat[:, 0].sum()
    total_load = node_feat[:, 1].sum()
    if total_gen > 0:
        node_feat[:, 0] /= total_gen
    if total_load > 0:
        node_feat[:, 1] /= total_load

    # 电压：线路两端电压平均聚合到变电站
    v_count = np.zeros(n_sub, dtype=np.float32)
    for i in range(meta["n_line"]):
        o = meta["edge_index"][0, i]
        e = meta["edge_index"][1, i]
        node_feat[o, 2] += obs.v_or[i]
        node_feat[e, 2] += obs.v_ex[i]
        v_count[o] += 1
        v_count[e] += 1
    v_count[v_count == 0] = 1
    node_feat[:, 2] /= v_count

    # 边特征：rho + line_status
    edge_feat = np.stack([obs.rho, obs.line_status], axis=1).astype(np.float32)
    return node_feat, edge_feat


RICH_NODE_DIM = 5
RICH_EDGE_DIM = 5


def obs_to_graph_v2(obs, meta):
    """丰富特征版（改进轴，exp_006 消融维度 2 变体）：
    节点 = [发电份额, 负荷份额, 平均电压, 平均|v-1|, 发电-负荷不平衡份额]
    边 = [rho, line_status, 1-rho(裕度), rho 与全网均值之差, 冷却剩余时间(归一化)]

    冷却特征（2026-09-13 修订）：对手攻击断线后有冷却期，期间重连非法
    （训练中大量非法动作的来源），冷却结束后重连合法——存活任务的核心机制。
    """

    node_feat, edge_feat = obs_to_graph(obs, meta)
    n_sub = meta["n_sub"]

    # 节点扩展
    v_dev = np.zeros(n_sub, dtype=np.float32)
    for i in range(meta["n_line"]):
        o = meta["edge_index"][0, i]
        e = meta["edge_index"][1, i]
        v_dev[o] += abs(obs.v_or[i] - 1.0)
        v_dev[e] += abs(obs.v_ex[i] - 1.0)
    v_count = np.zeros(n_sub, dtype=np.float32)
    for i in range(meta["n_line"]):
        o = meta["edge_index"][0, i]
        e = meta["edge_index"][1, i]
        v_count[o] += 1
        v_count[e] += 1
    v_count[v_count == 0] = 1
    v_dev /= v_count
    imb = node_feat[:, 0] - node_feat[:, 1]  # 发电份额 - 负荷份额

    node_rich = np.concatenate(
        [node_feat, v_dev[:, None], imb[:, None]], axis=1).astype(np.float32)

    # 边扩展
    margin = 1.0 - obs.rho
    rho_dev = obs.rho - np.mean(obs.rho)
    cooldown = getattr(obs, "time_before_cooldown_line", None)
    if cooldown is None:  # 防御：无此属性（旧版本/异常观测）时补零
        cooldown = np.zeros(meta["n_line"], dtype=np.float32)
    else:
        cooldown = np.asarray(cooldown, dtype=np.float32)
    cd_max = float(cooldown.max())
    cd_norm = cooldown / cd_max if cd_max > 0 else cooldown
    edge_rich = np.stack(
        [obs.rho, obs.line_status, margin, rho_dev, cd_norm], axis=1).astype(np.float32)
    return node_rich, edge_rich
