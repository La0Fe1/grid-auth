"""离线生成 N-1 可行性标签（N1Gate 训练数据，exp_003 安全柱）。

标签定义（2026-09-13 修订，见 decisions.md）：
"toggle ℓ 后电网完美 N-1 安全"在本算例几乎不可满足（初始 N-1 安全度仅 10/20），
导致标签全零、门退化——已弃用。
现行定义 = 可操作的安全准则（运行人员实际执行的硬准则）：
对状态 s、连通线路 ℓ，post = s.simulate(toggle ℓ)（同一时间步，不推进时序），
  y_ℓ = 1 若 post 无过载（max rho < 0.95 留裕度）且 N-1 安全度不恶化
              （n1_score(post) ≥ n1_score(s)）
        0 若 post 不可行（simulate 失败/无解）或违反上述任一条件
产出 {"meta": 图元信息, "samples": [(node_feat, edge_feat, y (E,))]}，
并打印标签正样本率（数据驱动检查标签平衡）。

用法（在项目根目录下以模块方式运行）：
  python -m src.training.gen_n1_labels --out data/processed/n1_gate_labels.pkl
  冒烟测试：--smoke（1 场景 × 2 步 × 每步 4 条线）
"""
import argparse
import pickle

import numpy as np
from grid2op import make
from lightsim2grid import LightSimBackend

from src.utils.env import DEFAULT_DATASET_PATH, _resolve_dataset
from src.utils.graph import build_graph_meta, obs_to_graph

RHO_MARGIN = 0.95


def _simulate_or_none(obs, act):
    """obs.simulate 的防御封装：失败/不可行 → None。"""
    try:
        res = obs.simulate(act)
    except Exception:
        return None
    so = res[0] if isinstance(res, tuple) else res
    return so


def _toggle_act(env, lid):
    act = env.action_space({})
    act.change_line_status = [int(lid)]
    return act


def _outage_or_none(obs, env, out_k, toggle_l=None):
    """从真实观测单层仿真组合动作（可选 toggle ℓ + 断开 k）→ 后继观测。

    注意（2026-09-13 修订）：禁止嵌套 simulate——对 simulate 得到的观测再调
    simulate 在 grid2op 1.12.5 中不可靠（实测全部返回失败），一律用组合动作
    从真实观测出发做单层仿真。
    """
    act = env.action_space({})
    if toggle_l is not None:
        act.change_line_status = [int(toggle_l)]
    vec = np.zeros(int(obs.n_line), dtype=int)
    vec[int(out_k)] = -1
    act.set_line_status = vec
    return _simulate_or_none(obs, act)


def n1_score_of(obs, env, toggle_l=None):
    """toggle_l 之后的 N-1 安全度：多少条连通线路（不含 ℓ）的故障后无过载。

    单层组合仿真实现（见 _outage_or_none 注释）。
    """
    if obs is None or not hasattr(obs, "line_status"):
        return 0
    safe = 0
    for k in np.nonzero(obs.line_status == 1)[0]:
        if toggle_l is not None and int(k) == int(toggle_l):
            continue
        so = _outage_or_none(obs, env, k, toggle_l)
        if so is not None and hasattr(so, "rho") and so.rho.max() < 1.0:
            safe += 1
    return safe


def safe_toggle(obs, env, l):
    """toggle ℓ 是否满足可操作安全准则：post 无过载（留裕度）且 N-1 安全度不恶化。"""
    post = _simulate_or_none(obs, _toggle_act(env, l))
    if post is None or not hasattr(post, "rho"):
        return False
    if post.rho.max() >= RHO_MARGIN:
        return False
    base_n1 = n1_score_of(obs, env)
    post_n1 = n1_score_of(obs, env, toggle_l=l)
    return post_n1 >= base_n1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="rte_case14_realistic")
    ap.add_argument("--n-scenarios", type=int, default=20)
    ap.add_argument("--step-stride", type=int, default=100)
    ap.add_argument("--out", default="data/processed/n1_gate_labels.pkl")
    ap.add_argument("--smoke", action="store_true", help="冒烟测试：极小规模跑通全流程")
    args = ap.parse_args()

    env = make(dataset=_resolve_dataset(args.case, DEFAULT_DATASET_PATH),
               backend=LightSimBackend())
    meta = build_graph_meta(env)
    n_line = meta["n_line"]

    if args.smoke:
        scenarios, steps, max_lines = [0], [0, 50], 4
    else:
        scenarios = list(range(args.n_scenarios))
        steps = list(range(0, 1000, args.step_stride))
        max_lines = n_line

    samples = []
    t_start = __import__("time").time()
    for sc in scenarios:
        for step in steps:
            env.set_id(sc)
            env.reset()
            env.fast_forward_chronics(step)
            obs = env.get_obs()
            nf, ef = obs_to_graph(obs, meta)
            base_n1 = n1_score_of(obs, env)
            y = np.zeros(n_line, dtype=np.float32)
            connected = np.nonzero(obs.line_status == 1)[0][:max_lines]
            for l in connected:
                y[int(l)] = 1.0 if safe_toggle(obs, env, l) else 0.0
            samples.append((nf, ef, y))
            print(f"sc{sc} step{step}: base_n1={base_n1}, 安全toggle线数={int(y.sum())}/{len(connected)}")

    with open(args.out, "wb") as f:
        pickle.dump({"meta": meta, "samples": samples}, f)
    all_y = np.concatenate([s[2] for s in samples])
    print(f"保存 {len(samples)} 个样本到 {args.out}，耗时 {__import__('time').time() - t_start:.0f}s；"
          f"正样本率={float(all_y.mean()):.3f}")


if __name__ == "__main__":
    main()
