"""exp_002 主表生成：读取各智能体评估 JSON，产出配对 Wilcoxon 对比表。

所有数值只来自各评估 JSON 的 per_seed/per_scenario 原始记录（反幻觉铁律 2）。
- 表格列 = 种子级 mean±std（5 种子，铁律要求）；
- p 值 = 场景级配对 Wilcoxon（5 种子 × 4 场景 = 20 对观测；
  5 对样本的配对 Wilcoxon 最小 p=0.0625，见 decisions.md 2026-09-13 决策）。

用法：
  python -m src.evaluation.make_table \
      --do-nothing experiments/exp_001/results/do_nothing.json \
      --agents 方法名=路径 方法名2=路径2 ... \
      --out experiments/exp_002/results/main_table.md
"""
import argparse
import json

import numpy as np

from src.utils.metrics import mean_std, paired_wilcoxon


def load_per_seed(path, field="survival"):
    with open(path) as f:
        d = json.load(f)
    per = d["per_seed"]
    return np.array([per[k][field] for k in sorted(per, key=int)], dtype=float)


def load_scenarios(path, field="survival"):
    with open(path) as f:
        d = json.load(f)
    return {int(k): v[field] for k, v in d["per_scenario"].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--do-nothing", required=True)
    ap.add_argument("--agents", nargs="+", required=True, help="'名称=json路径' 列表")
    ap.add_argument("--fields", nargs="+",
                    default=["survival", "reward", "overflow_steps", "n_toggles"])
    ap.add_argument("--out", default="experiments/exp_002/results/main_table.md")
    args = ap.parse_args()

    agents = []
    for item in args.agents:
        name, path = item.split("=", 1)
        agents.append((name, path))

    dn = {f: load_per_seed(args.do_nothing, f) for f in args.fields}
    dn_scen = {f: load_scenarios(args.do_nothing, f) for f in args.fields}
    lines = []
    header = "| 方法 | " + " | ".join(args.fields) + " | p(survival vs do-nothing, 场景级) |"
    lines.append(header)
    lines.append("|---|" + "|".join(["---"] * len(args.fields)) + "|---|")
    lines.append(f"| do-nothing | " + " | ".join(mean_std(dn[f]) for f in args.fields) + " | — |")
    for name, path in agents:
        vals = {f: load_per_seed(path, f) for f in args.fields}
        scen = {f: load_scenarios(path, f) for f in args.fields}
        common = sorted(set(scen["survival"]) & set(dn_scen["survival"]))
        a = np.array([scen["survival"][c] for c in common], dtype=float)
        b = np.array([dn_scen["survival"][c] for c in common], dtype=float)
        stat, p = paired_wilcoxon(a, b)
        lines.append(f"| {name} | " + " | ".join(mean_std(vals[f]) for f in args.fields)
                     + f" | {p:.4f} |")

    table = "\n".join(lines)
    print(table)
    with open(args.out, "w") as f:
        f.write(table + "\n")
    print(f"\n表已保存: {args.out}")


if __name__ == "__main__":
    main()
