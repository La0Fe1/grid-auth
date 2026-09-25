"""exp_014 鲁棒性运行器：失配水平扫描与分布变体。

配置（缩减规模 20 状态 × M=1000）：
- ε ∈ {0, 0.05, 0.1, 0.2, 0.4}（H2 单调性 + ε=0 极限一致性）；
- gaussian(ε=0.2)：高斯量测失配（箱外无界 → 区间法残余误放率，展示保证边界）；
- missing20%(ε=0.2)：20% 负荷不可观测（箱加宽）。

输出：experiments/exp_014/results/robustness_table.md + per-config JSON。

用法：python -m src.evaluation.run_robustness
"""
import json
import os
import subprocess
import sys

PY = sys.executable
COMMON = ["-u", "-m", "src.evaluation.auth_eval",
          "--n-states", "20", "--lamdas", "2", "3", "4",
          "--gt-m", "1000", "--conv-checkpoints", "100", "300", "1000",
          # 真值参考固定（跨配置比较口径一致）
          "--gt-eps", "0.2", "--gt-margin", "0.05"]

CONFIGS = [
    ("eps=0.00", ["--load-eps", "0.0"]),
    ("eps=0.05", ["--load-eps", "0.05"]),
    ("eps=0.10", ["--load-eps", "0.1"]),
    ("eps=0.20", ["--load-eps", "0.2"]),
    ("eps=0.40", ["--load-eps", "0.4"]),
    ("gaussian_eps0.20", ["--load-eps", "0.2", "--gt-dist", "gaussian"]),
    ("missing20_eps0.20", ["--load-eps", "0.2", "--missing-frac", "0.2"]),
]


def main():
    os.makedirs("experiments/exp_014/results", exist_ok=True)
    os.makedirs("experiments/exp_014/logs", exist_ok=True)
    table = [("配置", "区间误放率", "区间误拒率", "标称误放率", "MC误放率")]
    for name, extra in CONFIGS:
        out = f"experiments/exp_014/results/rob_{name}.json"
        log = f"experiments/exp_014/logs/rob_{name}.log"
        cmd = [PY] + COMMON + extra + ["--out", out]
        with open(log, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
        with open(out) as f:
            d = json.load(f)
        s = d["summary"]
        table.append((name,
                      f"{s['interval']['false_pass_rate']:.4f}",
                      f"{s['interval']['false_reject_rate']:.4f}",
                      f"{s['nominal']['false_pass_rate']:.4f}",
                      f"{s['mc']['false_pass_rate']:.4f}"))
        print(table[-1])
    with open("experiments/exp_014/results/robustness_table.md", "w") as f:
        for row in table:
            f.write("| " + " | ".join(row) + " |\n")
    print("表已保存: experiments/exp_014/results/robustness_table.md")


if __name__ == "__main__":
    main()
