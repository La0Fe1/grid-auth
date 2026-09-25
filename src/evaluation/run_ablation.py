"""exp_013 消融运行器：6 维 × 3 变体（基线 + 每维 2 变体 = 13 配置）。

每个配置以子进程调用 auth_eval（缩减规模：20 状态 × 5 种子 × 5 候选 ×
λ{2,3,4} × M=1000），汇总为消融表（区间验证的误放率/误拒率 + 标称对照）。

用法：python -m src.evaluation.run_ablation
"""
import json
import os
import subprocess
import sys

PY = sys.executable
COMMON = ["-u", "-m", "src.evaluation.auth_eval",
          "--n-states", "20", "--lamdas", "2", "3", "4",
          "--gt-m", "1000", "--conv-checkpoints", "100", "300", "1000"]

DIMS = {
    "eps": [{"label": "eps=0.05", "extra": ["--load-eps", "0.05"]},
            {"label": "eps=0.10", "extra": ["--load-eps", "0.1"]}],
    "verification": [{"label": "+connectivity", "extra": ["--require-connected"]},
                     {"label": "+connectivity+theta", "extra": ["--require-connected",
                                                                "--theta-cap", "0.3"]}],
    "lines_model": [{"label": "uniform-x", "extra": ["--lines-file",
                                                     "data/processed/lines_uniform.pkl"]},
                    {"label": "telemetry-30", "extra": ["--lines-file",
                                                        "data/processed/lines_tele30.pkl"]}],
    "proposal": [{"label": "random-proposals", "extra": ["--proposal-source", "random"]},
                 {"label": "llm-proposals", "extra": ["--proposals",
                                                      "data/processed/llm_proposals.json"]}],
    "margin": [{"label": "margin=0", "extra": ["--margin", "0"]},
               {"label": "margin=0.02", "extra": ["--margin", "0.02"]}],
    "criterion": [{"label": "prob99", "extra": ["--prob-threshold", "0.99"]},
                  {"label": "prob95", "extra": ["--prob-threshold", "0.95"]}],
}


def prepare_lines_files():
    """生成消融用的线路模型文件（uniform 与 30 状态遥测识别）。"""
    import pickle
    import numpy as np

    from src.utils.env import Grid2OpGraphEnv
    from experiments.exp_011_interval_auth.spike_interval_auth import (
        estimate_lines_from_telemetry,
    )
    env = Grid2OpGraphEnv(case="rte_case14_realistic", test=False)
    obs = env.env_glop.reset()
    n_line = env.n_line
    uniform = [{"or": int(obs.line_or_to_subid[l]),
                "ex": int(obs.line_ex_to_subid[l]), "x": 0.1}
               for l in range(n_line)]
    with open("data/processed/lines_uniform.pkl", "wb") as f:
        pickle.dump(uniform, f)
    tele30, _ = estimate_lines_from_telemetry(env, n_states=30)
    with open("data/processed/lines_tele30.pkl", "wb") as f:
        pickle.dump(tele30, f)


def main():
    os.makedirs("experiments/exp_013/results", exist_ok=True)
    os.makedirs("experiments/exp_013/logs", exist_ok=True)
    prepare_lines_files()

    def run_one(name, extra):
        out = f"experiments/exp_013/results/abl_{name}.json"
        log = f"experiments/exp_013/logs/abl_{name}.log"
        cmd = [PY] + COMMON + extra + ["--out", out]
        with open(log, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, check=False)
        if os.path.exists(out):
            with open(out) as f:
                d = json.load(f)
            s = d["summary"]
            return (s["interval"]["false_pass_rate"],
                    s["interval"]["false_reject_rate"],
                    s["nominal"]["false_pass_rate"],
                    d["elapsed_seconds"])
        return None

    rows = []
    base = run_one("baseline", [])
    rows.append(("baseline(eps=0.2,margin=0.05,flow,top)", base))
    for dim, variants in DIMS.items():
        for v in variants:
            rows.append((v["label"], run_one(v["label"], v["extra"])))

    table = [("配置", "区间误放率", "区间误拒率", "标称误放率", "耗时s")]
    for label, r in rows:
        table.append((label,) + tuple(f"{x:.4f}" if isinstance(x, float) else str(x)
                                      for x in r))
    with open("experiments/exp_013/results/ablation_table.md", "w") as f:
        for row in table:
            f.write("| " + " | ".join(str(c) for c in row) + " |\n")
    for row in table:
        print(row)


if __name__ == "__main__":
    main()
