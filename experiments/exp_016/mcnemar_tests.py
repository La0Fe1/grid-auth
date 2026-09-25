"""exp_016 补测：验证器间的配对 McNemar 检验（误放/误拒决策不一致性）。

数据：full_rerun 产出的全量用例记录（每用例含四验证器的放行判定与
真值危险标志）。McNemar 检验比较同一批用例上两个验证器的
"错误决策不一致对"是否对称：
  b = A 误放且 B 未误放 的用例数；c = B 误放且 A 未误放 的用例数；
  χ² = (b−c)²/(b+c)，p 值取自由度为 1 的卡方分布（b+c 为 0 时 p=1）。
误放 = 放行 且 真值存在危险实现（gt_danger_rate > 0）；
误拒 = 拒绝 且 真值无危险实现。
区间验证器误放恒 0（构造性保证），故与标称/MC 的对比 p 值将极小——如实报告。

用法：python -m experiments.exp_016.mcnemar_tests
输出：experiments/exp_016/results/mcnemar_tests.json
"""
import json
import os

import numpy as np
from scipy.stats import chi2


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def mcnemar(n_a_only, n_b_only):
    b, c = float(n_a_only), float(n_b_only)
    if b + c == 0:
        return 1.0, 0.0
    stat = (b - c) ** 2 / (b + c)
    return float(chi2.sf(stat, 1)), stat


def run(records, ok_key, label):
    out = {}
    pairs = [("nominal", "interval"), ("mc", "interval"), ("nominal", "mc")]
    for a, b in pairs:
        fp_a, fp_b, fp_both, fp_neither = 0, 0, 0, 0
        fr_a, fr_b, fr_both, fr_neither = 0, 0, 0, 0
        for r in records:
            danger = r["gt_danger_rate"] > 0
            rel_a, rel_b = bool(r[ok_key[a]]), bool(r[ok_key[b]])
            e_a = rel_a and danger
            e_b = rel_b and danger
            fp_both += int(e_a and e_b)
            fp_a += int(e_a and not e_b)
            fp_b += int(e_b and not e_a)
            fp_neither += int(not e_a and not e_b)
            r_a = (not rel_a) and (not danger)
            r_b = (not rel_b) and (not danger)
            fr_both += int(r_a and r_b)
            fr_a += int(r_a and not r_b)
            fr_b += int(r_b and not r_a)
            fr_neither += int(not r_a and not r_b)
        p_fp, s_fp = mcnemar(fp_a, fp_b)
        p_fr, s_fr = mcnemar(fr_a, fr_b)
        out[f"{a}-vs-{b}"] = {
            "false_pass": {"only_A": fp_a, "only_B": fp_b, "both": fp_both,
                           "neither": fp_neither, "mcnemar_chi2": round(s_fp, 2),
                           "p": p_fp},
            "false_reject": {"only_A": fr_a, "only_B": fr_b, "both": fr_both,
                             "neither": fr_neither, "mcnemar_chi2": round(s_fr, 2),
                             "p": p_fr},
        }
        print(f"[{label}] {a} vs {b}: fp discordant ({fp_a},{fp_b}) p={p_fp:.3e}; "
              f"fr discordant ({fr_a},{fr_b}) p={p_fr:.3e}", flush=True)
    return out


def main():
    ok_key = {"nominal": "nominal_ok", "interval": "interval_ok", "mc": "mc_ok"}
    result = {}
    files = [
        ("main_14bus", "experiments/exp_012/results/auth_eval_rte_case14_realistic_full_cases.jsonl"),
        ("llm_14bus", "experiments/exp_017/results/llm_auth_eval_100_cases.jsonl"),
    ]
    for label, path in files:
        if os.path.exists(path):
            result[label] = run(load(path), ok_key, label)
    path_out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "results", "mcnemar_tests.json")
    with open(path_out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"结果已保存: {path_out}")


if __name__ == "__main__":
    main()
