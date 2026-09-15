"""写作门控（AGENTS.md 4.2.1 错别字检查）：对 main.tex 的指定章节逐段拼写检查；
同一段落 ≥2 个警告 → 必须重写该段。

实现说明（decisions.md 记录）：本机无 Java（language_tool_python 不可用）、
无 aspell → 用 pyspellchecker（纯 Python 拼写检查）执行错别字门控；
语法层面由逐句人工复核 + 领域术语白名单补充。

用法：python paper/lang_gate.py [section名]（默认 Introduction）
"""
import re
import sys

from spellchecker import SpellChecker


def extract_section(tex_path, sec_name):
    with open(tex_path, encoding="utf-8") as f:
        text = f.read()
    if sec_name == "Abstract":
        m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    else:
        m = re.search(rf"\\section\{{{re.escape(sec_name)}\}}(.*?)(?=\\section|\Z)",
                      text, re.S)
    if not m:
        raise SystemExit(f"未找到章节: {sec_name}")
    return m.group(1)


def strip_latex(s):
    s = re.sub(r"\\cite\{[^}]*\}", "", s)
    s = re.sub(r"\\ref\{[^}]*\}", "Section X", s)
    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\begin\{itemize\}|\\end\{itemize\}", "", s)
    s = re.sub(r"\\item", "", s)
    s = re.sub(r"\\textit\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\textbf\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?\{[^}]*\}", "", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = re.sub(r"[%$&_{}]", " ", s)
    return s


# 领域术语白名单（电网/LLM 专有名词）
DOMAIN_WHITELIST = {
    "twin", "twingridshield", "safevolt", "elecbench", "poweragentbench",
    "grid2op", "l2rpn", "telemetry", "telemetries", "proposer", "proposers",
    "chronics", "bus", "buses", "substation", "substations", "island",
    "islanding", "disconnect", "disconnections", "reactive", "injections",
    "nadir", "powerline", "powerlines", "benchmarks", "voltage", "voltages",
    "llm", "rl", "dc", "lp", "mc", "iec", "ieee", "runtime", "workflows",
    "false-pass", "false-reject", "trade-off", "closed-loop", "well-formed",
    "fine-tuned", "self-verifying", "verifier-aware", "telemetry-based",
    "network-model", "cross-system", "voltage-angle", "decision-support",
    "dispatcher-assistant", "control-room", "uncertainty-robust",
    "susceptance", "lps", "json", "endpoints", "snapshots", "rad",
    "multi-step", "uncertainties", "icaps", "idf", "datasets", "pdf", "lccc",
    "op", "rpn", "geq",
}


def spell_check(text):
    checker = SpellChecker(language="en")
    words = re.findall(r"[A-Za-z][A-Za-z\-']*", text)
    unknown = set()
    for w in checker.unknown(words):
        wl = w.lower()
        if wl in DOMAIN_WHITELIST:
            continue
        if "-" in w:
            # 连字符复合词：拆开逐部验证（closed-loop → closed+loop）
            parts = [p for p in w.split("-") if p]
            if parts and not checker.unknown(parts):
                continue
        unknown.add(w)
    return sorted(unknown)


def main():
    sec = sys.argv[1] if len(sys.argv) > 1 else "Introduction"
    raw = extract_section("paper/main.tex", sec)
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
    checker = SpellChecker(language="en")
    total = 0
    rewrite_needed = False
    for i, p in enumerate(paragraphs):
        plain = strip_latex(p)
        if len(plain.split()) < 5:
            continue
        unknown = spell_check(plain)
        n = len(unknown)
        total += n
        flag = "  <== REWRITE" if n >= 2 else ""
        if n:
            rewrite_needed = rewrite_needed or n >= 2
            print(f"段{i}: {n} 疑似拼写{flag}: {unknown[:8]}")
    print(f"\n{sec} 合计拼写警告: {total}（段落数 {len(paragraphs)}）")
    print("门控判定:", "存在需重写的段落" if rewrite_needed else "通过")


if __name__ == "__main__":
    main()
