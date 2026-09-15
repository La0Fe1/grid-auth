"""main_anon.tex → MPCE 初审格式（单栏、行号、编号引用、[H] 定位、按图微调宽度）。"""
import re

src = open("main_anon.tex", encoding="utf-8").read()

src = src.replace(
    "\\documentclass[journal]{IEEEtran}",
    "\\documentclass[11pt,a4paper]{article}\n"
    "\\usepackage[a4paper,margin=2.5cm]{geometry}\n"
    "\\usepackage{lineno}\n\\linenumbers\n"
    "\\usepackage{cite}",
)
src = src.replace("\\usepackage{balance}\n", "")
src = src.replace(
    "\\begin{IEEEkeywords}\nLLM agents, power grid, runtime verification, "
    "interval analysis, power flow,\nsafety.\n\\end{IEEEkeywords}\n",
    "\\noindent\\textbf{Keywords:} LLM agents; power grid; runtime verification; "
    "interval analysis; power flow; safety.\\par\n",
)
src = src.replace("\\begin{abstract}", "\\begin{abstract}\n\\noindent\\textbf{Abstract:} ")
src = src.replace("\\bibliographystyle{IEEEtran}", "\\bibliographystyle{unsrt}")

# 按图微调单栏宽度（新图纵横比）
WIDTHS = {
    "fig_framework.pdf": "0.82",
    "fig_telemetry.pdf": "0.42",
    "fig_main.pdf": "0.48",
    "fig_convergence.pdf": "0.42",
    "fig_llm.pdf": "0.80",
    "fig_eps.pdf": "0.45",
    "fig_ablation.pdf": "0.55",
}
for name, w in WIDTHS.items():
    src = src.replace(f"\\includegraphics[width=0.75\\textwidth]{{figures/{name}}}",
                      f"\\includegraphics[width={w}\\textwidth]{{figures/{name}}}")

open("main_mpce_anon.tex", "w", encoding="utf-8").write(src)
print("main_mpce_anon.tex regenerated")
