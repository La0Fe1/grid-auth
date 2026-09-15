"""main_anon.tex → MPCE 初审格式（单栏、行号、编号引用、图宽调整）。"""
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
src = src.replace("width=\\columnwidth", "width=0.75\\textwidth")

open("main_mpce_anon.tex", "w", encoding="utf-8").write(src)
print("main_mpce_anon.tex written;", "IEEEkeywords" in src and "IEEEkeywords still present" or "IEEEkeywords removed")
