"""重建匿名版：main.tex 去掉作者块 → main_anon.tex（保证两版内容一致）。"""
import re

t = open("main.tex", encoding="utf-8").read()
t = re.sub(r"\\author\{.*?\n\}\n", "", t, flags=re.S)
open("main_anon.tex", "w", encoding="utf-8").write(t)
print("main_anon.tex rebuilt; figures:", t.count("\\begin{figure}"),
      "| Zhenyu left:", t.count("Zhenyu"))
