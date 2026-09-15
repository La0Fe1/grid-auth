"""论文图表统一样式（dataviz skill 参考调色板，浅色面，全部经验证）。

规则要点：
- 分类色按固定槽位顺序（1蓝 2橙 3青 4黄），不用轮盘色；
- 量级用蓝色单色顺序阶（浅→深 = 小→大）；
- 禁用双 y 轴；文字用墨色 token 不带系列色；网格细线弱化。
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 参考调色板（浅色面，已验证）
C_SURFACE = "#fcfcfb"
C_INK = "#0b0b0b"
C_SEC = "#52514e"
C_MUTED = "#898781"
C_GRID = "#e1e0d9"
C_AXIS = "#c3c2b7"
CAT = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100",
       5: "#e87ba4", 6: "#008300", 7: "#4a3aa7", 8: "#e34948"}
SEQ_BLUE = {100: "#cde2fb", 150: "#b7d3f6", 200: "#9ec5f4", 250: "#86b6ef",
            300: "#6da7ec", 350: "#5598e7", 400: "#3987e5", 450: "#2a78d6",
            500: "#256abf", 550: "#1c5cab", 600: "#184f95", 650: "#104281",
            700: "#0d366b"}

plt.rcParams.update({
    "figure.facecolor": C_SURFACE,
    "axes.facecolor": C_SURFACE,
    "savefig.facecolor": C_SURFACE,
    "text.color": C_INK,
    "axes.labelcolor": C_INK,
    "xtick.color": C_SEC,
    "ytick.color": C_SEC,
    "axes.edgecolor": C_AXIS,
    "axes.linewidth": 0.8,
    "grid.color": C_GRID,
    "grid.linewidth": 0.6,
    "font.family": "Segoe UI",
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def styled(ax):
    ax.grid(axis="y", alpha=0.7)
    ax.set_axisbelow(True)
    return ax


def savefig(fig, name):
    """保存 PDF（矢量，论文用）+ PNG 300dpi（Word 用）。"""
    pdf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    png_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures_png")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    fig.savefig(os.path.join(pdf_dir, name), bbox_inches="tight")
    fig.savefig(os.path.join(png_dir, name.replace(".pdf", ".png")),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")
