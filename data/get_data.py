"""数据集获取脚本（AGENTS.md 3.2：数据获取必须可执行、可复现）。

幂等：已存在则跳过。三个来源按优先级：
1. 项目内 data/grid2op_data/ 已有 → 跳过；
2. 旧主目录 ~/data_grid2op 已有解压数据（只读来源，见 data/README.md）→ 复制进项目；
3. 均无 → 从 Grid2Op 官方 releases（datasets-v0.1.0，URL 已核实 2026-09-13）
   下载 tar.bz2 到 data/raw/ 并解压到 data/grid2op_data/。

说明（2026-09-13 已核实）：
- l2rpn_icaps_2021（36 变电站）与 l2rpn_idf_2023（118 变电站）的完整数据集不在
  官方 releases 中，本仓库使用 grid2op 内置开发模式（make(dataset=..., test=True)，
  数据内嵌于 grid2op 包），无需下载。
- l2rpn_2019 已弃用：官方 datasets-v0.1.0 的 tarball 也是旧 PypowNet 格式
  （_N_*.csv.bz2），grid2op 1.12.5 加载报 ChronicsError（上游打包不一致，
  2026-09-13 实测确认）。36-bus 系统改用 l2rpn_icaps_2021。

用法（项目根目录）：python data/get_data.py
"""
import os
import shutil
import sys
import tarfile
import urllib.request

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_PROJECT_ROOT, "data", "grid2op_data")
_RAW_DIR = os.path.join(_PROJECT_ROOT, "data", "raw")
_LEGACY_HOME = os.path.expanduser("~/data_grid2op")
_BASE_URL = "https://github.com/Tezirg/Grid2Op/releases/download/datasets-v0.1.0/{}"

DATASETS = {
    "rte_case14_realistic": "rte_case14_realistic.tar.bz2",
}


def _valid(dst):
    """有效性校验：config.py 存在、chronics 有非空场景目录、文件格式为新格式。

    （2026-09-13 两条教训：①遗留 l2rpn_2019 的 chronics 目录为空；
    ②旧 PypowNet 格式文件名是 _N_*.csv.bz2，grid2op 1.12 加载报 ChronicsError。
    新格式场景目录内含 load_*.csv.bz2 等文件。）
    """
    if not (os.path.isdir(dst) and os.path.exists(os.path.join(dst, "config.py"))):
        return False
    chronics = os.path.join(dst, "chronics")
    if not os.path.isdir(chronics):
        return False
    for d in sorted(os.listdir(chronics)):
        p = os.path.join(chronics, d)
        if not os.path.isdir(p):
            continue
        files = os.listdir(p)
        if not files:
            continue
        if any(f.startswith("load") for f in files):
            return True
        if any(f.startswith("_N_") for f in files):
            return False  # 旧格式，不可用
    return False


def ensure_dataset(name):
    """确保数据集就位，返回 (路径, 来源说明)。无效副本会被删除并回退下一来源。"""
    dst = os.path.join(_DATA_DIR, name)
    if _valid(dst):
        return dst, "已存在，跳过"

    legacy = os.path.join(_LEGACY_HOME, name)
    if _valid(legacy):
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        os.makedirs(_DATA_DIR, exist_ok=True)
        shutil.copytree(legacy, dst)
        if _valid(dst):
            return dst, "从旧主目录复制（只读来源）"
        shutil.rmtree(dst)
        print(f"旧主目录副本无效（{legacy}），回退到 tarball/官方下载")

    tarball_name = DATASETS[name]
    tarball = os.path.join(_RAW_DIR, tarball_name)
    os.makedirs(_RAW_DIR, exist_ok=True)
    if not os.path.exists(tarball):
        url = _BASE_URL.format(tarball_name)
        print(f"下载 {url}")
        urllib.request.urlretrieve(url, tarball)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(_DATA_DIR, exist_ok=True)
    with tarfile.open(tarball, "r:bz2") as tf:
        tf.extractall(_DATA_DIR)
    if not _valid(dst):
        raise RuntimeError(f"{name} 的 tarball 解压后无效（可能损坏），请重新下载")
    return dst, "官方 release 下载并解压"


def verify_env(dataset, test):
    """快速加载校验（冒烟级）；失败不中断，如实打印。

    注意：用项目内路径（data/grid2op_data/<name>）而非环境名——环境名会
    解析到用户主目录 ~/data_grid2op（0.1 违规且可能是损坏副本，2026-09-13 教训）。
    """
    from grid2op import make
    from lightsim2grid import LightSimBackend
    local = os.path.join(_DATA_DIR, dataset)
    target = local if os.path.isdir(local) else dataset
    try:
        env = make(dataset=target, test=test, backend=LightSimBackend())
        print(f"  [OK] {dataset} test={test}: n_sub={env.n_sub} n_line={env.n_line}")
    except Exception as e:
        print(f"  [FAIL] {dataset} test={test}: {type(e).__name__}: {str(e)[:120]}")


def main():
    for name in DATASETS:
        path, how = ensure_dataset(name)
        print(f"{name}: {how} → {path}")

    print("\n校验：")
    verify_env("rte_case14_realistic", False)
    verify_env("l2rpn_icaps_2021", True)   # 内置开发模式
    verify_env("l2rpn_idf_2023", True)     # 内置开发模式
    print("\n完成。icaps_2021/idf_2023 完整数据集无官方下载源（2026-09-13 核实），"
          "本仓库用内置开发模式评估跨算例部署；l2rpn_2019 因上游格式不兼容弃用（见注释）。")


if __name__ == "__main__":
    main()
