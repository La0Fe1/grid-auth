# 数据说明

> 按 AGENTS.md 3.1：数据来源、许可、预处理步骤（可执行脚本）。

## 数据来源与许可（2026-09-13 更新）
- **rte_case14_realistic**（14 变电站 / 20 线路，训练与主评估）：Grid2Op 官方数据集（RTE 发布，L2RPN 比赛用）。来源：官方 releases `datasets-v0.1.0`，或本机旧主目录 `~/data_grid2op`（只读复制来源）。许可：MPL-2.0（Grid2Op 项目）。
- **l2rpn_icaps_2021**（36 变电站 / 59 线路，跨算例部署评估）：grid2op 内置开发模式（test=True），数据内嵌于 grid2op 包；完整数据集无官方下载源。
- **l2rpn_idf_2023**（118 变电站 / 186 线路，跨算例部署评估）：同上，内置开发模式。
- **l2rpn_2019 已弃用**：官方 tarball 为旧 PypowNet 格式（_N_*.csv.bz2），grid2op 1.12.5 加载报 ChronicsError（2026-09-13 实测，上游打包不一致）。

## 获取与预处理（可执行）
- 获取脚本：`python data/get_data.py`（幂等：校验 config.py + chronics 非空 + 新格式；来源优先级：项目已有 → 旧主目录复制 → 官方下载）。
- 预处理：环境观测 → 图结构的转换在 `src/utils/graph.py`（纯函数，含归一化），无需离线预处理文件。
- N-1 标签生成：`python -m src.training.gen_n1_labels`（输出 data/processed/n1_gate_labels.pkl）。

## 遗留数据说明
- 顶层 `data/`（旧布局）原有的 `l2rpn_2019.tar.bz2`（旧格式）已删除；`rte_case14_realistic.tar.bz2` 位于 `data/raw/`（git 忽略，由 get_data.py 重新获取）。
