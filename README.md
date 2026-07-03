# hf_auto_record

具身智能数据采集团队看板 —— 批量拉取 Hugging Face 组织下的数据集,并统计团队产能、贡献与增长趋势。

面向数据采集团队负责人:一屏掌握「今天产出多少小时数据、谁贡献的、增长趋势如何」。

## 功能

- **批量拉取**:自动发现某组织(默认 `TacVerse`)下全部数据集,增量同步到按日期归档的目录 `pulls/<YYMMDD>/`,并生成聚合报告 `pull_result_*.json`。
- **仅统计**:只读取每个数据集的 `meta/info.json`(不下载数据文件),秒级获得 episodes / frames / 时长等统计。
- **检查新增**:对比 Hub 与本地的数据集名称,列出新增/缺失。
- **团队看板 GUI**(PySide6),三个页签:
  - **看板**:KPI 卡片(总数 / 总小时 / 今日新增小时 / 今日新增 episodes / 目标完成度)+ 可排序筛选的数据集表格(含 robot_type、任务数、上传者、最后更新、今日新增)。
  - **趋势**:每日新增小时(柱)+ 累计小时(折线)。
  - **分组统计**:按 robot_type / 上传者 / 任务 维度汇总。

数据基于 [LeRobot](https://github.com/huggingface/lerobot) 数据集格式(`meta/info.json`、`stats.json` 等);上传者来自 HF 提交记录的 author。

## 环境

```bash
pip install "huggingface_hub" PySide6 pyqtgraph
```

## 用法

命令行(批量拉取整个组织):

```bash
python download_dataset.py                     # 拉取默认组织全部数据集
python download_dataset.py --org <ORG>         # 指定组织
python download_dataset.py --repo-id A/x --repo-id B/y   # 只拉指定数据集
```

图形界面(团队看板):

```bash
python gui_app.py
```

## 文件

- `download_dataset.py` —— 拉取 / 统计 / 分析的核心逻辑(CLI 与 GUI 共用)。
- `gui_app.py` —— PySide6 团队看板。
