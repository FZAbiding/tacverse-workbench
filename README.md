# TacVerse 多模态物理具身数据集工作台

（tacverse-workbench）批量拉取 Hugging Face 组织下的数据集，统计团队产能、贡献与增长趋势。

面向数据采集团队负责人：一屏掌握「今天产出多少小时数据、谁贡献的、质量如何、增长趋势如何」。

数据基于 [LeRobot](https://github.com/huggingface/lerobot) 数据集格式（`meta/info.json`、`stats.json` 等）；上传者取自各数据集 HF 提交记录的初始 author。

---

## 功能

- **批量拉取**：自动发现某组织（默认 `TacVerse`）下全部数据集，增量同步到按日期归档的 `pulls/<YYMMDD>/`，并生成聚合报告。
- **仅统计**（快）：只读取每个数据集的 `meta/info.json`（不下载数据文件），秒级获得 episodes / frames / 时长等统计。
- **检查新增**：对比 Hub 与本地的数据集名称，列出新增 / 缺失。
- **团队看板 GUI**（PySide6），页签：
  - **看板**：KPI 卡片（数据集总数 / 总小时 / 总 episodes / 今日新增小时 / 今日新增 episodes / 目标完成度 / **今日 MVP ⭐**）+ 可排序筛选的数据集表格（含 **均时长(s)** 质量指标、robot_type、任务数、HF ID、上传者中文名、最后更新、今日新增）。表格默认按 HF「最近更新」排序，和网页一致。
  - **趋势**：每日新增小时（柱，仅显示实际拉取过的日期，不留空白）+ 累计小时（折线）。
  - **分组统计**：按 上传者 / 任务 / robot_type 维度汇总（横向柱状，中文名不重叠），并以 Hugging Face `last_modified` 归日、优先使用 commit history 差分、缺失时用本地快照兜底展示单组单日新增总时长。
  - **数据集编辑**：左侧是与看板一致的数据集详情表（选中要操作的数据集）；右侧两组功能——① **改名 / 改 Prompt**（本地 pyarrow 生成新副本，可推送回 Hub）；② 调用 lerobot 官方 `dataset_tools` 的 **删除 episodes / 拆分 / 合并 / 增加特征 / 删除特征**。详见下方[「数据集编辑」](#数据集编辑生成新副本不改动原数据)。
  - **Viewer**：内嵌 `xense_lerobot_viewer`（Next.js），3D 回放 / 语言标注 / 标签编辑。
- **质量检查**：命名规范、均时长（20~600s）、Prompt 词数（10~50 词）等规则内置，看板表格用 ✅/⚠️/❌ 标注；阈值在 `config.json` 的 `checks` 段可调。
- **登录状态指示器**：顶栏实时显示「已登录: xxx · 可见 N 个数据集」，一眼判断 token 权限是否正确（私有库需要有权限的账号才可见）。
- **切换账号**：顶栏按钮，运行时粘贴新的 HF token 即可切换（仅本次运行有效，不落盘）。

---

## 环境安装

### 1. Python 环境

推荐统一使用 `lerobot-xense`（mamba/conda）环境：

```bash
mamba activate lerobot-xense
```

### 2. Python 依赖

```bash
pip install "huggingface_hub" PySide6 pyqtgraph pyarrow
```

> `pyarrow` 用于读/写数据集的 `meta/*.parquet`（「数据集编辑」页签改 prompt 时会写回）；
> 上传编辑后的副本用已有的 `huggingface_hub`。在 `lerobot-xense` 环境里这些通常已安装。
>
> **务必在 `lerobot-xense` 环境里启动**（`mamba activate lerobot-xense && python main_app.py`）。
> 「数据集编辑」页签里的 **删除/拆分/合并/增删特征** 会以子进程方式调用 lerobot 官方
> `dataset_tools`，因此需要该环境里的 `lerobot` 包；改名 / 改 prompt 则是本地 pyarrow 实现，
> 不依赖 lerobot。

### 3. 系统依赖（Linux 必装 ⚠️）

PySide6 6.5+ 的 Qt xcb 平台插件需要 `libxcb-cursor0`，**缺了会直接报错无法启动**：

```
From 6.5.0, xcb-cursor0 or libxcb-cursor0 is needed to load the Qt xcb platform plugin.
```

安装：

```bash
sudo apt update
sudo apt install -y libxcb-cursor0
```

> 之前有同事反馈起不来，基本都是缺这个系统库。装上即可。

---

## HF Token 配置（拉取私有数据集的关键 ⚠️）

TacVerse 的大部分数据集是**私有**的。HF 接口只会返回「当前 token 有权限看到」的仓库——**匿名或无权限的 token 只能看到公开的少数几个**。所以拉全部数据集，必须用一个**属于该组织、有读权限**的账号 token。

### token 从哪来

用有组织权限的账号登录 https://huggingface.co/settings/tokens ，**新建 token 时 Token type 选 `Read`（经典读 token）**——它能访问该账号有权限的所有仓库（含组织私有库）。
> 不要用 fine-grained（细粒度）token，除非你在其 scope 里显式勾上了目标组织的「Read access to contents of all repos」，否则照样看不到私有库。

### 怎么让程序用上 token（任选其一）

- **方式 A（推荐，一次生效）**：命令行登录，程序会自动读取缓存的登录 token：
  ```bash
  huggingface-cli login      # 粘贴上面的 Read token
  ```
- **方式 B（临时）**：设置环境变量后启动：
  ```bash
  export HF_TOKEN=hf_你的token
  python main_app.py
  ```
- **方式 C（运行时）**：直接开 GUI，点顶栏 **「切换账号」** 粘贴 token。

程序取 token 的优先级：`$HF_TOKEN` → `huggingface-cli login` 缓存 → 匿名。

### 验证 token 是否有权限

```bash
python -c "from huggingface_hub import HfApi; print(HfApi().dataset_info('TacVerse/taccap-g1-candybowl-0702').private)"
```
能打印 `True` 说明有权限；报 `404` 说明账号 / token 权限不够（需组织管理员把你的账号加进组织，或换经典 Read token）。启动 GUI 后，顶栏指示器显示的「可见 N 个」也能直接反映权限是否正确。

---

## 用法

### 命令行（批量拉取整个组织）

```bash
python download_dataset.py                                # 拉取默认组织全部数据集
python download_dataset.py --org <ORG>                    # 指定组织
python download_dataset.py --repo-id A/x --repo-id B/y    # 只拉指定数据集
```

### 图形界面（团队看板）

```bash
python main_app.py
```

进去后：点 **「仅拉取统计信息」**（快，只读信息，不下载）、**「下载当前选中数据集」**（只下选中的一个，省时）或 **「拉取组织及其下所有数据集」**（全量下载 + 累积历史，较慢）。

### 数据集编辑（生成新副本，不改动原数据）

在 **「数据集编辑」** 页签，左表选中一个**已下载**的数据集（未下载的行不能编辑；可先用「下载当前选中数据集」拉下来）。所有操作都**输出为新副本**到 `pulls/<今天>/<输出名>/`，**不会改动原数据集**。

**① 改名 / 改 Prompt（本地实现，无需 lerobot）**
- 直接编辑该数据集的 Prompt（`meta/tasks.parquet` 里的任务指令）和/或输出名，点 **「生成新副本」**。
- 只改写元数据（`meta/`），`data/` 与 `videos/` 用硬链接进副本、**不复制大文件**，秒级完成。
- 需要时点 **「推送到 Hub」** 把副本上传（默认私有仓库）。

**② 数据集操作（调用 lerobot 官方 `dataset_tools`）**
在下拉框选择操作，填参数后点 **「执行操作」**：

| 操作 | 参数 | 说明 |
|---|---|---|
| 删除 episodes | 序号，如 `0,2,5` | 删除指定 episode 并重建索引 |
| 拆分数据集 | `train:0.8,val:0.2` 或 `train:0-4,val:5-6` | 按比例或序号区间拆成多个 `<输出名>_train`… |
| 合并数据集 | 勾选多个已下载数据集 | 合并成一个（输出名用「输出数据集名」） |
| 增加特征 | 特征名 / dtype / shape / 填充值 | 新增一个常量填充的特征列 |
| 删除特征 | 勾选要删的特征 / 相机 | 移除特征（必填字段不可删） |

> 这些操作以**子进程**方式运行 lerobot，保证与官方框架一致，并把崩溃与主界面隔离。删除/拆分/合并会**重编码视频**（用 CPU `libx264`，较慢，请耐心等待）。完成后新数据集自动出现在表格里（标记为「已下载」）。

---

## 配置与数据文件

- **`config.json`**（核心配置，随仓库提交）：
  - `uploader_names`：**你手工维护**的 `HF ID -> 中文名` 映射。新增成员在这里加一行 `"hf_id": "中文名"`（改完重启 GUI 生效）；查不到的 ID 在界面显示为 `未知`。
- **`pull_history.local.json`**（本地运行历史，已被 git 忽略）：
  - 程序每次拉取 / 统计会自动追加精简历史快照，作为每日新增 / 趋势的数据源。
  - 该文件只保存在本机，不提交到仓库，避免泄露或误同步数据集统计历史。
- **`hf_change_history.local.json`**（本地 HF 变更历史，已被 git 忽略）：
  - 程序统计 Hugging Face commit history 中 `meta/info.json` 的差分，作为今日新增 / MVP / 单组单日新增的优先数值来源；缺失项才使用本地快照辅助兜底。
  - 该文件只保存在本机，不提交到仓库，避免同步提交历史统计细节。
- **`pulls/`**：拉取下来的原始数据集（含多 GB 视频），**已被 git 忽略**，不随代码同步，以节省仓库体积。

---

## 文件

- `main_app.py` —— PySide6 团队看板（GUI 入口）。
- `download_dataset.py` —— 拉取 / 统计 / 分析 / 配置读写的核心逻辑（CLI 与 GUI 共用）。
- `checks.py` —— 数据集质量检查插件注册表（命名 / 均时长 / Prompt 等规则）。
- `dataset_editor.py` —— 「改名 / 改 Prompt」的本地 pyarrow 实现（Qt-free，不依赖 lerobot）。
- `lerobot_ops.py` / `lerobot_ops_runner.py` —— 删除 / 拆分 / 合并 / 增删特征：workbench 侧封装 + 调用 lerobot `dataset_tools` 的子进程执行器。
- `config.json` —— 上传者中文名映射 + 质量检查阈值。
- `pull_history.local.json` —— 本地拉取 / 统计历史（自动生成，git 忽略）。
- `hf_change_history.local.json` —— 本地 Hugging Face 变更历史缓存（自动生成，git 忽略）。
