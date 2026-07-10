#!/usr/bin/env python3
"""TacVerse 多模态物理具身数据集工作台 — PySide6 dashboard over Hugging Face.

Wraps the logic in download_dataset.py. Top bar (org combo + actions + progress
+ speed) is shared; below it a tabbed dashboard:

  * 看板   -> KPI cards (+ today's MVP) + filterable, sortable dataset table.
  * 趋势   -> daily new-hours bar + cumulative-hours line (pyqtgraph).
  * 分组统计 -> rollup by uploader / task / robot_type, table + horizontal bars.

Buttons: 统计当前数据集 (stats only, no download) / 拉取当前数据集 (download) /
检查新增数据集 (name diff) / 打开本地目录 / 切换账号 (swap HF token).

Run in the lerobot-xense env:  python main_app.py
"""

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox,
    QProgressBar, QPushButton, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
    QWidget,
)

import annotations_reader as ann
import tasks_reader as tsk
import checks as chk_mod
import viewer_service as vsvc
import download_dataset as dd

OUT_DIR = "pulls"
RECENT_ORGS = ["TacVerse", "Xense"]  # seeds the editable org combo

# HF uploader id -> Chinese name lives in the unified config.json ("uploader_names"
# section — edit that to add people). Ids with no entry render as 未知. Loaded once
# at startup; edit the file then restart to pick up new names.
_UPLOADER_NAMES = dd.load_uploader_names()

# Thresholds for the custom quality checks (config.json "checks" section).
# Loaded once at startup; edit the file then restart to change standards.
_CHECKS_CFG = dd.load_config().get("checks", {})


def uploader_cn(hf_id):
    """Map an HF uploader id to its Chinese name, or 未知 if absent/unknown."""
    return _UPLOADER_NAMES.get(hf_id, "未知") if hf_id else "未知"


# Local, git-ignored file where the "切换账号" dialog persists its token so it
# survives restarts without being committed / shared with other users.
TOKEN_FILE = Path(__file__).resolve().parent / ".hf_token"


def load_saved_token():
    """Return the locally-persisted token (from the 切换账号 dialog), or None."""
    try:
        tok = TOKEN_FILE.read_text(encoding="utf-8").strip()
        return tok or None
    except OSError:
        return None


def save_token(tok):
    """Persist `tok` to the git-ignored .hf_token (0600), or clear it if empty."""
    try:
        if tok and tok.strip():
            TOKEN_FILE.write_text(tok.strip() + "\n", encoding="utf-8")
            os.chmod(TOKEN_FILE, 0o600)
        elif TOKEN_FILE.exists():
            TOKEN_FILE.unlink()
    except OSError:
        pass


def resolve_token():
    """HF token to talk to the Hub with.

    Priority: the token saved by the "切换账号" dialog (so the account you pick
    in the UI sticks across restarts), then $HF_TOKEN, then the token cached by
    `huggingface-cli login`. Private datasets are only visible when this token
    belongs to an org member — e.g. a TacVerse member sees TacVerse's private
    repos.
    """
    saved = load_saved_token()
    if saved:
        return saved
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok
    try:
        from huggingface_hub import get_token
        return get_token()
    except Exception:
        return None

pg.setConfigOptions(background="w", foreground="k", antialias=True)

# Dashboard table columns: (header, dataset key, kind). "__delta__" is special.
TABLE_COLS = [
    ("数据集", "dataset_name", "str"),
    ("本地", "__local__", "num"),  # raw files downloaded under pulls/ → openable in viewer
    ("episodes", "total_episodes", "num"),
    ("frames", "total_frames", "num"),
    ("小时", "duration_hours", "num"),
    ("均时长(s)", "__avg_sec__", "num"),  # avg seconds/episode — quality signal
    ("检查", "__check__", "num"),  # custom quality-check badge (✅/⚠️N/❌N)
    ("fps", "fps", "num"),
    ("robot_type", "robot_type", "str"),
    ("任务数", "total_tasks", "num"),
    ("HF ID", "uploader", "str"),
    ("上传者", "__uploader_cn__", "str"),
    ("最后更新", "last_modified", "date"),
    ("今日新增ep", "__delta__", "num"),
]

# Column that carries last_modified — the table's default sort key. Derived so it
# stays correct if columns are inserted/reordered above.
DATE_COL = next(i for i, (_, k, _) in enumerate(TABLE_COLS) if k == "last_modified")

# Order = dropdown order; first entry (上传者) is the default. robot_type last.
ROLLUP_DIMS = {
    "上传者": lambda d: uploader_cn(d.get("uploader")),
    "任务": lambda d: dd.task_prefix(d["dataset_name"]),
    "robot_type": lambda d: d.get("robot_type"),
}


def fmt_day(yymmdd):
    """'260703' -> '2026-07-03'. Returns the input unchanged if unparseable."""
    try:
        return dt.datetime.strptime(yymmdd, "%y%m%d").strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return yymmdd or "—"


def days_between(yymmdd_from, yymmdd_to):
    """Whole days from one YYMMDD date to another, or None if either is unparseable."""
    try:
        a = dt.datetime.strptime(yymmdd_from, "%y%m%d")
        b = dt.datetime.strptime(yymmdd_to, "%y%m%d")
        return (b - a).days
    except (ValueError, TypeError):
        return None


def fmt_value(v):
    """Render a value: thousands separators for numbers, — for None/empty."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:,}"
    if v is None or v == "":
        return "—"
    return str(v)


def fmt_speed(bytes_per_sec):
    """Human-readable transfer rate, e.g. '12.3 MB/s'."""
    rate = max(float(bytes_per_sec), 0.0)
    for unit in ("B/s", "KB/s", "MB/s", "GB/s"):
        if rate < 1024 or unit == "GB/s":
            return f"{rate:.1f} {unit}"
        rate /= 1024


def dir_size(path):
    """Total bytes of materialized files under path (skips hf .cache blobs)."""
    p = Path(path)
    if not p.exists():
        return 0
    total = 0
    for f in p.rglob("*"):
        if ".cache" in f.parts:
            continue
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            pass
    return total


class NumericItem(QTableWidgetItem):
    """Table item that displays formatted text but sorts by a numeric key."""

    def __init__(self, text, sort_key):
        super().__init__(text)
        self.sort_key = sort_key

    def __lt__(self, other):
        if isinstance(other, NumericItem):
            return self.sort_key < other.sort_key
        return super().__lt__(other)


# --------------------------------------------------------------------------- #
# Worker threads (network + downloads run off the UI thread)
# --------------------------------------------------------------------------- #
class PullWorker(QThread):
    """Discover an org's datasets and pull them all, streaming progress."""

    log = Signal(str)
    progress = Signal(int, int)  # done, total
    done = Signal(dict, str)     # report, out_path
    error = Signal(str)

    def __init__(self, org, out_dir, token):
        super().__init__()
        self.org, self.out_dir, self.token = org, out_dir, token

    def run(self):
        try:
            dd.normalize_proxy_env()
            self.log.emit(f"Discovering datasets under '{self.org}' ...")
            meta = dd.discover_datasets_meta(self.org, self.token)
            repo_ids = [m["id"] for m in meta]
            meta_map = {m["id"]: m["last_modified"] for m in meta}
            self.log.emit(f"Found {len(repo_ids)} datasets.")
            if not repo_ids:
                self.error.emit(f"No datasets found under '{self.org}'.")
                return
            report, out_path = dd.run_pull(
                repo_ids, out_dir=self.out_dir, org=self.org, token=self.token,
                meta_map=meta_map, with_uploader=True,
                log=self.log.emit, progress=lambda d, t: self.progress.emit(d, t),
            )
            self.done.emit(report, str(out_path) if out_path else "")
        except Exception as exc:
            self.error.emit(str(exc))


class StatsWorker(QThread):
    """Fetch stats only (meta/info.json + commits) — no dataset files pulled."""

    log = Signal(str)
    progress = Signal(int, int)
    done = Signal(dict)
    error = Signal(str)

    def __init__(self, org, token):
        super().__init__()
        self.org, self.token = org, token

    def run(self):
        try:
            dd.normalize_proxy_env()
            self.log.emit(f"Discovering datasets under '{self.org}' ...")
            meta = dd.discover_datasets_meta(self.org, self.token)
            repo_ids = [m["id"] for m in meta]
            meta_map = {m["id"]: m["last_modified"] for m in meta}
            self.log.emit(f"Found {len(repo_ids)} datasets.")
            if not repo_ids:
                self.error.emit(f"No datasets found under '{self.org}'.")
                return
            report = dd.collect_stats(
                repo_ids, org=self.org, token=self.token,
                meta_map=meta_map, with_uploader=True,
                log=self.log.emit, progress=lambda d, t: self.progress.emit(d, t),
            )
            self.done.emit(report)
        except Exception as exc:
            self.error.emit(str(exc))


class CheckWorker(QThread):
    """Compare Hub dataset names against the last pulled report (names only)."""

    result = Signal(list, list, int, int)  # new, removed, hub_count, local_count
    error = Signal(str)

    def __init__(self, org, out_dir, token):
        super().__init__()
        self.org, self.out_dir, self.token = org, out_dir, token

    def run(self):
        try:
            dd.normalize_proxy_env()
            hub = set(dd.discover_datasets(self.org, self.token))
            local = set()
            latest = dd.find_latest_report(self.out_dir)
            if latest:
                report = json.loads(Path(latest).read_text())
                local = {d["dataset_name"] for d in report.get("datasets", [])}
            self.result.emit(sorted(hub - local), sorted(local - hub),
                             len(hub), len(local))
        except Exception as exc:
            self.error.emit(str(exc))


class IdentityWorker(QThread):
    """Resolve who the current token logs in as and how many org datasets it can
    see — so the status bar can flag token/permission problems at a glance."""

    done = Signal(str, bool, str, int)  # username, has_token, org, count(-1=err)

    def __init__(self, org, token):
        super().__init__()
        self.org, self.token = org, token

    def run(self):
        dd.normalize_proxy_env()
        name = ""
        if self.token:
            try:
                from huggingface_hub import HfApi
                name = HfApi().whoami(token=self.token).get("name", "") or ""
            except Exception:
                name = ""  # token present but invalid/expired
        try:
            count = len(dd.discover_datasets_meta(self.org, self.token))
        except Exception:
            count = -1
        self.done.emit(name, bool(self.token), self.org, count)


class ReportWorker(QThread):
    """Fetch the viewer's /report analysis off the UI thread (it can take tens
    of seconds). `seq` lets the UI ignore results from stale selections."""

    done = Signal(int, str, object, str)  # seq, rel_path, report|None, error

    def __init__(self, viewer, rel_path, seq):
        super().__init__()
        self.viewer, self.rel_path, self.seq = viewer, rel_path, seq

    def run(self):
        report, err = self.viewer.report(self.rel_path, timeout=180)
        self.done.emit(self.seq, self.rel_path, report, err or "")


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TacVerse 多模态物理具身数据集工作台")
        # Large default for a 2560x1440 display, but kept clearly below the work
        # area (~82% w / ~85% h) and centred: opening too close to full-screen
        # makes some window managers auto-maximize the window a moment after it
        # maps. Start in the normal (non-maximized) state explicitly.
        target_w, target_h = 2200, 1300
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            target_w = min(target_w, int(avail.width() * 0.82))
            target_h = min(target_h, int(avail.height() * 0.85))
        self.setWindowState(Qt.WindowNoState)
        self.resize(target_w, target_h)
        if screen:
            frame = self.frameGeometry()
            frame.moveCenter(avail.center())
            self.move(frame.topLeft())
        self.token = resolve_token()
        self.worker = None
        self.report = None
        self.history = []
        self._id_workers = []  # in-flight IdentityWorkers (kept alive until done)
        self._id_seq = 0       # monotonic id; only the latest check may update UI
        # Vendored viewer (xense_lerobot_viewer) managed as a black-box service.
        # Port 3001 keeps it separate from any viewer the user runs on 3000, so
        # workbench always launches its own instance bound to the pulls root.
        self.viewer = vsvc.ViewerService(port=3001)
        self._report_workers = []   # in-flight ReportWorkers
        self._report_seq = 0        # only the latest selection's report renders
        self._report_cache = {}     # rel_path -> report dict (per session)

        self._build_ui()

        # Auto-start the viewer so the analysis panel works without a manual
        # step. Non-blocking; the Viewer tab's status shows progress.
        if self.viewer.available():
            self.viewer.start(self._viewer_root(), wait=False)

        self._watch_dir = None
        self._prev_bytes = 0
        self._prev_t = None
        self.speed_timer = QTimer(self)
        self.speed_timer.setInterval(1000)
        self.speed_timer.timeout.connect(self._tick_speed)

        # Trends can render immediately from accumulated history; the 看板
        # KPI/table stay empty until 统计/拉取.
        self.history = dd.load_history(OUT_DIR)
        self._refresh_trends()
        self.status.setText("就绪，请选择「统计当前数据集」(快) 或「拉取当前数据集」。")
        self._refresh_identity()  # populate the login/visibility indicator

    # ---- UI construction -------------------------------------------------- #
    def _build_ui(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("组织:"))
        self.org_combo = QComboBox()
        self.org_combo.setEditable(True)
        self.org_combo.addItems(RECENT_ORGS)
        self.org_combo.setMinimumWidth(160)
        self.org_combo.currentIndexChanged.connect(self._refresh_identity)
        self.org_combo.lineEdit().editingFinished.connect(self._refresh_identity)
        top.addWidget(self.org_combo)

        # Two primary actions, statistics first (fast, read-only) then the full
        # pull. They get a bold colored look; the utilities that follow stay plain
        # and sit behind a vertical divider so the split reads at a glance.
        self.btn_stats = QPushButton("统计当前数据集")
        self.btn_pull = QPushButton("拉取当前数据集")
        self.btn_check = QPushButton("检查新增数据集")
        self.btn_open = QPushButton("打开本地目录")
        self.btn_stats.clicked.connect(self.on_stats)
        self.btn_pull.clicked.connect(self.on_pull)
        self.btn_check.clicked.connect(self.on_check)
        self.btn_open.clicked.connect(self.on_open_dir)

        primary_css = (
            "QPushButton { font-weight: bold; padding: 6px 16px; border-radius: 6px;"
            " color: white; background: %s; }"
            "QPushButton:hover { background: %s; }"
            "QPushButton:disabled { background: #B0B0B0; }"
        )
        self.btn_stats.setStyleSheet(primary_css % ("#34A853", "#2E9247"))
        self.btn_pull.setStyleSheet(primary_css % ("#4C8BF5", "#3B7AE0"))
        secondary_css = (
            "QPushButton { padding: 5px 12px; border-radius: 6px; color: #444;"
            " border: 1px solid #C4C4C4; background: #F5F5F5; }"
            "QPushButton:hover { background: #ECECEC; }"
        )
        for b in (self.btn_stats, self.btn_pull):
            b.setMinimumHeight(34)
            top.addWidget(b)
        divider = QFrame()
        divider.setFrameShape(QFrame.VLine)
        divider.setFrameShadow(QFrame.Sunken)
        top.addWidget(divider)
        for b in (self.btn_check, self.btn_open):
            b.setStyleSheet(secondary_css)
            top.addWidget(b)

        # Viewer service controls, up here in the toolbar (the "Viewer" tab is
        # kept for now but may be removed later — these are the canonical ones).
        vdiv = QFrame()
        vdiv.setFrameShape(QFrame.VLine)
        vdiv.setFrameShadow(QFrame.Sunken)
        top.addWidget(vdiv)
        self.top_viewer_dot = QLabel("● Viewer")
        self.top_viewer_dot.setToolTip("Viewer 服务状态")
        top.addWidget(self.top_viewer_dot)
        self.top_viewer_start = QPushButton("启动")
        self.top_viewer_stop = QPushButton("停止")
        self.top_viewer_home = QPushButton("首页")
        self.top_viewer_start.clicked.connect(self._viewer_start)
        self.top_viewer_stop.clicked.connect(self._viewer_stop)
        self.top_viewer_home.clicked.connect(self._viewer_open_home)
        for b in (self.top_viewer_start, self.top_viewer_stop, self.top_viewer_home):
            b.setStyleSheet(secondary_css)
            top.addWidget(b)

        top.addSpacing(16)
        top.addWidget(QLabel("每日目标(小时):"))
        self.target_spin = QSpinBox()
        self.target_spin.setRange(0, 100000)
        self.target_spin.setValue(10)
        self.target_spin.valueChanged.connect(self._refresh_kpis)
        top.addWidget(self.target_spin)
        top.addStretch()
        self.btn_account = QPushButton("切换账号")
        self.btn_account.setStyleSheet(secondary_css)
        self.btn_account.clicked.connect(self.on_switch_account)
        top.addWidget(self.btn_account)
        # Login / visibility indicator — surfaces token & org-permission problems
        # (e.g. "未登录(匿名) · TacVerse 可见 11 个") without any digging.
        self.identity_label = QLabel("登录状态: 检测中…")
        self.identity_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.identity_label.setStyleSheet("color:#888;")
        top.addWidget(self.identity_label)
        root.addLayout(top)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_dashboard_tab(), "看板")
        self.tabs.addTab(self._build_trends_tab(), "趋势")
        self.tabs.addTab(self._build_rollup_tab(), "分组统计")
        self.tabs.addTab(self._build_viewer_tab(), "Viewer")
        root.addWidget(self.tabs, 1)

        # Progress: status line + (bar + speed)
        self.status = QLabel("就绪")
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        root.addWidget(self.status)
        prog_row = QHBoxLayout()
        self.bar = QProgressBar()
        self.bar.setValue(0)
        prog_row.addWidget(self.bar, 1)
        self.speed_label = QLabel("—")
        self.speed_label.setMinimumWidth(90)
        self.speed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        prog_row.addWidget(self.speed_label)
        root.addLayout(prog_row)

    def _build_dashboard_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)

        # KPI cards
        self.kpi_labels = {}
        cards = QHBoxLayout()
        for key, title in [
            ("total_datasets", "数据集总数"), ("total_hours", "总小时数"),
            ("total_episodes", "总 episodes"),
            ("new_hours", "今日新增小时"), ("new_episodes", "今日新增episodes"),
            ("completion", "目标完成度"),
        ]:
            cards.addWidget(self._make_card(key, title))
        cards.addWidget(self._make_mvp_card())
        v.addLayout(cards)

        # Which earlier pull the "今日新增" figures are measured against.
        self.baseline_hint = QLabel("")
        self.baseline_hint.setStyleSheet("color: #888; font-size: 12px;")
        v.addWidget(self.baseline_hint)

        # Filter box
        filt = QHBoxLayout()
        filt.addWidget(QLabel("筛选:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("按 名称 / robot_type / 上传者 过滤…")
        self.filter_edit.textChanged.connect(self._apply_filter)
        filt.addWidget(self.filter_edit)
        self.only_issues = QCheckBox("只看有问题的")
        self.only_issues.toggled.connect(self._apply_filter)
        filt.addWidget(self.only_issues)
        v.addLayout(filt)

        # Table
        self.table = QTableWidget(0, len(TABLE_COLS))
        self.table.setHorizontalHeaderLabels([c[0] for c in TABLE_COLS])
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.cellDoubleClicked.connect(self._open_row_link)
        self.table.itemSelectionChanged.connect(self._on_dataset_selected)
        hdr = self.table.horizontalHeader()
        # Dataset name: wide, user-resizable (Stretch left it too narrow to show
        # long ids). The last column stretches to absorb any trailing slack.
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        for i in range(1, len(TABLE_COLS)):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hdr.setStretchLastSection(True)
        self.table.setColumnWidth(0, 440)

        # Master-detail: table (left) + language-annotation Prompt panel (right).
        split = QSplitter(Qt.Horizontal)
        split.addWidget(self.table)
        split.addWidget(self._build_prompt_panel())
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 1)
        split.setCollapsible(1, True)
        split.setSizes([1180, 400])
        v.addWidget(split, 1)

        self.table_hint = QLabel("点「统计当前数据集」加载数据集列表(双击行打开 HF 页面)。")
        v.addWidget(self.table_hint)
        return w

    def _build_prompt_panel(self):
        """Right-side read-only Prompt panel for the selected dataset.

        Two stacked sections, both read straight from the pulled files:
          1. 任务指令 (task)  — meta/tasks.parquet, native to every dataset.
          2. 语言标注 (viewer) — meta/lerobot_annotations.json, only if authored.
        A single centered label covers the "nothing selected / not pulled" case."""
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(8, 0, 0, 0)

        title_row = QHBoxLayout()
        title = QLabel("📋 ANNOTATIONS 标注")
        title.setStyleSheet("font-weight: bold;")
        title_row.addWidget(title)
        title_row.addStretch()
        self.open_viewer_btn = QPushButton("🔍 在 Viewer 打开")
        self.open_viewer_btn.setToolTip("在浏览器的 Viewer 里打开选中的数据集（需先启动 Viewer）")
        self.open_viewer_btn.clicked.connect(self._open_selected_in_viewer)
        title_row.addWidget(self.open_viewer_btn)
        pv.addLayout(title_row)

        self.prompt_meta = QLabel("")
        self.prompt_meta.setStyleSheet("color: #888; font-size: 12px;")
        self.prompt_meta.setWordWrap(True)
        pv.addWidget(self.prompt_meta)

        # --- Section 1: native task instructions (meta/tasks.parquet) --------
        self.task_box = QWidget()
        tb = QVBoxLayout(self.task_box)
        tb.setContentsMargins(0, 4, 0, 0)
        task_hd = QLabel("Language Instruction")
        task_hd.setStyleSheet("font-weight: bold; color: #555;")
        tb.addWidget(task_hd)
        self.task_list = QListWidget()
        self.task_list.setWordWrap(True)
        self.task_list.setMaximumHeight(140)
        tb.addWidget(self.task_list)
        self.task_note = QLabel("")
        self.task_note.setStyleSheet("color: #999; font-size: 12px;")
        self.task_note.setWordWrap(True)
        tb.addWidget(self.task_note)
        pv.addWidget(self.task_box)

        # --- Section 1.5: viewer /report analysis (key info without WebUI) ---
        self.report_box = QWidget()
        rb = QVBoxLayout(self.report_box)
        rb.setContentsMargins(0, 4, 0, 0)
        report_hd = QLabel("Viewer 分析 (report)")
        report_hd.setStyleSheet("font-weight: bold; color: #555;")
        rb.addWidget(report_hd)
        # Indeterminate marquee shown while the background analysis runs, so it
        # reads as "working" rather than "frozen".
        self.report_progress = QProgressBar()
        self.report_progress.setRange(0, 0)  # 0..0 = animated indeterminate
        self.report_progress.setTextVisible(False)
        self.report_progress.setMaximumHeight(6)
        self.report_progress.setVisible(False)
        rb.addWidget(self.report_progress)
        self.report_tree = QTreeWidget()
        self.report_tree.setHeaderHidden(True)
        self.report_tree.setWordWrap(True)
        self.report_tree.setRootIsDecorated(False)
        self.report_tree.setMaximumHeight(240)
        rb.addWidget(self.report_tree)
        self.report_note = QLabel("")
        self.report_note.setStyleSheet("color: #999; font-size: 12px;")
        self.report_note.setWordWrap(True)
        rb.addWidget(self.report_note)
        pv.addWidget(self.report_box)

        # --- Section 2: viewer language annotations -------------------------
        self.anno_box = QWidget()
        ab = QVBoxLayout(self.anno_box)
        ab.setContentsMargins(0, 4, 0, 0)
        anno_hd = QLabel("语言标注 (viewer)")
        anno_hd.setStyleSheet("font-weight: bold; color: #555;")
        ab.addWidget(anno_hd)

        ep_row = QHBoxLayout()
        ep_row.addWidget(QLabel("集:"))
        self.prompt_ep = QComboBox()
        self.prompt_ep.currentIndexChanged.connect(self._refresh_prompt_tree)
        ep_row.addWidget(self.prompt_ep, 1)
        self.prompt_ep_wrap = QWidget()
        self.prompt_ep_wrap.setLayout(ep_row)
        ab.addWidget(self.prompt_ep_wrap)

        self.prompt_tree = QTreeWidget()
        self.prompt_tree.setHeaderHidden(True)
        self.prompt_tree.setWordWrap(True)
        self.prompt_tree.setRootIsDecorated(True)
        ab.addWidget(self.prompt_tree, 1)

        self.anno_note = QLabel("")
        self.anno_note.setStyleSheet("color: #999; font-size: 12px;")
        self.anno_note.setWordWrap(True)
        ab.addWidget(self.anno_note)
        pv.addWidget(self.anno_box, 1)

        # --- Section 3: quality checks (grouped by provider) ----------------
        self.check_box = QWidget()
        cb = QVBoxLayout(self.check_box)
        cb.setContentsMargins(0, 4, 0, 0)
        check_hd = QLabel("检查")
        check_hd.setStyleSheet("font-weight: bold; color: #555;")
        cb.addWidget(check_hd)
        self.check_tree = QTreeWidget()
        self.check_tree.setHeaderHidden(True)
        self.check_tree.setWordWrap(True)
        self.check_tree.setRootIsDecorated(True)
        cb.addWidget(self.check_tree, 1)
        pv.addWidget(self.check_box, 1)

        # --- Fallback: nothing selected -------------------------------------
        self.prompt_empty = QLabel("选择左侧数据集查看 Prompt。")
        self.prompt_empty.setStyleSheet("color: #999;")
        self.prompt_empty.setWordWrap(True)
        self.prompt_empty.setAlignment(Qt.AlignCenter)
        pv.addWidget(self.prompt_empty, 1)

        self._prompt_doc = {"episodes": {}, "updated_at": None}
        self._show_prompt_empty("选择左侧数据集查看 Prompt。")
        return panel

    def _make_card(self, key, title):
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        cv = QVBoxLayout(card)
        t = QLabel(title)
        t.setStyleSheet("color: #666; font-size: 12px;")
        val = QLabel("—")
        val.setStyleSheet("font-size: 22px; font-weight: bold;")
        val.setTextInteractionFlags(Qt.TextSelectableByMouse)
        cv.addWidget(t)
        cv.addWidget(val)
        self.kpi_labels[key] = val
        return card

    def _make_mvp_card(self):
        """Special card: today's top contributor (by new hours) + their tallies."""
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        cv = QVBoxLayout(card)
        t = QLabel("今日 MVP ⭐")
        t.setStyleSheet("color: #666; font-size: 12px;")
        self.mvp_name_lbl = QLabel("—")
        self.mvp_name_lbl.setStyleSheet(
            "font-size: 22px; font-weight: bold; color:#F9A825;")
        self.mvp_name_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.mvp_sub_lbl = QLabel("")
        self.mvp_sub_lbl.setStyleSheet("color:#888; font-size: 11px;")
        cv.addWidget(t)
        cv.addWidget(self.mvp_name_lbl)
        cv.addWidget(self.mvp_sub_lbl)
        return card

    def _build_trends_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.trend_hint = QLabel("")
        v.addWidget(self.trend_hint)
        self.daily_plot = pg.PlotWidget(title="每日新增小时数")
        self.daily_plot.showGrid(x=False, y=True, alpha=0.3)
        v.addWidget(self.daily_plot)
        self.cum_plot = pg.PlotWidget(title="累计小时数")
        self.cum_plot.showGrid(x=False, y=True, alpha=0.3)
        v.addWidget(self.cum_plot)
        return w

    def _build_rollup_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        row = QHBoxLayout()
        row.addWidget(QLabel("分组维度:"))
        self.dim_combo = QComboBox()
        self.dim_combo.addItems(list(ROLLUP_DIMS.keys()))
        self.dim_combo.currentTextChanged.connect(self._refresh_rollup)
        row.addWidget(self.dim_combo)
        row.addStretch()
        v.addLayout(row)

        self.rollup_table = QTableWidget(0, 5)
        self.rollup_table.setHorizontalHeaderLabels(
            ["分组", "数据集数", "episodes", "小时", "占比%"])
        self.rollup_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.rollup_table.verticalHeader().setVisible(False)
        self.rollup_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        v.addWidget(self.rollup_table)
        self.rollup_plot = pg.PlotWidget(title="各分组小时数")
        self.rollup_plot.showGrid(x=False, y=True, alpha=0.3)
        v.addWidget(self.rollup_plot)
        return w

    # ---- Viewer tab (vendored xense_lerobot_viewer, black-box service) ---- #
    def _viewer_root(self):
        """The dataset root the viewer scans (contract ①): the latest pull-date
        folder under pulls/ (so it shows the most recent pull, without the
        per-date duplicates you'd get by pointing at pulls/ itself). Falls back
        to pulls/ when there are no date folders yet."""
        base = Path(OUT_DIR)
        dates = sorted((p for p in base.glob("*")
                        if p.is_dir() and p.name.isdigit()),
                       key=lambda p: p.name)
        return str((dates[-1] if dates else base).resolve())

    def _build_viewer_tab(self):
        """Reserved space for the viewer: service status + controls.

        The viewer serves ALL its features over the web; this tab drives its
        lifecycle and opens it in the browser. The placeholder area is kept so
        a future phase can drop an embedded web view in without restructuring.
        """
        w = QWidget()
        v = QVBoxLayout(w)

        self.viewer_status = QLabel("")
        self.viewer_status.setStyleSheet("font-size: 15px;")
        v.addWidget(self.viewer_status)
        self.viewer_detail = QLabel("")
        self.viewer_detail.setStyleSheet("color: #888; font-size: 12px;")
        self.viewer_detail.setWordWrap(True)
        v.addWidget(self.viewer_detail)

        row = QHBoxLayout()
        self.viewer_start_btn = QPushButton("启动 Viewer")
        self.viewer_start_btn.clicked.connect(self._viewer_start)
        self.viewer_stop_btn = QPushButton("停止")
        self.viewer_stop_btn.clicked.connect(self._viewer_stop)
        self.viewer_home_btn = QPushButton("打开首页")
        self.viewer_home_btn.clicked.connect(self._viewer_open_home)
        for b in (self.viewer_start_btn, self.viewer_stop_btn, self.viewer_home_btn):
            row.addWidget(b)
        row.addStretch()
        v.addLayout(row)

        self.viewer_placeholder = QLabel(
            "Viewer 以网页形式提供全部功能（数据集预览 / 健康检查 / 3D 回放 / 标注）。\n"
            "点「启动 Viewer」后，用「打开首页」，或在「看板」选中数据集点「🔍 在 Viewer 打开」。\n\n"
            "（此区域为预留：后续可在此内嵌网页视图）")
        self.viewer_placeholder.setAlignment(Qt.AlignCenter)
        self.viewer_placeholder.setWordWrap(True)
        self.viewer_placeholder.setStyleSheet(
            "color: #aaa; border: 1px dashed #ccc; padding: 24px;")
        v.addWidget(self.viewer_placeholder, 1)

        self._viewer_tick = 0
        self._viewer_count = None
        self.viewer_timer = QTimer(self)
        self.viewer_timer.timeout.connect(self._refresh_viewer_status)
        self.viewer_timer.start(2000)
        self._refresh_viewer_status()
        return w

    def _viewer_start(self):
        if not self.viewer.available():
            msg = f"viewer 未就绪：请在 {self.viewer.viewer_dir} 执行 bun install"
            self.viewer_detail.setText(msg)
            self.status.setText(msg)
            return
        ok, msg = self.viewer.start(self._viewer_root(), wait=False)
        self.status.setText(f"Viewer: {msg}")
        self._viewer_count = None
        self._refresh_viewer_status()

    def _viewer_stop(self):
        self.viewer.stop()
        self._viewer_count = None
        self.status.setText("Viewer 已停止")
        self._refresh_viewer_status()

    def _viewer_open_home(self):
        if not self.viewer.is_running():
            self.status.setText("Viewer 未启动：请先点「启动 Viewer」")
            return
        self.viewer.open_home()
        self.status.setText(f"已打开首页: {self.viewer.home_url()}")

    def _refresh_viewer_status(self):
        st = self.viewer.status()
        if not st["running"]:
            color, text = "#c62828", "未启动"
        elif st["ready"]:
            color, text = "#2e7d32", "运行中"
        else:
            color, text = "#F9A825", "启动中…"

        # Refresh the dataset count occasionally (every ~6s) to avoid hammering
        # the discovery API on every tick.
        self._viewer_tick += 1
        if st["ready"] and self._viewer_tick % 3 == 0:
            self._viewer_count = self.viewer.dataset_count()
        elif not st["ready"]:
            self._viewer_count = None
        extra = f" · 可见数据集 {self._viewer_count}" if self._viewer_count is not None else ""

        # Toolbar controls (canonical).
        self.top_viewer_dot.setText(
            f'<span style="color:{color}">●</span> Viewer: {text} · {st["port"]}')
        self.top_viewer_start.setEnabled(not st["running"])
        self.top_viewer_stop.setEnabled(st["managed"])
        self.top_viewer_home.setEnabled(st["ready"])

        # Keep the (soon-to-be-optional) Viewer tab in sync if it still exists.
        if hasattr(self, "viewer_status"):
            self.viewer_status.setText(
                f'<span style="color:{color}">●</span> Viewer: {text} · 端口 {st["port"]}')
            self.viewer_detail.setText(
                f'数据根: {st["root"] or self._viewer_root()}{extra}   ({st["url"]})')
            self.viewer_start_btn.setEnabled(not st["running"])
            self.viewer_stop_btn.setEnabled(st["managed"])
            self.viewer_home_btn.setEnabled(st["ready"])

    def _open_selected_in_viewer(self):
        d = self._selected_dataset()
        if not d:
            self.status.setText("请先在左侧选中一个数据集")
            return
        if not self.viewer.is_running():
            self.status.setText("Viewer 未启动：请到「Viewer」页点「启动 Viewer」")
            return
        rel = self.viewer.dataset_rel_path(d, root=self._viewer_root())
        if not rel:
            self.status.setText(
                f"该数据集不在数据根下（未拉取到 {OUT_DIR}/），无法在 Viewer 打开")
            return
        self.viewer.open_dataset(rel)
        self.status.setText(f"已在浏览器打开: {self.viewer.dataset_url(rel)}")

    # ---- Rendering -------------------------------------------------------- #
    def _refresh_all(self):
        self._refresh_kpis()
        self._refresh_table()
        self._refresh_trends()
        self._refresh_rollup()

    def _current_deltas(self):
        if not self.report:
            return {}
        return dd.compute_deltas(self.report, self.history)

    def _refresh_baseline_hint(self):
        """Spell out which earlier pull the「今日新增」figures are compared against."""
        base = dd.find_baseline(self.report, self.history) if self.report else None
        if not base:
            self.baseline_hint.setText(
                "「今日新增」暂无历史基准 —— 这是首次拉取，下方增量即为全部总量。")
            return
        day = fmt_day(base.get("date"))
        gap = days_between(base.get("date"), (self.report or {}).get("date"))
        ago = f"（{gap} 天前）" if gap else ""
        self.baseline_hint.setText(
            f"「今日新增」= 相较于 {day}{ago} 最近一次拉取结果的增量。")

    def _refresh_kpis(self):
        r = self.report
        if not r:
            for lbl in self.kpi_labels.values():
                lbl.setText("—")
            self.baseline_hint.setText("")
            self.mvp_name_lbl.setText("—")
            self.mvp_sub_lbl.setText("")
            return
        self._refresh_baseline_hint()
        deltas = self._current_deltas()
        self._refresh_mvp(deltas)
        new_hours = round(sum(d["d_hours"] for d in deltas.values()), 2)
        new_eps = sum(d["d_episodes"] for d in deltas.values())
        target = self.target_spin.value()
        pct = f"{round(100 * new_hours / target)}%" if target else "—"
        self.kpi_labels["total_datasets"].setText(fmt_value(r.get("total_datasets")))
        self.kpi_labels["total_hours"].setText(fmt_value(r.get("total_hours")))
        self.kpi_labels["total_episodes"].setText(fmt_value(r.get("total_episodes")))
        self.kpi_labels["new_hours"].setText(f"+{new_hours}")
        self.kpi_labels["new_episodes"].setText(f"+{fmt_value(new_eps)}")
        self.kpi_labels["completion"].setText(pct)

    def _refresh_mvp(self, deltas):
        """Today's MVP = the person whose datasets added the most new hours today.

        Attributes each dataset's 今日新增 (delta) to its uploader's Chinese name,
        then picks the top by hours. Shows their hours + episodes underneath.
        """
        by_person = {}
        for d in (self.report.get("datasets", []) if self.report else []):
            dv = deltas.get(d["dataset_name"], {})
            agg = by_person.setdefault(uploader_cn(d.get("uploader")),
                                       {"hours": 0.0, "eps": 0})
            agg["hours"] += dv.get("d_hours", 0) or 0
            agg["eps"] += dv.get("d_episodes", 0) or 0
        top = max(by_person.items(), key=lambda kv: kv[1]["hours"], default=None)
        if not top or top[1]["hours"] <= 0:
            self.mvp_name_lbl.setText("—")
            self.mvp_sub_lbl.setText("今日暂无新增贡献")
            return
        name, agg = top
        self.mvp_name_lbl.setText(name)
        self.mvp_sub_lbl.setText(
            f"{round(agg['hours'], 2)} 小时 · {fmt_value(agg['eps'])} episodes")

    def _downloaded_leaves(self):
        """Leaf names of datasets whose raw files are downloaded under pulls/.

        A dataset counts as downloaded when some pulls/<date>/<leaf>/meta/info.json
        exists (a full 拉取 writes it; 统计-only never touches pulls/). Only these
        can be opened in the viewer. Scanned once per table refresh."""
        return {info.parent.parent.name
                for info in Path(OUT_DIR).glob("*/*/meta/info.json")}

    def _refresh_table(self):
        r = self.report
        datasets = r.get("datasets", []) if r else []
        deltas = self._current_deltas()
        downloaded = self._downloaded_leaves()  # dataset leaf names present in pulls/
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(datasets))
        for row, d in enumerate(datasets):
            for col, (_, key, kind) in enumerate(TABLE_COLS):
                if key == "__local__":
                    leaf = (d.get("dataset_name") or "").split("/")[-1]
                    dl = leaf in downloaded
                    item = NumericItem("✅ 已下载" if dl else "—", 1 if dl else 0)
                    item.setToolTip(
                        "原始文件已下载到本地 pulls/，可在 Viewer 打开" if dl else
                        "未下载（仅统计信息）；先「拉取」才能在 Viewer 打开")
                    if dl:
                        item.setForeground(QBrush(QColor("#2e7d32")))
                elif key == "__delta__":
                    dv = deltas.get(d["dataset_name"], {})
                    n = dv.get("d_episodes", 0)
                    if dv.get("is_new"):
                        txt, color = f"🆕 +{n}", "#1565C0"   # newly created dataset
                    elif n > 0:
                        txt, color = f"⬆ +{n}", "#2e7d32"    # grew vs previous pull day
                    elif n < 0:
                        txt, color = f"⬇ {n}", "#c62828"     # shrank vs previous
                    else:
                        txt, color = "➖ 0", "#9e9e9e"        # unchanged (持平)
                    item = NumericItem(txt, n)
                    item.setForeground(QBrush(QColor(color)))
                elif key == "__avg_sec__":
                    eps = d.get("total_episodes") or 0
                    hrs = d.get("duration_hours") or 0
                    v = round(hrs * 3600 / eps, 1) if eps else 0
                    item = NumericItem(fmt_value(v), v)
                elif key == "__check__":
                    results, agg = chk_mod.run_checks(d, cfg=_CHECKS_CFG)
                    txt, sort_key = chk_mod.badge(agg)
                    item = NumericItem(txt, sort_key)
                    item.setToolTip("\n".join(
                        f"{chk_mod.icon(x.status)} {x.title}: {x.message}"
                        for x in results))
                elif key == "__uploader_cn__":
                    item = QTableWidgetItem(uploader_cn(d.get("uploader")))
                elif kind == "num":
                    v = d.get(key)
                    item = NumericItem(fmt_value(v), v if isinstance(v, (int, float)) else -1)
                elif kind == "date":
                    v = d.get(key) or ""
                    # Show day granularity but sort by the full ISO timestamp
                    # (ISO strings sort chronologically), so the default 最后更新↓
                    # order reproduces HF's "Recently updated" ranking — same-day
                    # datasets keep their real order instead of shuffling.
                    item = NumericItem(v[:10] if v else "—", v or "")
                else:
                    item = QTableWidgetItem(fmt_value(d.get(key)))
                if col == 0:
                    item.setData(Qt.UserRole, d)  # stash the row's dataset dict
                self.table.setItem(row, col, item)
        self.table.setSortingEnabled(True)
        # Default order: most-recently-updated first (matches org page / 发现顺序).
        self.table.sortItems(DATE_COL, Qt.DescendingOrder)
        self.table_hint.setText(
            f"共 {len(datasets)} 个数据集，双击行打开 HF 页面；点表头排序。"
            if datasets else "点「统计当前数据集」加载数据集列表。")
        self._apply_filter()

    def _apply_filter(self):
        q = self.filter_edit.text().strip().lower()
        only_issues = self.only_issues.isChecked()
        for row in range(self.table.rowCount()):
            d = self.table.item(row, 0).data(Qt.UserRole) or {}
            hay = " ".join(str(d.get(k, "")) for k in
                           ("dataset_name", "robot_type", "uploader")).lower()
            hay += " " + uploader_cn(d.get("uploader")).lower()
            hide = bool(q) and q not in hay
            if not hide and only_issues:
                _, agg = chk_mod.run_checks(d, cfg=_CHECKS_CFG)
                hide = agg["worst"] == chk_mod.OK
            self.table.setRowHidden(row, hide)

    def _open_row_link(self, row, _col):
        d = self.table.item(row, 0).data(Qt.UserRole) or {}
        link = d.get("link")
        if link:
            QDesktopServices.openUrl(QUrl(link))
            self.status.setText(f"已打开: {link}")

    # ---- Language-annotation Prompt panel (read-only, 方式1 读文件) ----------
    def _selected_dataset(self):
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return item.data(Qt.UserRole) if item else None

    def _show_prompt_empty(self, msg):
        """Show only the centered fallback label (nothing selected)."""
        self.prompt_empty.setText(msg)
        self.prompt_empty.setVisible(True)
        self.task_box.setVisible(False)
        self.report_box.setVisible(False)
        self.anno_box.setVisible(False)
        self.check_box.setVisible(False)

    def _on_dataset_selected(self):
        d = self._selected_dataset()
        if not d:
            self.prompt_meta.setText("")
            self._show_prompt_empty("选择左侧数据集查看 Prompt / 检查。")
            return

        name = (d.get("dataset_name") or "").split("/")[-1]
        # Task text carried inline in the record (fetched during 统计/拉取) is
        # preferred — it means the prompt shows without any local file.
        inline_tasks = d.get("tasks") if isinstance(d.get("tasks"), list) else None
        task_path = tsk.resolve_path(d, OUT_DIR)
        anno_path = ann.resolve_path(d, OUT_DIR)

        # Checks run off the record itself (name / duration / prompt), so the
        # panel is useful for any selected row even before a full pull.
        self.prompt_empty.setVisible(False)
        self.task_box.setVisible(True)
        self.report_box.setVisible(True)
        self.anno_box.setVisible(True)
        self.check_box.setVisible(True)

        n_tasks = self._refresh_tasks(inline_tasks, task_path)
        n_anno_eps, total_eps = self._refresh_annotations(anno_path)
        agg = self._refresh_checks(d)
        self._refresh_report(d)

        bits = [f"数据集: {name}", f"{n_tasks} 条指令"]
        if anno_path:
            bits.append(f"{n_anno_eps}/{total_eps} 集有标注")
        if agg["n_fail"] or agg["n_warn"]:
            bits.append(f"检查 {chk_mod.badge(agg)[0]}")
        self.prompt_meta.setText(" · ".join(bits))

    def _refresh_checks(self, d):
        """Populate the 检查 tree (grouped by provider). Returns the aggregate."""
        self.check_tree.clear()
        results, agg = chk_mod.run_checks(
            d, providers=("custom", "viewer"), cfg=_CHECKS_CFG)
        provider_cn = {"custom": "自定义检查", "viewer": "Viewer 检查"}
        by_provider = {}
        for r in results:
            by_provider.setdefault(r.provider, []).append(r)
        for provider in ("custom", "viewer"):
            group = by_provider.get(provider)
            if not group:
                continue
            parent = QTreeWidgetItem([provider_cn.get(provider, provider)])
            f = parent.font(0)
            f.setBold(True)
            parent.setFont(0, f)
            self.check_tree.addTopLevelItem(parent)
            for r in group:
                line = f"{chk_mod.icon(r.status)} {r.title}: {r.message}"
                node = QTreeWidgetItem([line])
                node.setToolTip(0, line)
                parent.addChild(node)
                for det in r.details:
                    node.addChild(QTreeWidgetItem([det]))
                node.setExpanded(True)
            parent.setExpanded(True)
        return agg

    # ---- Viewer /report analysis (async, key info without the WebUI) ------ #
    def _report_show_note(self, msg, busy=False):
        self.report_tree.setVisible(False)
        self.report_tree.clear()
        self.report_progress.setVisible(busy)  # marquee only while analyzing
        self.report_note.setVisible(True)
        self.report_note.setText(msg)

    def _refresh_report(self, d):
        """Show the viewer's /report analysis for the selected dataset. Fetched
        in a background thread (it can take tens of seconds); cached per
        session; stale selections are ignored via a monotonic seq."""
        self._report_seq += 1
        seq = self._report_seq
        if not self.viewer.is_running():
            self._report_show_note("Viewer 未运行；启动后显示分析（Viewer 页）。")
            return
        rel = self.viewer.dataset_rel_path(d, root=self._viewer_root())
        if not rel:
            self._report_show_note(
                "该数据集不在 Viewer 数据根（最新拉取日），暂无分析。")
            return
        cached = self._report_cache.get(rel)
        if cached is not None:
            self._render_report(cached)
            return
        self._report_show_note("分析中…（首次约 10–30s）", busy=True)
        w = ReportWorker(self.viewer, rel, seq)
        w.done.connect(self._on_report_done)
        self._report_workers.append(w)
        w.start()

    def _on_report_done(self, seq, rel, report, err):
        self._report_workers = [w for w in self._report_workers if w.isRunning()]
        if report is not None:
            self._report_cache[rel] = report
        if seq != self._report_seq:
            return  # user moved to another dataset; ignore stale result
        if report is None:
            self._report_show_note(f"分析失败: {err}")
            return
        self._render_report(report)

    def _render_report(self, r):
        """Render the key /report fields as flat rows in the analysis tree."""
        self.report_progress.setVisible(False)
        self.report_note.setVisible(False)
        self.report_tree.setVisible(True)
        self.report_tree.clear()

        def row(text):
            item = QTreeWidgetItem([text])
            item.setToolTip(0, text)
            self.report_tree.addTopLevelItem(item)

        integ = r.get("integrity") or {}
        issues = integ.get("issues") or []
        row(f"完整性: {integ.get('status', '?')}"
            + (f" · {'; '.join(issues)}" if issues else ""))
        ds = r.get("dataset") or {}
        row(f"摄像头: {len(ds.get('cameras') or [])} · fps {ds.get('fps')}")

        q = r.get("quality") or {}
        el = q.get("episodeLength")
        if el:
            row(f"时长(s): 最短 {el.get('shortest')} · 最长 {el.get('longest')}"
                f" · 均 {el.get('mean')}")
        sm = q.get("smoothness")
        if sm:
            c = sm.get("counts") or {}
            row(f"平滑度: {(sm.get('verdict') or {}).get('label')}"
                f"  (smooth {c.get('smooth', 0)} / jerky {c.get('jerky', 0)})")
        if q:
            row(f"抖动集: {len(q.get('jerkyEpisodes') or [])}"
                f" · 低运动集: {len(q.get('lowMovementEpisodes') or [])}")

        t = r.get("training") or {}
        sc = t.get("suggestedChunkLength")
        if sc:
            row(f"建议 chunk: {sc.get('steps')} 步 ({round(sc.get('seconds', 0), 2)}s)")
        elif "training" in r:
            row("建议 chunk: —")
        cd = t.get("controlDelay")
        if cd:
            row(f"控制延迟: {cd.get('meanSteps')} 步"
                f" ({round(cd.get('seconds', 0), 3)}s)"
                f" {'因果✓' if cd.get('causalOk') else '非因果✗'}")
        sv = t.get("speedVariance")
        if sv:
            tail = " · 需速度归一" if sv.get("needsVelocityNorm") else ""
            row(f"速度方差: {(sv.get('verdict') or {}).get('label')}"
                f" (cv {round(sv.get('cv', 0), 3)}){tail}")

        meta = r.get("meta") or {}
        if meta.get("sampledEpisodes") is not None:
            row(f"（抽样 {meta.get('sampledEpisodes')} 集）")

    def _refresh_tasks(self, inline, path):
        """Fill the task-instruction list. Prefers inline task rows (from the
        stats/pull record); falls back to reading a local tasks.parquet.
        Returns the task count."""
        self.task_list.clear()

        def note(msg):
            self.task_list.setVisible(False)
            self.task_note.setVisible(True)
            self.task_note.setText(msg)

        if inline:
            tasks = inline
        elif path:
            tasks, err = tsk.load(path)
            if err:
                note(err)
                return 0
        else:
            note("无 Language Instruction(该数据集未提供 tasks.parquet)。")
            return 0

        if not tasks:
            note("无 Language Instruction(该数据集未提供 tasks.parquet)。")
            return 0

        for row in tasks:
            item = QListWidgetItem(f"[{row['index']}] {row['task']}")
            item.setToolTip(row["task"])
            self.task_list.addItem(item)
        self.task_list.setVisible(True)
        self.task_note.setVisible(False)
        return len(tasks)

    def _refresh_annotations(self, path):
        """Fill the viewer-annotation tree. Returns (annotated_eps, total_eps)."""
        doc, err = ann.load(path) if path else ({"episodes": {}}, None)
        self._prompt_doc = doc
        eps = ann.episodes_with_atoms(doc)
        total_eps = len(doc.get("episodes", {}))

        def note(msg):
            self.prompt_ep_wrap.setVisible(False)
            self.prompt_tree.setVisible(False)
            self.anno_note.setVisible(True)
            self.anno_note.setText(msg)

        if not path:
            note("暂无 viewer 语言标注(可在 viewer 中编辑生成)。")
            return 0, 0
        if err:
            note(err)
            return 0, total_eps
        if not eps:
            note("暂无 viewer 语言标注(可在 viewer 中编辑生成)。")
            return 0, total_eps

        self.prompt_ep.blockSignals(True)
        self.prompt_ep.clear()
        for ep in eps:
            self.prompt_ep.addItem(f"ep {ep}", ep)
        self.prompt_ep.setCurrentIndex(0)
        self.prompt_ep.blockSignals(False)

        self.anno_note.setVisible(False)
        self.prompt_ep_wrap.setVisible(True)
        self.prompt_tree.setVisible(True)
        self._refresh_prompt_tree()
        return len(eps), total_eps

    def _refresh_prompt_tree(self):
        self.prompt_tree.clear()
        ep = self.prompt_ep.currentData()
        if ep is None:
            return
        atoms = ann.atoms_for_episode(self._prompt_doc, ep)
        for style, group in ann.group_by_style(atoms):
            parent = QTreeWidgetItem([f"{ann.style_label(style)} ({len(group)})"])
            f = parent.font(0)
            f.setBold(True)
            parent.setFont(0, f)
            self.prompt_tree.addTopLevelItem(parent)
            for atom in group:
                text = ann.atom_text(atom)
                if ann.is_event_style(style):
                    ts = atom.get("timestamp")
                    if isinstance(ts, (int, float)):
                        text = f"{ts:.1f}s  {text}"
                child = QTreeWidgetItem([text])
                child.setToolTip(0, text)
                parent.addChild(child)
            parent.setExpanded(True)

    def _refresh_trends(self):
        series = dd.daily_series(self.history)
        self.daily_plot.clear()
        self.cum_plot.clear()
        if not series:
            self.trend_hint.setText("暂无历史数据。执行「拉取当前数据集」后按天积累趋势。")
            return
        self.trend_hint.setText(
            "" if len(series) >= 2 else "当前仅 1 天数据，多日拉取后可见增长趋势。")
        # Categorical x = only days that were actually pulled, packed side by side
        # (未统计的日期不占位，不留空白). fmt_day makes labels read 07-03 not 260703.
        x = list(range(len(series)))
        labels = [fmt_day(s["date"])[5:] for s in series]  # MM-DD
        ticks = [list(zip(x, labels))]
        bg = pg.BarGraphItem(x=x, height=[s["total_hours"] for s in series],
                             width=0.8, brush="#4C8BF5")
        self.daily_plot.addItem(bg)
        self.daily_plot.getAxis("bottom").setTicks(ticks)
        self.daily_plot.setXRange(-0.5, len(series) - 0.5, padding=0)
        self.cum_plot.plot(x, [s["cum_hours"] for s in series],
                           pen=pg.mkPen("#34A853", width=2), symbol="o",
                           symbolBrush="#34A853")
        self.cum_plot.getAxis("bottom").setTicks(ticks)
        self.cum_plot.setXRange(-0.5, len(series) - 0.5, padding=0.02)

    def _refresh_rollup(self):
        self.rollup_table.setRowCount(0)
        self.rollup_plot.clear()
        if not self.report:
            return
        dim = self.dim_combo.currentText()
        rows = dd.rollup(self.report.get("datasets", []), ROLLUP_DIMS[dim])
        self.rollup_table.setRowCount(len(rows))
        for i, g in enumerate(rows):
            vals = [g["group"], g["count"], g["episodes"], g["hours"], g["pct_hours"]]
            for j, v in enumerate(vals):
                if j == 0:
                    item = QTableWidgetItem(str(v))
                else:
                    item = NumericItem(fmt_value(v), v)
                self.rollup_table.setItem(i, j, item)
        # Horizontal bars: one row per group so the labels (中文名 / 任务名) read
        # left-to-right and never overlap, however many groups there are. Cap the
        # chart to the top 20 by hours (the table above still lists them all).
        plot_rows = rows[:20]
        n = len(plot_rows)
        ys = [n - 1 - i for i in range(n)]  # rows are hours-desc -> largest on top
        bg = pg.BarGraphItem(x0=0, y=ys, height=0.7,
                             width=[g["hours"] for g in plot_rows], brush="#F9A825")
        self.rollup_plot.addItem(bg)

        def _short(s, k=24):
            s = str(s)
            return s if len(s) <= k else s[:k - 1] + "…"

        labels = [_short(g["group"]) for g in plot_rows]
        left = self.rollup_plot.getAxis("left")
        left.setTicks([[(ys[i], labels[i]) for i in range(n)]])
        # Widen the y-axis to the longest label so nothing is clipped; CJK glyphs
        # take ~2x the width of a latin char, so weight them double when sizing.
        vis = max((sum(2 if ord(c) > 0x2E80 else 1 for c in s) for s in labels),
                  default=8)
        left.setWidth(min(300, max(70, 12 + vis * 8)))
        self.rollup_plot.getAxis("bottom").setTicks(None)  # auto numeric hour scale
        self.rollup_plot.setYRange(-0.5, n - 0.5, padding=0.02)
        max_h = max((g["hours"] for g in plot_rows), default=1) or 1
        self.rollup_plot.setXRange(0, max_h, padding=0.05)  # bars start at 0, no left gap
        self.rollup_plot.setTitle(
            f"各分组小时数（前 {n}/{len(rows)}）" if len(rows) > n else "各分组小时数")

    # ---- Login / visibility indicator ------------------------------------- #
    def on_switch_account(self):
        """Prompt for an account label + HF token, apply it, and re-check identity.

        The token is what actually authenticates; the account field is just a
        note (the real login name is confirmed by whoami in the indicator). The
        token is kept in memory for this session only — it is never written to
        disk. For a persistent login use `huggingface-cli login` or $HF_TOKEN.
        """
        from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout

        dlg = QDialog(self)
        dlg.setWindowTitle("切换账号 / Token")
        dlg.setMinimumWidth(440)
        form = QFormLayout(dlg)

        acc_edit = QLineEdit()
        acc_edit.setPlaceholderText("可留空，登录后会自动从 token 识别真实账号")
        tok_edit = QLineEdit()
        tok_edit.setPlaceholderText("hf_… 粘贴 HF access token")
        tok_edit.setEchoMode(QLineEdit.Password)
        show_btn = QPushButton("显示")
        show_btn.setCheckable(True)
        show_btn.setFixedWidth(48)
        show_btn.toggled.connect(
            lambda on: tok_edit.setEchoMode(
                QLineEdit.Normal if on else QLineEdit.Password))
        tok_row = QHBoxLayout()
        tok_row.setContentsMargins(0, 0, 0, 0)
        tok_row.addWidget(tok_edit, 1)
        tok_row.addWidget(show_btn)
        tok_wrap = QWidget()
        tok_wrap.setLayout(tok_row)

        form.addRow("账号(选填):", acc_edit)
        form.addRow("Token:", tok_wrap)
        hint = QLabel("Token 会保存到本地 .hf_token（已被 git 忽略，不会上传或"
                      "同步给他人），下次启动自动使用。清除请删除该文件。")
        hint.setStyleSheet("color:#888; font-size:12px;")
        hint.setWordWrap(True)
        form.addRow(hint)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        form.addRow(bb)

        if dlg.exec() != QDialog.Accepted:
            return
        token = tok_edit.text().strip()
        if not token:
            QMessageBox.warning(self, "提示", "Token 不能为空。")
            return
        self.token = token
        save_token(token)  # persist locally (gitignored) for next runs
        acc = acc_edit.text().strip()
        self.status.setText(
            f"已应用并保存 Token{'（'+acc+'）' if acc else ''}，正在校验身份与可见数量 ...")
        self._refresh_identity()

    def _refresh_identity(self, *_):
        """Kick off a background check of who we are + how many datasets we see."""
        org = self.org_combo.currentText().strip()
        if not org:
            return
        self.identity_label.setText("登录状态: 检测中…")
        self.identity_label.setStyleSheet("color:#888;")
        self._id_seq += 1
        seq = self._id_seq
        w = IdentityWorker(org, self.token)
        w.done.connect(lambda name, has, o, cnt, seq=seq:
                       self._on_identity(seq, name, has, o, cnt))
        w.finished.connect(lambda w=w: self._id_workers.remove(w)
                           if w in self._id_workers else None)
        self._id_workers.append(w)  # hold a ref so the QThread isn't GC'd mid-run
        w.start()

    def _on_identity(self, seq, name, has_token, org, count):
        # Only the most recent check may update the label — a slower older worker
        # (e.g. the startup one) must not clobber a fresh account-switch result.
        if seq != self._id_seq:
            return
        cnt = f"可见 {count} 个数据集" if count >= 0 else "数据集数查询失败"
        if not has_token:
            who, color = "未登录(匿名)", "#F9A825"
        elif name:
            who, color = f"已登录: {name}", "#34A853"
        else:
            who, color = "已登录: token 无效/过期", "#EA4335"
        self.identity_label.setText(f"{who} · {org} {cnt}")
        self.identity_label.setStyleSheet(f"color:{color}; font-weight:bold;")

    def closeEvent(self, event):
        # Stop the viewer subprocess we launched so it doesn't outlive the app.
        try:
            self.viewer.stop()
        except Exception:
            pass
        # Let any in-flight identity checks finish so the QThread isn't destroyed
        # mid-run (Qt would otherwise warn / crash on close during a check).
        for w in list(self._id_workers):
            w.wait(2000)
        for w in list(self._report_workers):
            w.wait(2000)
        super().closeEvent(event)

    # ---- Button handlers -------------------------------------------------- #
    def _set_busy(self, busy):
        for b in (self.btn_pull, self.btn_stats, self.btn_check, self.btn_open):
            b.setEnabled(not busy)

    def on_pull(self):
        org = self.org_combo.currentText().strip()
        if not org:
            QMessageBox.warning(self, "提示", "请填写组织名。")
            return
        self._set_busy(True)
        self.bar.setValue(0)
        self.status.setText(f"开始拉取 {org} ...")
        self._watch_dir = Path(OUT_DIR) / dt.datetime.now().strftime("%y%m%d")
        self._prev_bytes = dir_size(self._watch_dir)
        self._prev_t = time.monotonic()
        self.speed_label.setText("0.0 B/s")
        self.speed_timer.start()
        self.worker = PullWorker(org, OUT_DIR, self.token)
        self.worker.log.connect(self.status.setText)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_pull_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def on_stats(self):
        org = self.org_combo.currentText().strip()
        if not org:
            QMessageBox.warning(self, "提示", "请填写组织名。")
            return
        self._set_busy(True)
        self.bar.setValue(0)
        self.status.setText(f"开始统计 {org}（仅读取信息，不下载）...")
        self.worker = StatsWorker(org, self.token)
        self.worker.log.connect(self.status.setText)
        self.worker.progress.connect(self._on_progress)
        self.worker.done.connect(self._on_stats_done)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_progress(self, done, total):
        self.bar.setMaximum(max(total, 1))
        self.bar.setValue(done)

    def _tick_speed(self):
        now = time.monotonic()
        cur = dir_size(self._watch_dir)
        elapsed = now - (self._prev_t or now)
        if elapsed > 0:
            self.speed_label.setText(fmt_speed((cur - self._prev_bytes) / elapsed))
        self._prev_bytes = cur
        self._prev_t = now

    def _stop_speed(self):
        self.speed_timer.stop()
        self.speed_label.setText("—")

    def _on_pull_done(self, report, out_path):
        self._stop_speed()
        self.report = report
        self.history = dd.load_history(OUT_DIR)  # new snapshot just written
        self._refresh_all()
        self._set_busy(False)
        fails = len(report.get("failures", []))
        msg = f"拉取完成: {report['count']}/{report['requested']} 个数据集"
        if fails:
            msg += f"，{fails} 个失败"
        self.status.setText(msg + (f"  ->  {out_path}" if out_path else ""))

    def _on_stats_done(self, report):
        self.report = report
        self._refresh_all()
        self._set_busy(False)
        fails = len(report.get("failures", []))
        msg = f"统计完成: {report['count']}/{report['requested']} 个数据集，共 {report['total_hours']} 小时"
        if fails:
            msg += f"，{fails} 个读取失败"
        self.status.setText(msg)

    def on_check(self):
        org = self.org_combo.currentText().strip()
        if not org:
            QMessageBox.warning(self, "提示", "请填写组织名。")
            return
        self._set_busy(True)
        self.status.setText(f"检查 {org} 是否有新增数据集 ...")
        self.worker = CheckWorker(org, OUT_DIR, self.token)
        self.worker.result.connect(self._on_check_result)
        self.worker.error.connect(self._on_error)
        self.worker.start()

    def _on_check_result(self, new, removed, hub_count, local_count):
        self._set_busy(False)
        self.status.setText(
            f"Hub {hub_count} 个 / 本地 {local_count} 个，"
            f"新增 {len(new)}，本地多出 {len(removed)}")
        lines = []
        if new:
            lines.append("🆕 新增 (Hub 上有、本地未拉取):\n  " + "\n  ".join(new))
        if removed:
            lines.append("⚠️ 本地多出 (Hub 上已无):\n  " + "\n  ".join(removed))
        if not lines:
            lines.append("本地与 Hub 数据集名称一致，无新增。")
        QMessageBox.information(self, "检查结果", "\n\n".join(lines))

    def on_open_dir(self):
        latest = dd.find_latest_report(OUT_DIR)
        target = Path(latest).parent if latest else Path(OUT_DIR)
        target.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))
        self.status.setText(f"已打开: {target}")

    def _on_error(self, msg):
        self._stop_speed()
        self._set_busy(False)
        self.status.setText(f"错误: {msg}")
        QMessageBox.critical(self, "错误", msg)


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
