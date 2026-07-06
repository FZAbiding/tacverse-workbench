#!/usr/bin/env python3
"""PySide6 team dashboard for embodied-AI dataset collection on Hugging Face.

Wraps the logic in download_dataset.py. Top bar (org combo + actions + progress
+ speed) is shared; below it a tabbed dashboard:

  * 看板   -> KPI cards + filterable, sortable dataset table.
  * 趋势   -> daily new-hours bar + cumulative-hours line (pyqtgraph).
  * 分组统计 -> rollup by robot_type / uploader / task, table + bar.

Buttons: 拉取当前数据集 (download) / 统计当前数据集 (stats only, no download) /
检查新增数据集 (name diff) / 打开本地目录.

Run in the lerobot-xense env:  python gui_app.py
"""

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QComboBox, QFrame, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton,
    QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

import download_dataset as dd

OUT_DIR = "pulls"
RECENT_ORGS = ["TacVerse", "Xense"]  # seeds the editable org combo

# HF uploader id -> human (Chinese) name. Edit uploader_names.json to add people;
# ids with no entry render as 未知. Resolved relative to this file so it works
# regardless of the process's working directory.
UPLOADER_NAMES_FILE = Path(__file__).parent / "uploader_names.json"


def load_uploader_names():
    try:
        return json.loads(UPLOADER_NAMES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_UPLOADER_NAMES = load_uploader_names()


def uploader_cn(hf_id):
    """Map an HF uploader id to its Chinese name, or 未知 if absent/unknown."""
    return _UPLOADER_NAMES.get(hf_id, "未知") if hf_id else "未知"

pg.setConfigOptions(background="w", foreground="k", antialias=True)

# Dashboard table columns: (header, dataset key, kind). "__delta__" is special.
TABLE_COLS = [
    ("数据集", "dataset_name", "str"),
    ("episodes", "total_episodes", "num"),
    ("frames", "total_frames", "num"),
    ("小时", "duration_hours", "num"),
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

ROLLUP_DIMS = {
    "robot_type": lambda d: d.get("robot_type"),
    "上传者": lambda d: uploader_cn(d.get("uploader")),
    "任务": lambda d: dd.task_prefix(d["dataset_name"]),
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


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HF 具身数据采集看板")
        self.resize(1080, 720)
        self.token = os.environ.get("HF_TOKEN")
        self.worker = None
        self.report = None
        self.history = []

        self._build_ui()

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

    # ---- UI construction -------------------------------------------------- #
    def _build_ui(self):
        root = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("组织:"))
        self.org_combo = QComboBox()
        self.org_combo.setEditable(True)
        self.org_combo.addItems(RECENT_ORGS)
        self.org_combo.setMinimumWidth(160)
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
        top.addSpacing(16)
        top.addWidget(QLabel("每日目标(小时):"))
        self.target_spin = QSpinBox()
        self.target_spin.setRange(0, 100000)
        self.target_spin.setValue(8)
        self.target_spin.valueChanged.connect(self._refresh_kpis)
        top.addWidget(self.target_spin)
        top.addStretch()
        root.addLayout(top)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_dashboard_tab(), "看板")
        self.tabs.addTab(self._build_trends_tab(), "趋势")
        self.tabs.addTab(self._build_rollup_tab(), "分组统计")
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
            ("new_hours", "今日新增小时"), ("new_episodes", "今日新增episodes"),
            ("completion", "目标完成度"),
        ]:
            cards.addWidget(self._make_card(key, title))
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
        v.addLayout(filt)

        # Table
        self.table = QTableWidget(0, len(TABLE_COLS))
        self.table.setHorizontalHeaderLabels([c[0] for c in TABLE_COLS])
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.cellDoubleClicked.connect(self._open_row_link)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(TABLE_COLS)):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        v.addWidget(self.table, 1)
        self.table_hint = QLabel("点「统计当前数据集」加载数据集列表(双击行打开 HF 页面)。")
        v.addWidget(self.table_hint)
        return w

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
            return
        self._refresh_baseline_hint()
        deltas = self._current_deltas()
        new_hours = round(sum(d["d_hours"] for d in deltas.values()), 2)
        new_eps = sum(d["d_episodes"] for d in deltas.values())
        target = self.target_spin.value()
        pct = f"{round(100 * new_hours / target)}%" if target else "—"
        self.kpi_labels["total_datasets"].setText(fmt_value(r.get("total_datasets")))
        self.kpi_labels["total_hours"].setText(fmt_value(r.get("total_hours")))
        self.kpi_labels["new_hours"].setText(f"+{new_hours}")
        self.kpi_labels["new_episodes"].setText(f"+{fmt_value(new_eps)}")
        self.kpi_labels["completion"].setText(pct)

    def _refresh_table(self):
        r = self.report
        datasets = r.get("datasets", []) if r else []
        deltas = self._current_deltas()
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(datasets))
        for row, d in enumerate(datasets):
            for col, (_, key, kind) in enumerate(TABLE_COLS):
                if key == "__delta__":
                    dv = deltas.get(d["dataset_name"], {})
                    n = dv.get("d_episodes", 0)
                    txt = ("🆕 " if dv.get("is_new") else "") + (f"+{n}" if n else "0")
                    item = NumericItem(txt, n)
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
        for row in range(self.table.rowCount()):
            d = self.table.item(row, 0).data(Qt.UserRole) or {}
            hay = " ".join(str(d.get(k, "")) for k in
                           ("dataset_name", "robot_type", "uploader")).lower()
            hay += " " + uploader_cn(d.get("uploader")).lower()
            self.table.setRowHidden(row, bool(q) and q not in hay)

    def _open_row_link(self, row, _col):
        d = self.table.item(row, 0).data(Qt.UserRole) or {}
        link = d.get("link")
        if link:
            QDesktopServices.openUrl(QUrl(link))
            self.status.setText(f"已打开: {link}")

    def _refresh_trends(self):
        series = dd.daily_series(self.history)
        self.daily_plot.clear()
        self.cum_plot.clear()
        if not series:
            self.trend_hint.setText("暂无历史数据。执行「拉取当前数据集」后按天积累趋势。")
            return
        self.trend_hint.setText(
            "" if len(series) >= 2 else "当前仅 1 天数据，多日拉取后可见增长趋势。")
        x = list(range(len(series)))
        dates = [s["date"] for s in series]
        ticks = [list(zip(x, dates))]
        bg = pg.BarGraphItem(x=x, height=[s["total_hours"] for s in series],
                             width=0.6, brush="#4C8BF5")
        self.daily_plot.addItem(bg)
        self.daily_plot.getAxis("bottom").setTicks(ticks)
        self.cum_plot.plot(x, [s["cum_hours"] for s in series],
                           pen=pg.mkPen("#34A853", width=2), symbol="o",
                           symbolBrush="#34A853")
        self.cum_plot.getAxis("bottom").setTicks(ticks)

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
        x = list(range(len(rows)))
        bg = pg.BarGraphItem(x=x, height=[g["hours"] for g in rows],
                             width=0.6, brush="#F9A825")
        self.rollup_plot.addItem(bg)
        self.rollup_plot.getAxis("bottom").setTicks(
            [list(zip(x, [g["group"] for g in rows]))])

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
