#!/usr/bin/env python3
"""TacVerse 多模态物理具身数据集工作台 — PySide6 dashboard over Hugging Face.

Wraps the logic in download_dataset.py. Top bar (org combo + actions + progress
+ speed) is shared; below it a tabbed dashboard:

  * 看板   -> KPI cards (+ today's MVP) + filterable, sortable dataset table.
  * 趋势   -> daily new-hours bar + cumulative-hours line (pyqtgraph).
  * 分组统计 -> rollup by uploader / task / robot_type, plus daily group growth.
  * 数据集编辑 -> 左侧同看板的数据集详情表；右侧两组操作：① 改名 / 改 prompt
    (本地 pyarrow，data+videos 硬链接生成新副本，可推 Hub)；② 删除/拆分/合并/
    增删特征 (子进程调用 lerobot 官方 dataset_tools，见 lerobot_ops_runner.py)。

Buttons: 仅拉取统计信息 (stats only, no download) / 下载当前选中数据集 (one dataset)
/ 拉取组织及其下所有数据集 (download all) / 检查新增数据集 (name diff) /
打开本地目录 / 切换账号 (swap HF token).

Run in the lerobot-xense env:  python main_app.py
"""

import datetime as dt
import json
import os
import sys
import time
from pathlib import Path


def _configure_qt_plugin_path():
    """Prefer the PySide6 plugin directory over conda's qt6-main plugins."""
    plugin_root = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "PySide6"
        / "Qt"
        / "plugins"
    )
    platform_root = plugin_root / "platforms"
    if platform_root.is_dir():
        os.environ["QT_PLUGIN_PATH"] = str(plugin_root)
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(platform_root)


_configure_qt_plugin_path()

import pyqtgraph as pg
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QStackedWidget,
    QSplitter, QTableView, QTableWidget, QTableWidgetItem, QTabWidget, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)


import annotations_reader as ann
import tasks_reader as tsk
import checks as chk_mod
import viewer_service as vsvc 
import download_dataset as dd
import dataset_editor as de
import lerobot_ops as lops

OUT_DIR = "pulls"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"  # logos / image assets
LOGO_PATH = ASSETS_DIR / "logo.png"
RECENT_ORGS = ["TacVerse", "Xense"]  # seeds the editable org combo

# HF uploader id -> Chinese name lives in config.json ("uploader_names"
# section — edit that to add people). Ids with no entry render as 未知. Loaded once
# at startup; edit the file then restart to pick up new names.
_UPLOADER_NAMES = dd.load_uploader_names()

# Thresholds for the custom quality checks (config.json "checks" section).
# Loaded once at startup; edit the file then restart to change standards.
_CHECKS_CFG = dd.load_config().get("checks", {})


def uploader_cn(hf_id):
    """Map an HF uploader id to its Chinese name, or 未知 if absent/unknown."""
    return _UPLOADER_NAMES.get(hf_id, "未知") if hf_id else "未知"


def _esc(s):
    """Minimal HTML escape for text placed into rich-text (QLabel) content."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


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
    "任务": lambda d: dd.task_prefix(d.get("dataset_name", "")),
    "robot_type": lambda d: d.get("robot_type"),
}


def fmt_day(yymmdd):
    """'260703' -> '2026-07-03'. Returns the input unchanged if unparseable."""
    try:
        return dt.datetime.strptime(yymmdd, "%y%m%d").strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return yymmdd or "—"


def fmt_day_wd(yymmdd):
    """'260703' -> '07-03\\n周四' (MM-DD + Chinese weekday) for trend axis labels.
    Weekday helps tell workdays from weekends/holidays at a glance."""
    try:
        d = dt.datetime.strptime(yymmdd, "%y%m%d")
        return d.strftime("%m-%d") + "\n周" + "一二三四五六日"[d.weekday()]
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


def _is_float(s):
    """True if s parses as a float (used to tell split fractions from indices)."""
    try:
        float(s)
        return True
    except (TypeError, ValueError):
        return False


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


class FrozenFirstColumnTable(QTableWidget):
    """QTableWidget with column 0 pinned while the rest scrolls horizontally."""

    def __init__(self, rows=0, columns=0, parent=None, frozen_width=440):
        super().__init__(rows, columns, parent)
        self._frozen_width = frozen_width
        self._frozen = QTableView(self)
        self._frozen.setModel(self.model())
        self._frozen.setSelectionModel(self.selectionModel())
        self._frozen.setFocusPolicy(Qt.NoFocus)
        self._frozen.setFrameShape(QFrame.NoFrame)
        self._frozen.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._frozen.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._frozen.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._frozen.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._frozen.verticalHeader().hide()
        self._frozen.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)
        self._frozen.horizontalHeader().sectionClicked.connect(self._sort_frozen_column)
        self._frozen.doubleClicked.connect(
            lambda index: self.cellDoubleClicked.emit(index.row(), index.column()))
        self.horizontalHeader().sectionResized.connect(self._update_frozen_width)
        self.verticalHeader().sectionResized.connect(self._update_frozen_row_height)
        self.verticalScrollBar().valueChanged.connect(
            self._frozen.verticalScrollBar().setValue)
        self._frozen.verticalScrollBar().valueChanged.connect(
            self.verticalScrollBar().setValue)
        self._sort_order = Qt.AscendingOrder

    def freeze_first_column(self, width=None):
        if width is not None:
            self._frozen_width = width
        for column in range(self.columnCount()):
            self._frozen.setColumnHidden(column, column != 0)
        self.hideColumn(0)
        self._frozen.setColumnWidth(0, self._frozen_width)
        self.setViewportMargins(self._frozen_width, 0, 0, 0)
        self._update_frozen_geometry()
        self._frozen.show()

    def _sort_frozen_column(self, column):
        self.sortItems(column, self._sort_order)
        self._sort_order = (Qt.DescendingOrder
                            if self._sort_order == Qt.AscendingOrder
                            else Qt.AscendingOrder)

    def _update_frozen_width(self, column, old_size, new_size):
        if column != 0:
            return
        self._frozen_width = new_size
        self._frozen.setColumnWidth(0, new_size)
        self.setViewportMargins(new_size, 0, 0, 0)
        self._update_frozen_geometry()

    def _update_frozen_row_height(self, row, old_size, new_size):
        self._frozen.setRowHeight(row, new_size)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_frozen_geometry()

    def scrollTo(self, index, hint=QAbstractItemView.EnsureVisible):
        if index.column() > 0:
            super().scrollTo(index, hint)

    def _update_frozen_geometry(self):
        self._frozen.setGeometry(
            self.verticalHeader().width() + self.frameWidth(),
            self.frameWidth(),
            self._frozen_width,
            self.viewport().height() + self.horizontalHeader().height(),
        )


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


class DownloadOneWorker(QThread):
    """Download a single selected dataset (not the whole org) to save time."""

    done = Signal(str)   # local_dir of the downloaded dataset
    error = Signal(str)

    def __init__(self, repo_id, out_dir, token):
        super().__init__()
        self.repo_id, self.out_dir, self.token = repo_id, out_dir, token

    def run(self):
        try:
            dd.normalize_proxy_env()
            day_dir = Path(self.out_dir) / dt.datetime.now().strftime("%y%m%d")
            day_dir.mkdir(parents=True, exist_ok=True)
            dd.pull_dataset(self.repo_id, day_dir, revision=None, token=self.token)
            self.done.emit(str(day_dir / self.repo_id.split("/")[-1]))
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


class EditWorker(QThread):
    """Write an edited copy of a pulled dataset off the UI thread: hard-link the
    heavy payload into a new dir, then rewrite the prompt in its metadata."""

    done = Signal(str, int)  # dst_dir, n_prompts_changed
    error = Signal(str)

    def __init__(self, src_dir, dst_dir, replacements):
        super().__init__()
        self.src_dir, self.dst_dir, self.replacements = src_dir, dst_dir, replacements

    def run(self):
        try:
            de.copy_dataset(self.src_dir, self.dst_dir)
            n = de.set_prompt(self.dst_dir, self.replacements)
            self.done.emit(str(self.dst_dir), n)
        except Exception as exc:
            # Roll back a half-written copy so a retry starts clean.
            try:
                import shutil
                if Path(self.dst_dir).exists():
                    shutil.rmtree(self.dst_dir)
            except Exception:
                pass
            self.error.emit(str(exc))


class PushWorker(QThread):
    """Upload an edited dataset copy to the Hub off the UI thread."""

    done = Signal(str)  # commit / repo URL
    error = Signal(str)

    def __init__(self, dst_dir, repo_id, token, private=True):
        super().__init__()
        self.dst_dir, self.repo_id = dst_dir, repo_id
        self.token, self.private = token, private

    def run(self):
        try:
            dd.normalize_proxy_env()
            url = de.push_to_hub(self.dst_dir, self.repo_id, self.token,
                                 private=self.private)
            self.done.emit(str(url))
        except Exception as exc:
            self.error.emit(str(exc))


class LerobotOpWorker(QThread):
    """Run a lerobot dataset operation (delete/split/merge/add/remove) via the
    subprocess runner off the UI thread, streaming the child's log lines."""

    log = Signal(str)
    done = Signal(dict)   # the runner's result dict
    error = Signal(str)

    def __init__(self, spec):
        super().__init__()
        self.spec = spec

    def run(self):
        try:
            result = lops.run_op(self.spec, log=self.log.emit)
            if result.get("ok"):
                self.done.emit(result)
            else:
                self.error.emit(result.get("error") or "操作失败")
        except Exception as exc:
            self.error.emit(str(exc))


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("TacVerse 多模态物理具身数据集工作台")
        if LOGO_PATH.is_file():
            self.setWindowIcon(QIcon(str(LOGO_PATH)))
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
        # 数据集编辑 state: the dataset being edited, its prompt editors, and the
        # last copy written (so 推送到 Hub knows what to upload). Workers held on
        # self so they are not GC'd mid-run.
        self._edit_src = None       # selected dataset dict for the edit tab
        self._prompt_edits = []     # [(task_index, old_task, QLineEdit)]
        self._last_copy_dir = None  # Path of the most recent edited copy
        self._last_copy_leaf = None # leaf name of that copy (for the repo id)
        self._edit_worker = None
        self._push_worker = None
        self._op_worker = None

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

        # Render the newest local report immediately so 看板 reflects the last
        # local pull/stat run without requiring network access on startup.
        self.history = dd.load_history(OUT_DIR)
        report, source = dd.load_latest_local_report(
            OUT_DIR, self.org_combo.currentText().strip() or dd.ORG)
        if report:
            self.report = report
            self._refresh_all()
            count = report.get("count", report.get("total_datasets", 0))
            requested = report.get("requested", count)
            self.status.setText(
                f"已加载本地数据: {count}/{requested} "
                f"个数据集，共 {report.get('total_hours', 0)} 小时  ->  {source}")
        else:
            self._refresh_trends()
            self.status.setText(
                "就绪：未发现本地拉取数据；可先点「仅拉取统计信息」或「拉取组织及其下所有数据集」。")
        self._refresh_identity()  # populate the login/visibility indicator

    # ---- UI construction -------------------------------------------------- #
    def _build_ui(self):
        root = QVBoxLayout(self)

        toolbar = QWidget()
        toolbar_v = QVBoxLayout(toolbar)
        toolbar_v.setContentsMargins(0, 0, 0, 0)
        toolbar_v.setSpacing(4)
        top = QHBoxLayout()
        aux = QHBoxLayout()
        for row in (top, aux):
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
        toolbar_v.addLayout(top)
        toolbar_v.addLayout(aux)

        def fit_button(button, min_width=0):
            button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            button.setMinimumWidth(max(min_width, button.sizeHint().width() + 12))

        def vline():
            line = QFrame()
            line.setFrameShape(QFrame.VLine)
            line.setFrameShadow(QFrame.Sunken)
            return line

        if LOGO_PATH.is_file():
            logo = QLabel()
            logo.setPixmap(QPixmap(str(LOGO_PATH)).scaledToHeight(
                30, Qt.SmoothTransformation))
            logo.setToolTip("TacVerse")
            top.addWidget(logo)
            top.addSpacing(8)
        top.addWidget(QLabel("组织:"))
        self.org_combo = QComboBox()
        self.org_combo.setEditable(True)
        self.org_combo.addItems(RECENT_ORGS)
        self.org_combo.setMinimumWidth(160)
        self.org_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.org_combo.currentIndexChanged.connect(self._refresh_identity)
        self.org_combo.lineEdit().editingFinished.connect(self._refresh_identity)
        top.addWidget(self.org_combo)

        # Three primary actions, in increasing cost: stats-only (fast) ->
        # download just the selected dataset -> pull the whole org (slow). Each
        # gets a bold colored look; a small 2nd line spells out the trade-off.
        # The utilities that follow stay plain, behind a vertical divider.
        self.btn_stats = QPushButton("仅拉取统计信息\n（不下载数据集）")
        self.btn_download = QPushButton("下载当前选中数据集")
        self.btn_pull = QPushButton("拉取组织及其下所有数据集\n（速度较慢）")
        self.btn_check = QPushButton("检查新增数据集")
        self.btn_open = QPushButton("打开本地目录")
        self.btn_stats.clicked.connect(self.on_stats)
        self.btn_download.clicked.connect(self.on_download_selected)
        self.btn_pull.clicked.connect(self.on_pull)
        self.btn_check.clicked.connect(self.on_check)
        self.btn_open.clicked.connect(self.on_open_dir)

        primary_css = (
            "QPushButton { font-weight: bold; padding: 5px 14px; border-radius: 6px;"
            " color: white; background: %s; }"
            "QPushButton:hover { background: %s; }"
            "QPushButton:disabled { background: #B0B0B0; }"
        )
        self.btn_stats.setStyleSheet(primary_css % ("#34A853", "#2E9247"))
        self.btn_download.setStyleSheet(primary_css % ("#F59E0B", "#D98A00"))
        self.btn_pull.setStyleSheet(primary_css % ("#4C8BF5", "#3B7AE0"))
        secondary_css = (
            "QPushButton { padding: 5px 12px; border-radius: 6px; color: #444;"
            " border: 1px solid #C4C4C4; background: #F5F5F5; }"
            "QPushButton:hover { background: #ECECEC; }"
        )
        for b in (self.btn_stats, self.btn_download, self.btn_pull):
            b.setMinimumHeight(42)
            fit_button(b)
            top.addWidget(b)
        top.addWidget(vline())
        for b in (self.btn_check, self.btn_open):
            b.setStyleSheet(secondary_css)
            fit_button(b)
            top.addWidget(b)

        top.addSpacing(12)
        top.addWidget(QLabel("每日目标(小时):"))
        self.target_spin = QSpinBox()
        self.target_spin.setRange(0, 100000)
        self.target_spin.setValue(10)
        self.target_spin.setMinimumWidth(76)
        self.target_spin.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.target_spin.valueChanged.connect(self._refresh_kpis)
        top.addWidget(self.target_spin)
        top.addStretch()

        # Viewer service controls, up here in the toolbar (the "Viewer" tab is
        # kept for now but may be removed later — these are the canonical ones).
        aux.addWidget(vline())
        self.top_viewer_dot = QLabel("● Viewer")
        self.top_viewer_dot.setToolTip("Viewer 服务状态")
        aux.addWidget(self.top_viewer_dot)
        self.top_viewer_start = QPushButton("启动")
        self.top_viewer_stop = QPushButton("停止")
        self.top_viewer_home = QPushButton("首页")
        self.open_viewer_btn = QPushButton("🔍 在 Viewer 打开")
        self.open_viewer_btn.setToolTip("在浏览器的 Viewer 里打开选中的数据集")
        self.top_viewer_start.clicked.connect(self._viewer_start)
        self.top_viewer_stop.clicked.connect(self._viewer_stop)
        self.top_viewer_home.clicked.connect(self._viewer_open_home)
        self.open_viewer_btn.clicked.connect(self._open_selected_in_viewer)
        for b in (self.top_viewer_start, self.top_viewer_stop,
                  self.top_viewer_home, self.open_viewer_btn):
            b.setStyleSheet(secondary_css)
            fit_button(b)
            aux.addWidget(b)

        aux.addStretch()
        self.btn_account = QPushButton("切换账号")
        self.btn_account.setStyleSheet(secondary_css)
        self.btn_account.clicked.connect(self.on_switch_account)
        fit_button(self.btn_account)
        aux.addWidget(self.btn_account)
        # Login / visibility indicator — surfaces token & org-permission problems
        # (e.g. "未登录(匿名) · TacVerse 可见 11 个") without any digging.
        self.identity_label = QLabel("登录状态: 检测中…")
        self.identity_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.identity_label.setStyleSheet("color:#888;")
        self.identity_label.setMinimumWidth(220)
        aux.addWidget(self.identity_label)

        # Live clock, far right.
        aux.addWidget(vline())
        self.clock_label = QLabel("")
        self.clock_label.setStyleSheet("color:#444; font-weight:bold;")
        self.clock_label.setMinimumWidth(160)
        aux.addWidget(self.clock_label)
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._tick_clock)
        self.clock_timer.start(1000)
        self._tick_clock()
        toolbar_scroll = QScrollArea()
        toolbar_scroll.setFrameShape(QFrame.NoFrame)
        toolbar_scroll.setWidgetResizable(False)
        toolbar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        toolbar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        toolbar_scroll.setMaximumHeight(toolbar.sizeHint().height() + 18)
        toolbar_scroll.setWidget(toolbar)
        root.addWidget(toolbar_scroll)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_dashboard_tab(), "看板")
        self.tabs.addTab(self._build_trends_tab(), "趋势")
        self.tabs.addTab(self._build_rollup_tab(), "分组统计")
        self.tabs.addTab(self._build_edit_tab(), "数据集编辑")
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

    # KPI cards: (report key, title, highlight?). 总小时数 is highlighted — it's
    # the key metric. Order per spec.
    KPI_CARDS = [
        ("total_datasets", "数据集总数", False),
        ("total_episodes", "总 episodes", False),
        ("new_episodes", "今日新增episodes", False),
        ("total_frames", "总 frames", False),
        ("total_hours", "总小时数", True),
        ("new_hours", "今日新增小时", False),
        ("completion", "目标完成度", False),
    ]

    def _build_dashboard_tab(self):
        """看板 = 左「数据集统计分区」(总览 + 详情表) | 右「数据集检查分区」(分析网格)."""
        w = QWidget()
        outer = QHBoxLayout(w)
        outer.setContentsMargins(4, 4, 4, 4)
        split = QSplitter(Qt.Horizontal)

        # ===== LEFT: 数据集统计分区 =====
        left = QGroupBox("数据集统计分区")
        left.setStyleSheet(
            "QGroupBox{font-weight:bold; border:1px solid #9ec5fe;"
            " border-radius:6px; margin-top:10px; background:#f6f9ff;}"
            "QGroupBox::title{subcontrol-origin:margin; left:10px; color:#1a73e8;}")
        lv = QVBoxLayout(left)

        # 数据集总览 (KPI cards, 4 per row)
        self.kpi_labels = {}
        kpi_grid = QGridLayout()
        for i, (key, title, hl) in enumerate(self.KPI_CARDS):
            kpi_grid.addWidget(self._make_card(key, title, hl), i // 4, i % 4)
        n = len(self.KPI_CARDS)
        kpi_grid.addWidget(self._make_mvp_card(), n // 4, n % 4)
        lv.addLayout(kpi_grid)

        self.baseline_hint = QLabel("")
        self.baseline_hint.setStyleSheet("color: #888; font-size: 12px;")
        lv.addWidget(self.baseline_hint)

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
        lv.addLayout(filt)

        # 数据集详情 (table)
        self.table = FrozenFirstColumnTable(0, len(TABLE_COLS), frozen_width=440)
        self.table.setHorizontalHeaderLabels([c[0] for c in TABLE_COLS])
        self.table.setSortingEnabled(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.verticalHeader().setVisible(False)
        self.table.cellDoubleClicked.connect(self._open_row_link)
        self.table.itemSelectionChanged.connect(self._on_dataset_selected)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Interactive)
        for i in range(1, len(TABLE_COLS)):
            hdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        hdr.setStretchLastSection(True)
        self.table.setColumnWidth(0, 440)
        self.table.freeze_first_column(440)
        lv.addWidget(self.table, 1)
        self.table_hint = QLabel("点「仅拉取统计信息」加载数据集列表(双击行打开 HF 页面)。")
        lv.addWidget(self.table_hint)

        # ===== RIGHT: 数据集检查分区 =====
        right = QGroupBox("数据集检查分区")
        right.setStyleSheet(
            "QGroupBox{font-weight:bold; border:1px solid #a3d9a5;"
            " border-radius:6px; margin-top:10px; background:#f5fbf5;}"
            "QGroupBox::title{subcontrol-origin:margin; left:10px; color:#2e7d32;}")
        rv = QVBoxLayout(right)
        rv.addWidget(self._build_prompt_panel())

        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setCollapsible(1, True)
        split.setSizes([1300, 900])
        outer.addWidget(split)
        return w

    @staticmethod
    def _panel_tree():
        t = QTreeWidget()
        t.setHeaderHidden(True)
        t.setWordWrap(True)
        t.setRootIsDecorated(False)
        return t

    def _build_prompt_panel(self):
        """Right-side detail panel, laid out as a grid that mirrors the viewer's
        tabs: ANNOTATIONS / STATISTICS / FILTERING / FRAMES / ACTION INSIGHTS.

        ANNOTATIONS shows the local task instruction + viewer annotations;
        STATISTICS / FILTERING / ACTION INSIGHTS are filled from the viewer
        /report analysis (fetched async), so the key info is visible without
        opening the viewer WebUI."""
        panel = QWidget()
        pv = QVBoxLayout(panel)
        pv.setContentsMargins(8, 0, 0, 0)

        self.prompt_meta = QLabel("")
        self.prompt_meta.setStyleSheet("color: #888; font-size: 12px;")
        self.prompt_meta.setWordWrap(True)
        pv.addWidget(self.prompt_meta)

        # Indeterminate marquee shown while the /report analysis runs.
        self.report_progress = QProgressBar()
        self.report_progress.setRange(0, 0)  # 0..0 = animated indeterminate
        self.report_progress.setTextVisible(False)
        self.report_progress.setMaximumHeight(6)
        self.report_progress.setVisible(False)
        pv.addWidget(self.report_progress)

        # --- grid of viewer-mirroring panels --------------------------------
        self.detail_grid = QWidget()
        grid = QGridLayout(self.detail_grid)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        grid.setRowStretch(0, 2)  # ANNOTATIONS(0,0 span2) | STATISTICS
        grid.setRowStretch(1, 2)  # ANNOTATIONS cont       | 检查规则
        grid.setRowStretch(2, 2)  # FILTERING              | FRAMES
        grid.setRowStretch(3, 3)  # ACTION INSIGHTS (span 2)

        # 浅绿色分组样式 — 与右侧「数据集检查分区」保持一致
        green_box_css = (
            "QGroupBox{font-weight:bold; border:1px solid #a3d9a5;"
            " border-radius:6px; margin-top:8px; background:#f5fbf5;}"
            "QGroupBox::title{subcontrol-origin:margin; left:10px; color:#2e7d32;}")

        # ANNOTATIONS 标注 — local task instruction + viewer language annotations
        ann_box = QGroupBox("ANNOTATIONS 标注")
        ann_box.setStyleSheet(green_box_css)
        al = QVBoxLayout(ann_box)
        al.addWidget(QLabel("Language Instruction"))
        self.task_list = QListWidget()
        self.task_list.setWordWrap(True)
        self.task_list.setMaximumHeight(84)
        al.addWidget(self.task_list)
        self.task_note = QLabel("")
        self.task_note.setStyleSheet("color: #999; font-size: 12px;")
        self.task_note.setWordWrap(True)
        al.addWidget(self.task_note)
        anno_hd = QLabel("语言标注 (viewer)")
        anno_hd.setStyleSheet("color: #555;")
        al.addWidget(anno_hd)
        ep_row = QHBoxLayout()
        ep_row.addWidget(QLabel("集:"))
        self.prompt_ep = QComboBox()
        self.prompt_ep.currentIndexChanged.connect(self._refresh_prompt_tree)
        ep_row.addWidget(self.prompt_ep, 1)
        self.prompt_ep_wrap = QWidget()
        self.prompt_ep_wrap.setLayout(ep_row)
        al.addWidget(self.prompt_ep_wrap)
        self.prompt_tree = self._panel_tree()
        self.prompt_tree.setRootIsDecorated(True)
        al.addWidget(self.prompt_tree, 1)
        self.anno_note = QLabel("")
        self.anno_note.setStyleSheet("color: #999; font-size: 12px;")
        self.anno_note.setWordWrap(True)
        al.addWidget(self.anno_note)
        grid.addWidget(ann_box, 0, 0, 2, 1)  # tall: spans rows 0-1

        # STATISTICS 统计信息 — dataset/episode stats (from report)
        stat_box = QGroupBox("STATISTICS 统计信息")
        sl = QVBoxLayout(stat_box)
        self.stat_view = QLabel("")
        self.stat_view.setWordWrap(True)
        self.stat_view.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.stat_view.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sl.addWidget(self.stat_view, 1)
        grid.addWidget(stat_box, 0, 1)

        # 检查规则 — custom quality-check results for the selected dataset
        rules_box = QGroupBox("检查规则")
        rules_box.setStyleSheet(green_box_css)
        rul = QVBoxLayout(rules_box)
        self.check_tree = self._panel_tree()
        self.check_tree.setRootIsDecorated(True)
        rul.addWidget(self.check_tree, 1)
        grid.addWidget(rules_box, 1, 1)

        # FILTERING 过滤器 — smoothness "Overall" verdict + breakdown lines
        filt_box = QGroupBox("FILTERING 过滤器")
        fl = QVBoxLayout(filt_box)
        self.filter_view = QLabel("")
        self.filter_view.setWordWrap(True)
        self.filter_view.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.filter_view.setTextInteractionFlags(Qt.TextSelectableByMouse)
        fl.addWidget(self.filter_view, 1)
        grid.addWidget(filt_box, 2, 0)

        # FRAMES 首位帧 — placeholder (not implemented yet)
        frames_box = QGroupBox("FRAMES 首位帧")
        frl = QVBoxLayout(frames_box)
        ph = QLabel("占位，暂未实现")
        ph.setStyleSheet("color: #bbb;")
        ph.setAlignment(Qt.AlignCenter)
        frl.addWidget(ph, 1)
        grid.addWidget(frames_box, 2, 1)

        # ACTION INSIGHTS 行动指导与训练配置 — training config (report)
        insight_box = QGroupBox("ACTION INSIGHTS 行动指导与训练配置")
        il = QVBoxLayout(insight_box)
        self.insight_view = QLabel("")
        self.insight_view.setWordWrap(True)
        self.insight_view.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.insight_view.setTextInteractionFlags(Qt.TextSelectableByMouse)
        il.addWidget(self.insight_view, 1)
        grid.addWidget(insight_box, 3, 0, 1, 2)

        pv.addWidget(self.detail_grid, 1)

        # --- Fallback: nothing selected -------------------------------------
        self.prompt_empty = QLabel("选择左侧数据集查看信息。")
        self.prompt_empty.setStyleSheet("color: #999;")
        self.prompt_empty.setWordWrap(True)
        self.prompt_empty.setAlignment(Qt.AlignCenter)
        pv.addWidget(self.prompt_empty, 1)

        self._prompt_doc = {"episodes": {}, "updated_at": None}
        self._show_prompt_empty("选择左侧数据集查看信息。")
        return panel

    def _make_card(self, key, title, highlight=False):
        card = QFrame()
        card.setFrameShape(QFrame.StyledPanel)
        if highlight:
            # 总小时数 — the key metric, visually distinct.
            card.setStyleSheet(
                "QFrame{background:#e8f5e9; border:1px solid #66bb6a;"
                " border-radius:6px;}")
        cv = QVBoxLayout(card)
        t = QLabel(title)
        t.setStyleSheet(
            "color:#1b5e20; font-size:12px; font-weight:bold;" if highlight
            else "color:#666; font-size:12px;")
        val = QLabel("—")
        val.setStyleSheet(
            "font-size:26px; font-weight:bold; color:#2e7d32;" if highlight
            else "font-size:22px; font-weight:bold;")
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
        # X labels are two lines ("07-03\n周五"); give the bottom axis enough
        # height so the weekday line isn't clipped.
        for pltw in (self.daily_plot, self.cum_plot):
            pltw.getAxis("bottom").setHeight(46)
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

        split = QSplitter(Qt.Vertical)

        summary = QWidget()
        summary_v = QVBoxLayout(summary)
        summary_v.setContentsMargins(0, 0, 0, 0)

        self.rollup_table = QTableWidget(0, 5)
        self.rollup_table.setHorizontalHeaderLabels(
            ["分组", "数据集数", "episodes", "小时", "占比%"])
        self.rollup_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.rollup_table.verticalHeader().setVisible(False)
        self.rollup_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch)
        summary_v.addWidget(self.rollup_table)
        self.rollup_plot = pg.PlotWidget(title="各分组小时数")
        self.rollup_plot.showGrid(x=False, y=True, alpha=0.3)
        summary_v.addWidget(self.rollup_plot)
        split.addWidget(summary)

        daily_group_box = QGroupBox("单组单日新增总时长")
        daily_group_v = QVBoxLayout(daily_group_box)
        self.daily_group_hint = QLabel("按每日最后一次快照对比上一天，随当前分组维度统计新增小时。")
        self.daily_group_hint.setStyleSheet("color:#888; font-size:12px;")
        daily_group_v.addWidget(self.daily_group_hint)
        self.daily_group_table = QTableWidget(0, 5)
        self.daily_group_table.setHorizontalHeaderLabels(
            ["日期", "分组", "新增小时", "新增episodes", "数据集数"])
        self.daily_group_table.setSortingEnabled(True)
        self.daily_group_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.daily_group_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.daily_group_table.verticalHeader().setVisible(False)
        daily_hdr = self.daily_group_table.horizontalHeader()
        daily_hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        daily_hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        for col in range(2, 5):
            daily_hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        daily_group_v.addWidget(self.daily_group_table, 1)
        split.addWidget(daily_group_box)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 2)
        split.setSizes([760, 360])
        v.addWidget(split, 1)
        return w

    # ---- 数据集编辑 tab (edit prompt / rename → new copy, optional push) ---- #
    def _build_edit_tab(self):
        """数据集编辑 = 左「数据集详情表」(同看板) | 右「编辑 + lerobot 操作」.

        Left is the same dataset detail list as 看板 so a dataset can be picked
        here directly. Right has two families:
          * 改名 / 改 Prompt — workbench-native pyarrow edits (no lerobot); the
            heavy payload is hard-linked, output is a new pulls/ copy.
          * 数据集操作 — delete / split / merge / add-feature / remove-feature,
            delegated to lerobot's REAL dataset_tools via a subprocess runner.
        """
        w = QWidget()
        outer = QHBoxLayout(w)
        outer.setContentsMargins(4, 4, 4, 4)
        split = QSplitter(Qt.Horizontal)

        # ===== LEFT: dataset detail table (mirrors 看板) =====
        left = QGroupBox("数据集详情（选中要编辑 / 操作的数据集）")
        left.setStyleSheet(
            "QGroupBox{font-weight:bold; border:1px solid #9ec5fe;"
            " border-radius:6px; margin-top:10px; background:#f6f9ff;}"
            "QGroupBox::title{subcontrol-origin:margin; left:10px; color:#1a73e8;}")
        lv = QVBoxLayout(left)
        ef = QHBoxLayout()
        ef.addWidget(QLabel("筛选:"))
        self.edit_filter = QLineEdit()
        self.edit_filter.setPlaceholderText("按 名称 / robot_type / 上传者 过滤…")
        self.edit_filter.textChanged.connect(self._apply_edit_filter)
        ef.addWidget(self.edit_filter)
        self.edit_only_downloaded = QCheckBox("只看已下载")
        self.edit_only_downloaded.setChecked(True)
        self.edit_only_downloaded.toggled.connect(self._apply_edit_filter)
        ef.addWidget(self.edit_only_downloaded)
        lv.addLayout(ef)

        self.edit_table = QTableWidget(0, len(TABLE_COLS))
        self.edit_table.setHorizontalHeaderLabels([c[0] for c in TABLE_COLS])
        self.edit_table.setSortingEnabled(True)
        self.edit_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.edit_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.edit_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.edit_table.verticalHeader().setVisible(False)
        self.edit_table.itemSelectionChanged.connect(self._refresh_edit_tab)
        ehdr = self.edit_table.horizontalHeader()
        ehdr.setSectionResizeMode(0, QHeaderView.Interactive)
        for i in range(1, len(TABLE_COLS)):
            ehdr.setSectionResizeMode(i, QHeaderView.ResizeToContents)
        ehdr.setStretchLastSection(True)
        self.edit_table.setColumnWidth(0, 380)
        lv.addWidget(self.edit_table, 1)
        split.addWidget(left)

        # ===== RIGHT: editing controls (scrollable) =====
        rscroll = QScrollArea()
        rscroll.setWidgetResizable(True)
        right = QWidget()
        rv = QVBoxLayout(right)

        self.edit_src_lbl = QLabel("—")
        self.edit_src_lbl.setStyleSheet("font-weight:bold;")
        self.edit_src_lbl.setWordWrap(True)
        self.edit_src_lbl.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rv.addWidget(self.edit_src_lbl)

        nrow = QHBoxLayout()
        nrow.addWidget(QLabel("输出数据集名:"))
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("输出文件夹名（最后一段，例如 taccap-g1-...-0713）")
        nrow.addWidget(self.edit_name, 1)
        rv.addLayout(nrow)
        rv.addWidget(self._hline())

        # ---- Group A: 改名 / 改 Prompt (workbench-native) ----
        boxA = QGroupBox("① 改名 / 改 Prompt（本地实现，生成新副本）")
        av = QVBoxLayout(boxA)
        self.edit_prompt_holder = QWidget()
        self.edit_prompt_box = QVBoxLayout(self.edit_prompt_holder)
        self.edit_prompt_box.setContentsMargins(0, 0, 0, 0)
        note = QLabel("（未选择数据集）")
        note.setStyleSheet("color:#888;")
        self.edit_prompt_box.addWidget(note)
        av.addWidget(self.edit_prompt_holder)
        arow = QHBoxLayout()
        self.btn_make_copy = QPushButton("生成新副本")
        self.btn_make_copy.setMinimumHeight(32)
        self.btn_make_copy.setStyleSheet(
            "QPushButton { font-weight:bold; padding:6px 16px; border-radius:6px;"
            " color:white; background:#34A853; }"
            "QPushButton:hover { background:#2E9247; }"
            "QPushButton:disabled { background:#B0B0B0; }")
        self.btn_make_copy.clicked.connect(self.on_make_copy)
        self.btn_push_copy = QPushButton("推送到 Hub")
        self.btn_push_copy.setMinimumHeight(32)
        self.btn_push_copy.setStyleSheet(
            "QPushButton { padding:6px 12px; border-radius:6px; color:#444;"
            " border:1px solid #C4C4C4; background:#F5F5F5; }"
            "QPushButton:hover { background:#ECECEC; }"
            "QPushButton:disabled { color:#AAA; }")
        self.btn_push_copy.clicked.connect(self.on_push_copy)
        arow.addWidget(self.btn_make_copy)
        arow.addWidget(self.btn_push_copy)
        arow.addStretch()
        av.addLayout(arow)
        rv.addWidget(boxA)

        # ---- Group B: lerobot 数据集操作 ----
        boxB = QGroupBox("② 数据集操作（lerobot：删 / 拆 / 并 / 特征）")
        bv = QVBoxLayout(boxB)
        oprow = QHBoxLayout()
        oprow.addWidget(QLabel("操作:"))
        self.op_combo = QComboBox()
        self.op_combo.addItems(
            ["删除 episodes", "拆分数据集", "合并数据集", "增加特征", "删除特征"])
        self.op_combo.currentIndexChanged.connect(self._on_op_changed)
        oprow.addWidget(self.op_combo, 1)
        bv.addLayout(oprow)
        self.op_stack = QStackedWidget()
        self.op_stack.addWidget(self._build_op_delete())
        self.op_stack.addWidget(self._build_op_split())
        self.op_stack.addWidget(self._build_op_merge())
        self.op_stack.addWidget(self._build_op_addfeat())
        self.op_stack.addWidget(self._build_op_rmfeat())
        bv.addWidget(self.op_stack)
        self.btn_run_op = QPushButton("执行操作（生成新数据集）")
        self.btn_run_op.setMinimumHeight(32)
        self.btn_run_op.setStyleSheet(
            "QPushButton { font-weight:bold; padding:6px 16px; border-radius:6px;"
            " color:white; background:#4C8BF5; }"
            "QPushButton:hover { background:#3B7AE0; }"
            "QPushButton:disabled { background:#B0B0B0; }")
        self.btn_run_op.clicked.connect(self.on_run_op)
        bv.addWidget(self.btn_run_op)
        self.op_note = QLabel(
            "输出写到 pulls/<今天>/，视频操作用 CPU 编码(libx264)较慢，请耐心等待。")
        self.op_note.setStyleSheet("color:#888; font-size:12px;")
        self.op_note.setWordWrap(True)
        bv.addWidget(self.op_note)
        rv.addWidget(boxB)

        self.edit_result = QLabel("")
        self.edit_result.setWordWrap(True)
        self.edit_result.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.edit_result.setStyleSheet("color:#2e7d32;")
        rv.addWidget(self.edit_result)
        rv.addStretch()

        rscroll.setWidget(right)
        split.addWidget(rscroll)
        split.setSizes([1080, 640])
        outer.addWidget(split)

        self._set_edit_enabled(False)
        return w

    def _hline(self):
        ln = QFrame()
        ln.setFrameShape(QFrame.HLine)
        ln.setFrameShadow(QFrame.Sunken)
        return ln

    # ---- op sub-forms (fields stashed on self; rebuilt lists on selection) ----
    def _build_op_delete(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel("要删除的 episode 序号（逗号分隔，如 0,2,5）:"))
        self.op_del_indices = QLineEdit()
        self.op_del_indices.setPlaceholderText("0,2,5")
        v.addWidget(self.op_del_indices)
        return w

    def _build_op_split(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel("拆分方式（比例或序号区间）:"))
        self.op_split_spec = QLineEdit()
        self.op_split_spec.setPlaceholderText("train:0.8,val:0.2  或  train:0-4,val:5-6")
        v.addWidget(self.op_split_spec)
        tip = QLabel("输出为 <输出名>_train / <输出名>_val 等，写到 pulls/<今天>/。")
        tip.setStyleSheet("color:#888; font-size:12px;")
        tip.setWordWrap(True)
        v.addWidget(tip)
        return w

    def _build_op_merge(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel("勾选要合并的数据集（需已下载，输出名用上方「输出数据集名」）:"))
        self.op_merge_list = QListWidget()
        self.op_merge_list.setMaximumHeight(160)
        v.addWidget(self.op_merge_list)
        return w

    def _build_op_addfeat(self):
        w = QWidget()
        g = QGridLayout(w)
        g.setContentsMargins(0, 0, 0, 0)
        g.addWidget(QLabel("特征名:"), 0, 0)
        self.op_add_name = QLineEdit()
        self.op_add_name.setPlaceholderText("reward")
        g.addWidget(self.op_add_name, 0, 1)
        g.addWidget(QLabel("dtype:"), 1, 0)
        self.op_add_dtype = QComboBox()
        self.op_add_dtype.addItems(["float32", "float64", "int64"])
        g.addWidget(self.op_add_dtype, 1, 1)
        g.addWidget(QLabel("shape:"), 2, 0)
        self.op_add_shape = QLineEdit("1")
        self.op_add_shape.setPlaceholderText("1  或  3（向量维度）")
        g.addWidget(self.op_add_shape, 2, 1)
        g.addWidget(QLabel("填充值:"), 3, 0)
        self.op_add_fill = QLineEdit("0")
        g.addWidget(self.op_add_fill, 3, 1)
        g.setColumnStretch(1, 1)
        return w

    def _build_op_rmfeat(self):
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(QLabel("勾选要删除的特征 / 相机:"))
        self.op_rm_list = QListWidget()
        self.op_rm_list.setMaximumHeight(160)
        v.addWidget(self.op_rm_list)
        return w

    def _on_op_changed(self, idx):
        self.op_stack.setCurrentIndex(idx)

    def _apply_edit_filter(self):
        q = self.edit_filter.text().strip().lower()
        only_dl = self.edit_only_downloaded.isChecked()
        downloaded = self._downloaded_leaves()
        for row in range(self.edit_table.rowCount()):
            item = self.edit_table.item(row, 0)
            d = item.data(Qt.UserRole) if item else {}
            d = d or {}
            hay = " ".join(str(d.get(k, "")) for k in
                           ("dataset_name", "robot_type", "uploader")).lower()
            hay += " " + uploader_cn(d.get("uploader")).lower()
            hide = bool(q) and q not in hay
            if not hide and only_dl:
                leaf = (d.get("dataset_name") or "").split("/")[-1]
                hide = leaf not in downloaded
            self.edit_table.setRowHidden(row, hide)

    def _set_edit_enabled(self, on):
        """Enable/disable the edit inputs (off when nothing editable is selected)."""
        for wdg in (self.edit_name, self.btn_make_copy, self.op_combo,
                    self.op_stack, self.btn_run_op):
            wdg.setEnabled(on)
        # Push only makes sense once a copy exists.
        self.btn_push_copy.setEnabled(on and self._last_copy_dir is not None)

    def _dataset_dir(self, d):
        """On-disk directory of a selected dataset, or None if not downloaded.

        Prefers the record's local_dir; else the newest pulls/*/<leaf>/ that has
        meta/info.json (mirrors tasks_reader/_downloaded_leaves resolution)."""
        local = (d or {}).get("local_dir")
        if local and (Path(local) / "meta" / "info.json").is_file():
            return Path(local)
        leaf = ((d or {}).get("dataset_name") or "").split("/")[-1]
        if not leaf:
            return None
        cands = [p.parent.parent for p in
                 Path(OUT_DIR).glob(f"*/{leaf}/meta/info.json")]
        return max(cands, key=lambda p: p.stat().st_mtime) if cands else None

    def _clear_prompt_edits(self):
        """Tear down the per-selection prompt editor rows."""
        self._prompt_edits = []
        while self.edit_prompt_box.count():
            item = self.edit_prompt_box.takeAt(0)
            child = item.widget()
            if child:
                child.deleteLater()

    def _selected_edit_dataset(self):
        """The dataset dict selected in the 数据集编辑 tab's own table, or None."""
        row = self.edit_table.currentRow()
        if row < 0:
            return None
        item = self.edit_table.item(row, 0)
        return item.data(Qt.UserRole) if item else None

    def _downloaded_dataset_dirs(self):
        """{leaf: newest dir} for every dataset materialized under pulls/."""
        out = {}
        for info in Path(OUT_DIR).glob("*/*/meta/info.json"):
            ddir = info.parent.parent
            leaf = ddir.name
            prev = out.get(leaf)
            if prev is None or ddir.stat().st_mtime > prev.stat().st_mtime:
                out[leaf] = ddir
        return out

    def _refresh_edit_tab(self):
        """Sync the edit tab to its table selection (called on select)."""
        # A copy is tied to the previously-selected source; reset on change.
        self._last_copy_dir = None
        self._last_copy_leaf = None
        self.edit_result.setText("")
        self._clear_prompt_edits()
        self._refresh_merge_list()

        d = self._selected_edit_dataset()
        src_dir = self._dataset_dir(d) if d else None
        self._edit_src = d
        if not d or src_dir is None:
            self.edit_src_lbl.setText("—")
            self.edit_name.setText("")
            note = QLabel("请选择一个已下载(已拉取)的数据集（仅统计的行不能编辑）。")
            note.setStyleSheet("color:#888;")
            self.edit_prompt_box.addWidget(note)
            self.op_rm_list.clear()
            self._set_edit_enabled(False)
            return

        name = d.get("dataset_name") or ""
        leaf = name.split("/")[-1]
        self.edit_src_lbl.setText(f"{name}  ·  {src_dir}")
        self.edit_name.setText(leaf)  # default: same name (user changes to rename)

        rows, err = tsk.load(src_dir / tsk.TASKS_REL)
        if err:
            note = QLabel(f"无法读取指令: {err}")
            note.setStyleSheet("color:#c62828;")
            self.edit_prompt_box.addWidget(note)
        elif not rows:
            note = QLabel("该数据集没有 tasks.parquet（无可编辑指令），仍可改名生成副本。")
            note.setStyleSheet("color:#888;")
            self.edit_prompt_box.addWidget(note)
        else:
            for r in rows:
                holder = QWidget()
                line = QHBoxLayout(holder)
                line.setContentsMargins(0, 0, 0, 0)
                line.addWidget(QLabel(f"#{r['index']}"))
                edit = QLineEdit(r["task"])
                line.addWidget(edit, 1)
                self.edit_prompt_box.addWidget(holder)
                self._prompt_edits.append((r["index"], r["task"], edit))

        self._refresh_rmfeat_list(src_dir)
        self._set_edit_enabled(True)

    def _refresh_merge_list(self):
        """Populate the merge candidate list with all downloaded datasets."""
        if not hasattr(self, "op_merge_list"):
            return
        self.op_merge_list.clear()
        for leaf, ddir in sorted(self._downloaded_dataset_dirs().items()):
            it = QListWidgetItem(leaf)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            it.setData(Qt.UserRole, str(ddir))
            self.op_merge_list.addItem(it)

    def _refresh_rmfeat_list(self, src_dir):
        """Populate the remove-feature list from the selected dataset's schema."""
        self.op_rm_list.clear()
        info = de.read_info(src_dir)
        required = {"timestamp", "frame_index", "episode_index", "index", "task_index"}
        for key in (info.get("features") or {}):
            if key in required:
                continue  # lerobot forbids removing these
            it = QListWidgetItem(key)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Unchecked)
            self.op_rm_list.addItem(it)

    def on_make_copy(self):
        d, src = self._edit_src, self._dataset_dir(self._edit_src)
        if not d or src is None:
            QMessageBox.warning(self, "提示", "请先选择一个已下载的数据集。")
            return
        try:
            new_leaf = de.validate_leaf(self.edit_name.text())
        except ValueError as exc:
            QMessageBox.warning(self, "名字不合法", str(exc))
            return

        # Prompt replacements: old string -> edited string (only real changes).
        replacements = {}
        for _idx, old, edit in self._prompt_edits:
            new = edit.text().strip()
            if new and new != old:
                replacements[old] = new

        dst = de.default_copy_dir(new_leaf, OUT_DIR)
        if dst.exists():
            QMessageBox.warning(
                self, "目标已存在",
                f"{dst} 已存在。请换一个新名字，或先删除该目录。")
            return
        if not replacements and new_leaf == src.name:
            QMessageBox.information(self, "无改动", "指令和名字都没有变化，未生成副本。")
            return

        self._set_busy(True)
        self.status.setText(f"生成副本 {dst} ...")
        self.edit_result.setText("")
        self._edit_worker = EditWorker(str(src), str(dst), replacements)
        self._edit_worker.done.connect(self._on_edit_done)
        self._edit_worker.error.connect(self._on_error)
        self._edit_worker.start()

    def _on_edit_done(self, dst_dir, n_changed):
        self._set_busy(False)
        self._last_copy_dir = Path(dst_dir)
        self._last_copy_leaf = Path(dst_dir).name
        self.btn_push_copy.setEnabled(True)
        note = f"已生成副本: {dst_dir}（修改 {n_changed} 条指令）"
        self.edit_result.setText(note)
        self.status.setText(note)
        self._refresh_table()  # the new copy now counts as 已下载
        QMessageBox.information(
            self, "完成",
            f"{note}\n\n数据/视频为硬链接（未额外占用磁盘）。\n"
            f"如需上传到 HuggingFace，点「推送到 Hub」。")

    def on_push_copy(self):
        if not self._last_copy_dir or not self._last_copy_dir.exists():
            QMessageBox.warning(self, "提示", "请先「生成新副本」。")
            return
        org = self.org_combo.currentText().strip()
        if not org:
            QMessageBox.warning(self, "提示", "请填写组织名。")
            return
        if not self.token:
            QMessageBox.warning(self, "未登录",
                                "当前没有 HuggingFace token，无法上传。请先「切换账号」。")
            return
        repo_id = f"{org}/{self._last_copy_leaf}"
        ok = QMessageBox.question(
            self, "确认上传",
            f"将把\n  {self._last_copy_dir}\n上传为 HuggingFace 数据集（私有）:\n"
            f"  {repo_id}\n\n确定继续？",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if ok != QMessageBox.Yes:
            return
        self._set_busy(True)
        self.status.setText(f"上传到 {repo_id} ...")
        self._push_worker = PushWorker(
            str(self._last_copy_dir), repo_id, self.token, private=True)
        self._push_worker.done.connect(self._on_push_done)
        self._push_worker.error.connect(self._on_error)
        self._push_worker.start()

    def _on_push_done(self, url):
        self._set_busy(False)
        msg = f"上传完成: {url}"
        self.edit_result.setText(msg)
        self.status.setText(msg)
        QMessageBox.information(self, "上传完成", msg)

    # ---- lerobot 操作 (delete / split / merge / add / remove) --------------- #
    @staticmethod
    def _parse_int_list(text):
        """'0,2,5' or '0-3,7' -> [0,2,5] / [0,1,2,3,7]. Raises ValueError."""
        out = []
        for part in text.replace("，", ",").split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                a, b = part.split("-", 1)
                out.extend(range(int(a), int(b) + 1))
            else:
                out.append(int(part))
        return out

    def _parse_splits(self, text):
        """'train:0.8,val:0.2' -> {train:0.8,val:0.2};
        'train:0-4,val:5-6' -> {train:[0..4], val:[5,6]}. Raises ValueError."""
        splits = {}
        for chunk in text.replace("，", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if ":" not in chunk:
                raise ValueError(f"格式应为 名字:值 —— '{chunk}'")
            name, val = chunk.split(":", 1)
            name, val = name.strip(), val.strip()
            if any(c in val for c in "-,") or (not _is_float(val)):
                # explicit index list (e.g. 0-4) grouped per split
                splits[name] = self._parse_int_list(val)
            else:
                splits[name] = float(val)
        kinds = {isinstance(v, float) for v in splits.values()}
        if len(kinds) > 1:
            raise ValueError("比例和序号区间不能混用")
        return splits

    def on_run_op(self):
        d, src = self._edit_src, self._dataset_dir(self._edit_src)
        op_idx = self.op_combo.currentIndex()
        org = self.org_combo.currentText().strip() or "TacVerse"

        # Merge does not require a selected row (it uses the checkbox list); the
        # others operate on the selected dataset.
        if op_idx != 2 and (not d or src is None):
            QMessageBox.warning(self, "提示", "请先在左表选择一个已下载的数据集。")
            return
        try:
            out_leaf = de.validate_leaf(self.edit_name.text())
        except ValueError as exc:
            QMessageBox.warning(self, "输出名不合法", str(exc))
            return

        spec = {"vcodec": lops.DEFAULT_VCODEC, "params": {}}
        out_dir = lops.default_out_dir(out_leaf, OUT_DIR)
        src_repo = d.get("dataset_name") if d else f"{org}/{out_leaf}"

        try:
            if op_idx == 0:  # delete episodes
                idx = self._parse_int_list(self.op_del_indices.text())
                if not idx:
                    raise ValueError("请填写要删除的 episode 序号。")
                spec.update(op="delete", sources=[{"repo_id": src_repo, "root": str(src)}],
                            out_dir=str(out_dir), out_repo_id=f"{org}/{out_leaf}")
                spec["params"]["episode_indices"] = idx
            elif op_idx == 1:  # split
                splits = self._parse_splits(self.op_split_spec.text())
                if not splits:
                    raise ValueError("请填写拆分方式。")
                spec.update(op="split", sources=[{"repo_id": src_repo, "root": str(src)}],
                            out_parent=str(out_dir.parent), out_leaf=out_leaf,
                            out_repo_id=f"{org}/{out_leaf}")
                spec["params"]["splits"] = splits
            elif op_idx == 2:  # merge
                sources = []
                for i in range(self.op_merge_list.count()):
                    it = self.op_merge_list.item(i)
                    if it.checkState() == Qt.Checked:
                        sources.append({"repo_id": f"{org}/{it.text()}",
                                        "root": it.data(Qt.UserRole)})
                if len(sources) < 2:
                    raise ValueError("请至少勾选 2 个数据集进行合并。")
                spec.update(op="merge", sources=sources,
                            out_dir=str(out_dir), out_repo_id=f"{org}/{out_leaf}")
            elif op_idx == 3:  # add feature
                name = self.op_add_name.text().strip()
                if not name:
                    raise ValueError("请填写特征名。")
                shape = self._parse_int_list(self.op_add_shape.text() or "1")
                if not shape:
                    raise ValueError("shape 至少要有一个维度，如 1 或 3。")
                fill_txt = (self.op_add_fill.text() or "0").strip()
                dtype = self.op_add_dtype.currentText()
                fill = int(fill_txt) if dtype == "int64" else float(fill_txt)
                spec.update(op="add_feature", sources=[{"repo_id": src_repo, "root": str(src)}],
                            out_dir=str(out_dir), out_repo_id=f"{org}/{out_leaf}")
                spec["params"].update(name=name, dtype=dtype, shape=shape, fill=fill)
            elif op_idx == 4:  # remove feature
                names = [self.op_rm_list.item(i).text()
                         for i in range(self.op_rm_list.count())
                         if self.op_rm_list.item(i).checkState() == Qt.Checked]
                if not names:
                    raise ValueError("请勾选要删除的特征。")
                spec.update(op="remove_feature", sources=[{"repo_id": src_repo, "root": str(src)}],
                            out_dir=str(out_dir), out_repo_id=f"{org}/{out_leaf}")
                spec["params"]["feature_names"] = names
        except ValueError as exc:
            QMessageBox.warning(self, "参数有误", str(exc))
            return

        # Guard against clobbering existing output dirs (split writes siblings).
        if spec["op"] == "split":
            clashes = [p for name in spec["params"]["splits"]
                       if (p := out_dir.parent / f"{out_leaf}_{name}").exists()]
            if clashes:
                QMessageBox.warning(self, "目标已存在",
                                    "以下输出目录已存在，请换名：\n" +
                                    "\n".join(str(c) for c in clashes))
                return
        elif out_dir.exists():
            QMessageBox.warning(self, "目标已存在",
                                f"{out_dir} 已存在。请换一个输出名。")
            return

        self._set_busy(True)
        self.edit_result.setText("")
        self.status.setText(f"执行 {self.op_combo.currentText()} ...（视频操作较慢）")
        self._op_worker = LerobotOpWorker(spec)
        self._op_worker.log.connect(self.status.setText)
        self._op_worker.done.connect(self._on_op_done)
        self._op_worker.error.connect(self._on_error)
        self._op_worker.start()

    def _on_op_done(self, result):
        self._set_busy(False)
        outs = result.get("outputs", [])
        lines = [f"{o['repo_id']}  ({o['episodes']} eps / {o['frames']} frames)\n  → {o['root']}"
                 for o in outs]
        msg = f"{self.op_combo.currentText()} 完成，生成 {len(outs)} 个数据集:"
        self.edit_result.setText(msg + "\n" + "\n".join(lines))
        self.status.setText(msg)
        self._refresh_table()  # new outputs now count as 已下载
        QMessageBox.information(self, "完成", msg + "\n\n" + "\n".join(lines))

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
        new_hours, new_eps = self._new_totals(deltas)
        target = self.target_spin.value()
        pct = f"{round(100 * new_hours / target)}%" if target else "—"
        self.kpi_labels["total_datasets"].setText(fmt_value(r.get("total_datasets")))
        self.kpi_labels["total_hours"].setText(fmt_value(r.get("total_hours")))
        self.kpi_labels["total_episodes"].setText(fmt_value(r.get("total_episodes")))
        self.kpi_labels["total_frames"].setText(fmt_value(r.get("total_frames")))
        self.kpi_labels["new_hours"].setText(f"+{new_hours}")
        self.kpi_labels["new_episodes"].setText(f"+{fmt_value(new_eps)}")
        self.kpi_labels["completion"].setText(pct)

    def _new_totals(self, deltas):
        """(new_hours, new_episodes) since the baseline day.

        Prefer the sum of per-dataset deltas; but when the baseline snapshot has
        no per-dataset detail (e.g. a backfilled aggregate-only day), those are
        all zero — fall back to the difference of the aggregate totals so 今日新增
        is still correct."""
        base = dd.find_baseline(self.report, self.history) if self.report else None
        if base and not base.get("datasets"):
            nh = round((self.report.get("total_hours") or 0)
                       - (base.get("total_hours") or 0), 2)
            ne = (self.report.get("total_episodes") or 0) \
                - (base.get("total_episodes") or 0)
            return nh, ne
        nh = round(sum(d["d_hours"] for d in deltas.values()), 2)
        ne = sum(d["d_episodes"] for d in deltas.values())
        return nh, ne

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

    def _fill_dataset_table(self, table, datasets, deltas, downloaded):
        """Populate a QTableWidget with the dataset detail rows (shared by the
        看板 table and the 数据集编辑 table so both show the same list)."""
        table.setSortingEnabled(False)
        table.setRowCount(len(datasets))
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
                table.setItem(row, col, item)
        table.setSortingEnabled(True)
        # Default order: most-recently-updated first (matches org page / 发现顺序).
        table.sortItems(DATE_COL, Qt.DescendingOrder)

    def _refresh_table(self):
        r = self.report
        datasets = r.get("datasets", []) if r else []
        deltas = self._current_deltas()
        downloaded = self._downloaded_leaves()  # dataset leaf names present in pulls/
        self._fill_dataset_table(self.table, datasets, deltas, downloaded)
        # Mirror the same list into the 数据集编辑 tab's table (if built).
        if hasattr(self, "edit_table"):
            self._fill_dataset_table(self.edit_table, datasets, deltas, downloaded)
            self._apply_edit_filter()
            self._refresh_merge_list()
        self.table_hint.setText(
            f"共 {len(datasets)} 个数据集，双击行打开 HF 页面；点表头排序。"
            if datasets else "点「仅拉取统计信息」加载数据集列表。")
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
        self.detail_grid.setVisible(False)
        self.report_progress.setVisible(False)

    def _on_dataset_selected(self):
        d = self._selected_dataset()
        if not d:
            self.prompt_meta.setText("")
            self._show_prompt_empty("选择左侧数据集查看信息。")
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
        self.detail_grid.setVisible(True)

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

    # ---- Viewer /report analysis (async → STATISTICS/FILTERING/INSIGHTS) --- #
    _VERDICT_COLOR = {
        "Smooth": "#2e7d32", "Consistent": "#2e7d32",
        "Moderate": "#ef8c00", "Moderate variance": "#ef8c00",
        "Jerky": "#c62828", "High variance": "#c62828", "N/A": "#9e9e9e",
    }

    def _report_set_note(self, msg, busy=False):
        """Put a status/placeholder message in the three report-driven boxes."""
        note = f"<span style='color:#999'>{_esc(msg)}</span>"
        for lbl in (self.stat_view, self.filter_view, self.insight_view):
            lbl.setText(note)
        self.report_progress.setVisible(busy)

    def _refresh_report(self, d):
        """Fill STATISTICS / FILTERING / ACTION INSIGHTS from the viewer /report
        analysis for the selected dataset. Fetched in a background thread (can
        take tens of seconds); cached per session; stale selections ignored."""
        self._report_seq += 1
        seq = self._report_seq
        if not self.viewer.is_running():
            self._report_set_note("Viewer 未运行；顶栏点「启动」后显示分析。")
            return
        rel = self.viewer.dataset_rel_path(d, root=self._viewer_root())
        if not rel:
            self._report_set_note("该数据集不在 Viewer 数据根（最新拉取日），暂无分析。")
            return
        cached = self._report_cache.get(rel)
        if cached is not None:
            self._render_report(cached)
            return
        self._report_set_note("分析中…（首次约 10–30s）", busy=True)
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
            self._report_set_note(f"分析失败: {err}")
            return
        self._render_report(report)

    def _render_report(self, r):
        """Split the /report fields across STATISTICS / FILTERING / ACTION
        INSIGHTS as clean rich-text (label muted, value bold, verdict colored)."""
        self.report_progress.setVisible(False)
        ds = r.get("dataset") or {}
        q = r.get("quality") or {}
        t = r.get("training") or {}
        sm = q.get("smoothness")

        def b(val):  # bold value
            return f"<b>{_esc(fmt_value(val) if isinstance(val, (int, float)) else val)}</b>"

        def verdict(label):
            c = self._VERDICT_COLOR.get(label, "#333")
            return f"<b style='color:{c}'>{_esc(label)}</b>"

        def detail(text):  # muted sub-line under a value (like the viewer badge)
            return (f"<div style='color:#8a8f99; font-size:11px;"
                    f" margin-top:2px'>{_esc(text)}</div>")

        def kv_table(rows):  # 内嵌 2 列表格：指标 | 值
            body = "".join(
                "<tr>"
                "<td style='padding:4px 8px; color:#5b6b5b; background:#eef7ee;"
                " border:1px solid #d4e7d4; white-space:nowrap;"
                " vertical-align:top'>" + lab + "</td>"
                "<td style='padding:4px 8px; border:1px solid #d4e7d4;"
                " vertical-align:top'>" + val + "</td>"
                "</tr>"
                for lab, val in rows)
            return ("<table width='100%' cellspacing='0' cellpadding='0' "
                    "style='border-collapse:collapse'>" + body + "</table>")

        # --- STATISTICS (内嵌表格，与「数据集统计分区」风格一致) ---
        integ = r.get("integrity") or {}
        st = integ.get("status", "?")
        st_col = "#2e7d32" if st == "ok" else "#c62828"
        st_html = f"<b style='color:{st_col}'>{_esc(st)}</b>"
        if integ.get("issues"):
            st_html += (f"<br><span style='color:#c62828; font-size:11px'>"
                        f"{_esc('; '.join(integ['issues']))}</span>")

        trows = [("完整性", st_html),
                 ("Episodes", b(ds.get("total_episodes"))),
                 ("Frames", b(ds.get("total_frames"))),
                 ("摄像头", b(len(ds.get("cameras") or []))),
                 ("fps", b(ds.get("fps")))]
        el = q.get("episodeLength")
        if el:
            trows.append(("时长 最短/最长 (s)",
                          f"{b(el.get('shortest'))} / {b(el.get('longest'))}"))
            trows.append(("时长 均值/中位 (s)",
                          f"{b(el.get('mean'))} / {b(el.get('median'))}"))
            trows.append(("时长 std", b(el.get("std"))))
        if q:
            trows.append(("抖动集", b(len(q.get("jerkyEpisodes") or []))))
            trows.append(("低运动集", b(len(q.get("lowMovementEpisodes") or []))))

        self.stat_view.setText(kv_table(trows))

        # --- FILTERING: smoothness "Overall" + breakdown lines ---
        if sm:
            label = (sm.get("verdict") or {}).get("label") or "—"
            html = f"<div style='line-height:150%'>Overall: {verdict(label)}"
            lines = sm.get("lines") or []
            if lines:
                html += "<ul style='margin:6px 0 0 -24px;'>" + "".join(
                    f"<li>{_esc(l)}</li>" for l in lines) + "</ul>"
            if sm.get("tip"):
                html += f"<div style='color:#999; margin-top:4px'>{_esc(sm['tip'])}</div>"
            html += "</div>"
            self.filter_view.setText(html)
        else:
            self.filter_view.setText("<span style='color:#999'>无平滑度数据</span>")

        # --- ACTION INSIGHTS: training config (内嵌表格) ---
        irows = []
        sc = t.get("suggestedChunkLength")
        if sc:
            secs = round(sc.get("seconds", 0), 2)
            val = (f"<b>{sc.get('steps')} 步 ({secs}s)</b>"
                   + detail("自相关首次降到 0.5 以下的中位滞后步数（跨各动作维度）——"
                            "即动作可预测、可打包为一个 chunk 的时域长度。"))
            irows.append(("建议 chunk 长度", val))
        elif "training" in r:
            irows.append(("建议 chunk 长度", "—"))
        cd = t.get("controlDelay")
        if cd:
            steps = cd.get("meanSteps")
            secs = round(cd.get("seconds", 0), 3)
            causal = ("<span style='color:#2e7d32'>· 因果 ✓</span>" if cd.get("causalOk")
                      else "<span style='color:#c62828'>· 非因果 ✗</span>")
            if isinstance(steps, (int, float)) and steps > 0:
                d = (f"状态变化平均滞后动作约 {steps} 帧；"
                     f"建议将 action[t] 与 state[t+{steps}] 对齐。")
            elif isinstance(steps, (int, float)) and steps < 0:
                d = f"动作平均滞后状态变化约 {-steps} 帧（预测性动作）。"
            else:
                d = "动作与状态基本同步，无明显时延。"
            val = f"<b>{steps} 步 ({secs}s)</b> {causal}" + detail(d)
            irows.append(("控制延迟", val))
        sv = t.get("speedVariance")
        if sv:
            tail = (" <span style='color:#c62828'>· 需速度归一</span>"
                    if sv.get("needsVelocityNorm") else "")
            val = (f"{verdict((sv.get('verdict') or {}).get('label'))} "
                   f"<span style='color:#8a8f99'>cv {round(sv.get('cv', 0), 3)}</span>{tail}")
            irows.append(("速度方差", val))
        meta = r.get("meta") or {}
        if meta.get("sampledEpisodes") is not None:
            irows.append(("抽样",
                          f"<span style='color:#aaa'>{meta.get('sampledEpisodes')} 集</span>"))
        self.insight_view.setText(kv_table(irows))

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
            item = QListWidgetItem(row["task"])
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
            self.trend_hint.setText(
                "暂无历史数据。执行「仅拉取统计信息」或「拉取组织及其下所有数据集」后按天积累趋势。")
            return
        self.trend_hint.setText(
            "" if len(series) >= 2 else "当前仅 1 天数据，多日拉取后可见增长趋势。")
        # Categorical x = only days that were actually pulled, packed side by side
        # (未统计的日期不占位，不留空白). fmt_day makes labels read 07-03 not 260703.
        x = list(range(len(series)))
        labels = [fmt_day_wd(s["date"]) for s in series]  # MM-DD + 周X
        ticks = [list(zip(x, labels))]
        bg = pg.BarGraphItem(x=x, height=[s.get("new_hours", 0) for s in series],
                             width=0.8, brush="#4C8BF5")
        self.daily_plot.addItem(bg)
        self.daily_plot.getAxis("bottom").setTicks(ticks)
        self.daily_plot.setXRange(-0.5, len(series) - 0.5, padding=0)
        self.cum_plot.plot(x, [s.get("total_hours", 0) for s in series],
                           pen=pg.mkPen("#34A853", width=2), symbol="o",
                           symbolBrush="#34A853")
        self.cum_plot.getAxis("bottom").setTicks(ticks)
        self.cum_plot.setXRange(-0.5, len(series) - 0.5, padding=0.02)

    def _refresh_daily_group_table(self, rows, dim):
        self.daily_group_table.setSortingEnabled(False)
        self.daily_group_table.setRowCount(len(rows))
        if not rows:
            self.daily_group_hint.setText(f"暂无可归因到「{dim}」的单日新增数据。")
        else:
            self.daily_group_hint.setText(
                f"按每日最后一次快照对比上一天，统计每个「{dim}」分组的新增小时。")
        for i, row in enumerate(rows):
            values = [
                fmt_day(row.get("date")),
                row.get("hours", 0),
                row.get("episodes", 0),
                row.get("datasets", 0),
            ]
            values.insert(1, row.get("group") or "—")
            for j, value in enumerate(values):
                if j >= 2:
                    item = NumericItem(fmt_value(value), value)
                else:
                    item = QTableWidgetItem(str(value))
                self.daily_group_table.setItem(i, j, item)
        self.daily_group_table.setSortingEnabled(True)
        self.daily_group_table.sortItems(0, Qt.DescendingOrder)

    def _refresh_rollup(self):
        self.rollup_table.setRowCount(0)
        self.rollup_plot.clear()
        dim = self.dim_combo.currentText()
        key_fn = ROLLUP_DIMS[dim]
        daily_rows = dd.daily_group_series(self.history, key_fn)
        self._refresh_daily_group_table(daily_rows, dim)
        if not self.report:
            return
        rows = dd.rollup(self.report.get("datasets", []), key_fn)
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

        def _short(s, k=42):
            s = str(s)
            return s if len(s) <= k else s[:k - 1] + "…"

        labels = [_short(g["group"]) for g in plot_rows]
        left = self.rollup_plot.getAxis("left")
        left.setTicks([[(ys[i], labels[i]) for i in range(n)]])
        # Widen the y-axis to the longest label so nothing is clipped; CJK glyphs
        # take ~2x the width of a latin char, so weight them double when sizing.
        vis = max((sum(2 if ord(c) > 0x2E80 else 1 for c in s) for s in labels),
                  default=8)
        left.setWidth(min(440, max(70, 12 + vis * 8)))
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
        for b in (self.btn_pull, self.btn_stats, self.btn_download,
                  self.btn_check, self.btn_open):
            b.setEnabled(not busy)
        # Edit-tab actions share the busy lock so a copy/push can't overlap a pull.
        if hasattr(self, "btn_make_copy"):
            self.btn_make_copy.setEnabled(not busy)
            self.btn_run_op.setEnabled(not busy)
            self.btn_push_copy.setEnabled(
                not busy and self._last_copy_dir is not None)

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

    def on_download_selected(self):
        """Download ONLY the dataset selected in the 看板 table (fast path)."""
        d = self._selected_dataset()
        if not d or not d.get("dataset_name"):
            QMessageBox.warning(self, "提示", "请先在「看板」表格里选中一个数据集。")
            return
        repo_id = d["dataset_name"]
        self._set_busy(True)
        self.bar.setMaximum(0)  # indeterminate — a single snapshot download
        self._watch_dir = Path(OUT_DIR) / dt.datetime.now().strftime("%y%m%d")
        self._prev_bytes = dir_size(self._watch_dir)
        self._prev_t = time.monotonic()
        self.speed_label.setText("0.0 B/s")
        self.speed_timer.start()
        self.status.setText(f"开始下载 {repo_id} ...")
        self.dl_worker = DownloadOneWorker(repo_id, OUT_DIR, self.token)
        self.dl_worker.done.connect(self._on_download_one_done)
        self.dl_worker.error.connect(self._on_error)
        self.dl_worker.start()

    def _on_download_one_done(self, local_dir):
        self._stop_speed()
        self.bar.setMaximum(1)
        self.bar.setValue(1)
        self._set_busy(False)
        self._refresh_table()  # the newly downloaded row now shows 已下载
        msg = f"下载完成: {local_dir}"
        self.status.setText(msg)
        QMessageBox.information(self, "完成", f"已下载到本地:\n{local_dir}")

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

    def _tick_clock(self):
        now = dt.datetime.now()
        week = "一二三四五六日"[now.weekday()]
        self.clock_label.setText(now.strftime(f"%Y-%m-%d 周{week} %H:%M:%S"))

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
        # Record the day's totals so 趋势 / 今日新增 have a daily baseline. 统计
        # produces per-dataset detail (from each info.json), so this snapshot is
        # a full baseline — previously only 拉取 wrote history, which is why days
        # that were only 统计'd never showed up.
        hist_note = ""
        try:
            dd.append_history(report)
            self.history = dd.load_history(OUT_DIR)
        except OSError as exc:
            hist_note = f"（历史未写入: {exc}）"
        self._refresh_all()
        self._set_busy(False)
        fails = len(report.get("failures", []))
        msg = f"统计完成: {report['count']}/{report['requested']} 个数据集，共 {report['total_hours']} 小时"
        if fails:
            msg += f"，{fails} 个读取失败"
        self.status.setText(msg + hist_note)

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


APP_ID = "tacverse-workbench"  # WM class / desktop-file base name (taskbar match)


def main():
    app = QApplication(sys.argv)
    # Taskbar/dock icon: an app-level icon plus a stable WM class that matches an
    # installed <APP_ID>.desktop, so GNOME/Ubuntu show the logo instead of the
    # generic gear. (setWindowIcon on the window alone is not enough on Linux.)
    app.setApplicationName(APP_ID)
    app.setApplicationDisplayName("TacVerse 多模态物理具身数据集工作台")
    app.setDesktopFileName(APP_ID)
    if LOGO_PATH.is_file():
        app.setWindowIcon(QIcon(str(LOGO_PATH)))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
