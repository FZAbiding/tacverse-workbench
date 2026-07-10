"""Manage the vendored xense_lerobot_viewer as a black-box web service.

workbench talks to the viewer ONLY through its three stable contracts, so the
viewer's source is never modified and stays upgradable from upstream:

  ① LOCAL_DATASET_ROOT env  → the shared dataset root the viewer scans
  ② HTTP on PORT            → home `/` and `/_local/<encodedPath>` deep links
  ③ side-effect files       → meta/xense_tags.json, meta/lerobot_annotations.json

This module owns the viewer subprocess lifecycle (start / health / stop) and
builds deep-link URLs. It is Qt-free so it can be unit-tested and reused.
"""

import base64
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.request
import webbrowser
from pathlib import Path

VIEWER_DIR = Path(__file__).resolve().parent / "vendor" / "lerobot_viewer"
DEFAULT_PORT = 3000


def encode_dataset_path(rel_path: str) -> str:
    """base64url(rel_path) without padding.

    Mirrors the viewer's encodeLocalDatasetPath so `/_local/<enc>` resolves to
    the same dataset the viewer discovered.
    """
    rel = rel_path.replace("\\", "/").strip("/")
    raw = base64.urlsafe_b64encode(rel.encode("utf-8")).decode("ascii")
    return raw.rstrip("=")


def find_bun():
    """Locate the bun executable (PATH or the default ~/.bun install)."""
    found = shutil.which("bun")
    if found:
        return found
    candidate = Path.home() / ".bun" / "bin" / "bun"
    return str(candidate) if candidate.is_file() else None


def _port_in_use(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


def _http_ok(url, timeout=1.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 500
    except Exception:
        return False


class ViewerService:
    """Supervises one `bun run dev` viewer process bound to a dataset root."""

    def __init__(self, viewer_dir=VIEWER_DIR, port=DEFAULT_PORT):
        self.viewer_dir = Path(viewer_dir)
        self.port = int(port)
        self.root = None
        self.proc = None
        self._log_path = None

    # --- URLs (contract ②) ------------------------------------------------
    @property
    def base_url(self):
        return f"http://127.0.0.1:{self.port}"

    def home_url(self):
        return self.base_url + "/"

    def dataset_url(self, rel_path, episode=None):
        url = f"{self.base_url}/_local/{encode_dataset_path(rel_path)}"
        if episode is not None:
            url += f"/{episode}"
        return url

    def dataset_rel_path(self, dataset, root=None):
        """Path of `dataset` relative to the dataset root, or None if not local.

        Uses the record's local_dir when present, else globs the root for a
        matching leaf directory (the stats-only case). Returns None when the
        dataset isn't under the root (so the caller can disable the open link).
        """
        root = Path(root or self.root or "").resolve()
        if not root:
            return None
        local_dir = (dataset or {}).get("local_dir")
        if local_dir:
            p = Path(local_dir).resolve()
            try:
                rel = p.relative_to(root)
                return str(rel)
            except ValueError:
                pass
        leaf = ((dataset or {}).get("dataset_name") or "").split("/")[-1]
        if not leaf:
            return None
        matches = [d for d in root.glob(f"*/{leaf}") if (d / "meta").is_dir()]
        matches += [d for d in root.glob(leaf) if (d / "meta").is_dir()]
        if not matches:
            return None
        best = max(matches, key=lambda d: d.stat().st_mtime)
        return str(best.relative_to(root))

    # --- lifecycle (contract ①②) -----------------------------------------
    def available(self):
        """viewer directory + installed deps present?"""
        return (self.viewer_dir / "package.json").is_file() and \
               (self.viewer_dir / "node_modules").is_dir()

    def is_running(self):
        if self.proc and self.proc.poll() is None:
            return True
        return _port_in_use(self.port)

    def is_ready(self):
        return _http_ok(self.home_url(), timeout=1.0)

    def start(self, root, wait=False, timeout=60, log_path=None):
        """Launch (or reuse) the viewer bound to `root`. Returns (ok, message).

        `wait=False` returns as soon as the process is spawned; poll status()
        for readiness. `wait=True` blocks until the home page answers or times
        out. If the port is already served, the existing instance is reused.
        """
        self.root = str(Path(root).resolve())
        if _port_in_use(self.port):
            return True, f"端口 {self.port} 已在运行，复用现有服务"
        if not self.available():
            return False, f"viewer 未就绪（缺 node_modules）: {self.viewer_dir}"
        bun = find_bun()
        if not bun:
            return False, "未找到 bun，请先安装 bun"

        env = dict(os.environ)
        env["LOCAL_DATASET_ROOT"] = self.root
        env["PORT"] = str(self.port)
        self._log_path = log_path
        out = open(log_path, "w") if log_path else subprocess.DEVNULL
        try:
            self.proc = subprocess.Popen(
                [bun, "run", "dev"],
                cwd=str(self.viewer_dir),
                env=env,
                stdout=out,
                stderr=subprocess.STDOUT if log_path else subprocess.DEVNULL,
                start_new_session=True,  # own process group → clean shutdown
            )
        except Exception as exc:
            return False, f"启动失败: {exc}"

        if not wait:
            return True, "启动中…"
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False, "viewer 进程意外退出（查看日志）"
            if _http_ok(self.home_url(), timeout=1.0):
                return True, f"运行中 · {self.base_url}"
            time.sleep(0.5)
        return False, f"启动超时（{timeout}s），可稍后重试或查看日志"

    def stop(self):
        """Terminate the managed process (and its group). Idempotent."""
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                try:
                    self.proc.terminate()
                except Exception:
                    pass
        self.proc = None

    # --- introspection (contract ②) --------------------------------------
    def dataset_count(self):
        """How many datasets the viewer currently sees (JSON API), or None."""
        try:
            with urllib.request.urlopen(
                    self.base_url + "/api/local-datasets", timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return len(data.get("datasets", []))
        except Exception:
            return None

    def report(self, rel_path, include=None, timeout=180):
        """Fetch the viewer's /report analysis JSON for a dataset.

        Returns (report_dict, None) on success, or (None, error_message). The
        analysis is computed server-side and can take tens of seconds — call
        this off the UI thread.
        """
        url = f"{self.base_url}/api/local-datasets/{encode_dataset_path(rel_path)}/report"
        if include:
            url += "?include=" + ",".join(include)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            return None, str(exc)
        if isinstance(data, dict) and data.get("ok") is False:
            return None, str(data.get("error") or "analysis failed")
        return data, None

    def status(self):
        running = self.is_running()
        return {
            "running": running,
            "ready": self.is_ready() if running else False,
            "managed": bool(self.proc and self.proc.poll() is None),
            "port": self.port,
            "root": self.root,
            "url": self.base_url,
        }

    # --- convenience openers ----------------------------------------------
    def open_home(self):
        webbrowser.open(self.home_url())

    def open_dataset(self, rel_path, episode=None):
        webbrowser.open(self.dataset_url(rel_path, episode))
