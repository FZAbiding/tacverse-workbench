"""User-defined dataset quality checks.

A small, Qt-free registry so new rules are added by appending one Check — the
UI never changes. Each check is scoped to a `provider` ("custom" for the rules
below; "viewer" is reserved for checks adapted from xense_lerobot_viewer later)
so the two families render in separate groups and stay fully independent.

Thresholds live in config.json's "checks" section (see DEFAULTS); pass that dict
as `cfg` to run_checks so standards can be tuned without touching code.
"""

import re
from dataclasses import dataclass, field
from typing import Callable

# --- status vocabulary ------------------------------------------------------
OK, WARN, FAIL, SKIP = "ok", "warn", "fail", "skip"
_ICON = {OK: "✅", WARN: "⚠️", FAIL: "❌", SKIP: "—"}
_SEVERITY = {OK: 0, SKIP: 0, WARN: 1, FAIL: 2}


def icon(status: str) -> str:
    return _ICON.get(status, "?")


# --- default thresholds (overridden by config.json "checks") ---------------
DEFAULTS = {
    "name_format": {
        # TacVerse/taccap-g1-<verb>-<noun...>-<MMDD>. Task id may be multi-word
        # and mixed-case (e.g. Open-cup-lid); the last segment must be 4 digits.
        "regex": r"^TacVerse/taccap-g1-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-\d{4}$",
    },
    "avg_duration": {"min_sec": 20, "max_sec": 600},
    "prompt": {
        "min_words": 10,
        "max_words": 50,
        # Characters that should not appear inside a prompt sentence.
        "illegal_chars": [
            "_", "-",
            "，", "。", "、", "；", "：", "？", "！",
            "（", "）", "“", "”", "‘", "’", "《", "》", "—", "·",
        ],
    },
    "local_quality": {
        # Local episode-level checks. They run only for a selected downloaded
        # dataset, not for every table row.
        "start_end_window": 10,
        "boundary_mad_factor": 6.0,
        "boundary_abs_threshold": 0.20,
        "jump_mad_factor": 8.0,
        "jump_abs_threshold": 0.15,
        "flicker_sample_step": 1,
        "flicker_luma_threshold": 45.0,
        "flicker_recover_ratio": 0.45,
        "max_video_frames": 12000,
        "max_issues": 80,
    },
}


def _cfg(cfg, key):
    """Merge a config subsection over its defaults."""
    merged = dict(DEFAULTS.get(key, {}))
    if cfg:
        merged.update(cfg.get(key, {}) or {})
    return merged


# --- data model -------------------------------------------------------------
@dataclass
class CheckResult:
    id: str
    title: str
    provider: str
    status: str
    message: str
    details: list = field(default_factory=list)  # optional sub-lines


@dataclass
class Check:
    id: str
    title: str
    provider: str
    fn: Callable  # fn(dataset, cfg) -> (status, message, details)


REGISTRY: list = []


def register(id, title, provider="custom"):
    """Decorator: append a check to the registry. Adding a rule = one entry."""
    def deco(fn):
        REGISTRY.append(Check(id, title, provider, fn))
        return fn
    return deco


# --- custom checks ----------------------------------------------------------
@register("name_format", "名称规范")
def _check_name(dataset, cfg):
    name = dataset.get("dataset_name") or ""
    if not name:
        return SKIP, "无数据集名", []
    pattern = _cfg(cfg, "name_format")["regex"]
    if re.match(pattern, name):
        return OK, "符合命名规范", []
    return FAIL, "不符合 TacVerse/taccap-g1-<动词-名词>-<日期>", []


def _avg_seconds(dataset):
    eps = dataset.get("total_episodes") or 0
    hrs = dataset.get("duration_hours") or 0
    if not eps:
        return None
    return hrs * 3600 / eps


@register("avg_duration", "均时长")
def _check_avg_duration(dataset, cfg):
    avg = _avg_seconds(dataset)
    if avg is None:
        return SKIP, "无 episodes，无法计算", []
    c = _cfg(cfg, "avg_duration")
    lo, hi = c["min_sec"], c["max_sec"]
    if avg < lo:
        return FAIL, f"均时长 {avg:.1f}s 偏短(<{lo}s)", []
    if avg > hi:
        return FAIL, f"均时长 {avg:.1f}s 偏长(>{hi}s)", []
    return OK, f"均时长 {avg:.1f}s，在 {lo}-{hi}s 内", []


def _check_one_prompt(text, idx, c):
    """Return a list of (status, line) findings for a single prompt string."""
    findings = []
    words = text.split()
    n = len(words)
    if n < c["min_words"] or n > c["max_words"]:
        findings.append(
            (WARN, f"[{idx}] 词数 {n}(需 {c['min_words']}-{c['max_words']})"))
    bad = [ch for ch in c["illegal_chars"] if ch in text]
    if bad:
        findings.append((FAIL, f"[{idx}] 含非法字符: {' '.join(dict.fromkeys(bad))}"))
    # (b) formula structure — heuristic placeholder, refined later. Only flags
    # obviously non-sentence prompts; a passing prompt is not asserted correct.
    if n < 3 or not words[0][:1].isalpha():
        findings.append((WARN, f"[{idx}] 结构可能不符合公式(待细化)"))
    return findings


@register("prompt_quality", "Prompt 规范")
def _check_prompt(dataset, cfg):
    tasks = dataset.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return SKIP, "无 Prompt(先统计/拉取)", []
    c = _cfg(cfg, "prompt")
    all_findings = []
    for row in tasks:
        idx = row.get("index", "?")
        text = row.get("task") or ""
        all_findings.extend(_check_one_prompt(text, idx, c))
    if not all_findings:
        return OK, "符合规范", []
    worst = max((_SEVERITY[s] for s, _ in all_findings), default=0)
    status = FAIL if worst >= _SEVERITY[FAIL] else WARN
    details = [line for _, line in all_findings]
    return status, f"{len(all_findings)} 项待改", details


@register("episode_local_quality", "Episode 级质量定位", provider="local_quality")
def _check_local_quality(dataset, cfg):
    """Locate camera flicker, boundary pose outliers and trajectory jumps.

    The heavy work is delegated to dataset_quality and cached by dataset path.
    This rule is intentionally under provider="local_quality" so table refreshes
    stay fast; the GUI invokes it only for the selected dataset detail panel.
    """
    import dataset_quality

    issues = dataset_quality.scan_dataset(
        dataset, out_dir="pulls", cfg=_cfg(cfg, "local_quality"))
    return format_local_quality_issues(issues)


def format_local_quality_issues(issues):
    """Convert dataset_quality Issue objects to one CheckResult payload."""
    actionable = [x for x in issues if x.severity in (WARN, FAIL)]
    skipped = [x for x in issues if x.severity == SKIP]
    if not actionable:
        if skipped:
            return SKIP, skipped[0].message, [x.message for x in skipped[1:]]
        return OK, "未发现 episode 级异常", []

    worst = max((_SEVERITY.get(x.severity, 0) for x in actionable), default=0)
    status = FAIL if worst >= _SEVERITY[FAIL] else WARN
    details = []
    for x in actionable:
        parts = []
        if x.episode_index is not None:
            parts.append(f"episode {x.episode_index}")
        if x.start_sec is not None and x.end_sec is not None:
            parts.append(f"{x.start_sec:.3f}s-{x.end_sec:.3f}s")
        elif x.frame_index is not None:
            parts.append(f"frame {x.frame_index}")
        if x.field:
            parts.append(x.field)
        prefix = " / ".join(parts)
        line = f"{prefix}: {x.message}" if prefix else x.message
        if x.clip_path:
            line += f" | video切片: {x.clip_path}"
        details.append(line)
    return status, f"{len(actionable)} 项 episode 级问题", details


# --- runner + aggregation ---------------------------------------------------
def run_checks(dataset, providers=("custom",), cfg=None):
    """Run every registered check for the given providers against one dataset.

    Returns (results, aggregate). `aggregate` = {worst, n_fail, n_warn} for a
    table badge. A check that raises degrades to a SKIP result (never crashes
    the dashboard).
    """
    results = []
    for chk in REGISTRY:
        if chk.provider not in providers:
            continue
        try:
            status, message, details = chk.fn(dataset, cfg)
        except Exception as exc:  # a broken rule must not break the UI
            status, message, details = SKIP, f"检查出错: {exc}", []
        results.append(
            CheckResult(chk.id, chk.title, chk.provider, status, message, details or []))
    return results, aggregate(results)


def aggregate(results):
    n_fail = sum(1 for r in results if r.status == FAIL)
    n_warn = sum(1 for r in results if r.status == WARN)
    worst = FAIL if n_fail else (WARN if n_warn else OK)
    return {"worst": worst, "n_fail": n_fail, "n_warn": n_warn}


def badge(agg):
    """(text, sort_key) for the dashboard's 检查 column. Higher key = worse."""
    if agg["n_fail"]:
        return f"❌{agg['n_fail']}", 200 + agg["n_fail"]
    if agg["n_warn"]:
        return f"⚠️{agg['n_warn']}", 100 + agg["n_warn"]
    return "✅", 0
