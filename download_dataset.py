#!/usr/bin/env python3
"""Pull one or more Hugging Face dataset repos into date-stamped folders.

Datasets are re-pulled on every run (`snapshot_download` syncs incrementally,
so newly merged files are fetched and unchanged ones are skipped). Each run
writes everything under a per-day folder:

    <out-dir>/<YYMMDD>/<dataset-name>/...             # dataset files
    <out-dir>/<YYMMDD>/pull_result_<YYMMDD>_<HHMM>.json  # aggregate summary

The list of datasets lives in DATASETS and can be overridden with repeated
`--repo-id` flags. The fields lifted from each dataset's meta/info.json are
declared in INFO_FIELDS, so extending the report is a one-line change.
"""

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path

# By default every dataset under this org is discovered and pulled. Override
# with --org, or pass explicit --repo-id flags to pull a specific subset.
ORG = "TacVerse"

HF_DATASET_URL = "https://huggingface.co/datasets/{repo_id}"

# Fields copied verbatim from meta/info.json into each dataset's summary.
# Extend this list to surface more of info.json (e.g. "fps", "total_tasks",
# "robot_type") with no other change. `key` is the output name, `source` the
# info.json key; `required=False` skips the field for datasets that lack it.
INFO_FIELDS = [
    {"key": "total_episodes", "source": "total_episodes"},
    {"key": "total_frames", "source": "total_frames"},
    {"key": "fps", "source": "fps", "required": False},
    {"key": "robot_type", "source": "robot_type", "required": False},
    {"key": "total_tasks", "source": "total_tasks", "required": False},
]

# Assumed capture rate (frames per second) when a dataset's info.json omits fps.
DEFAULT_FPS = 30


def normalize_proxy_env() -> None:
    """Make the shell proxy vars parseable by httpx (huggingface_hub 1.x).

    httpx rejects a schemeless `socks://` proxy URL. The http(s)_proxy vars
    already cover HTTPS traffic to the Hub, so drop the offending ALL_PROXY
    vars and normalize any remaining socks:// value to socks5://.
    """
    for var in ("ALL_PROXY", "all_proxy"):
        os.environ.pop(var, None)
    for var in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        val = os.environ.get(var)
        if val and val.startswith("socks://"):
            os.environ[var] = "socks5://" + val[len("socks://"):]


def _apply_info(summary: dict, info: dict) -> dict:
    """Fill INFO_FIELDS + derived duration_hours into `summary` from an info dict."""
    for field in INFO_FIELDS:
        src = field["source"]
        if src in info:
            summary[field["key"]] = info[src]
        elif field.get("required", True):
            summary[field["key"]] = None
    # Recording duration in hours = frames / fps / 3600.
    frames = summary.get("total_frames")
    if frames is not None:
        fps = summary.get("fps") or DEFAULT_FPS
        summary["duration_hours"] = round(frames / fps / 3600, 3)
    return summary


def build_summary(repo_id: str, local_dir: str) -> dict:
    """Assemble a per-dataset summary from a *downloaded* dataset directory.

    Derived fields (name, link, local_dir) plus every entry in INFO_FIELDS read
    from meta/info.json. Missing info.json or missing keys degrade gracefully.
    """
    summary = {
        "dataset_name": repo_id,
        "link": HF_DATASET_URL.format(repo_id=repo_id),
        "local_dir": str(local_dir),
    }
    info_path = Path(local_dir) / "meta" / "info.json"
    if info_path.is_file():
        _apply_info(summary, json.loads(info_path.read_text()))
    else:
        # Not a LeRobot-style dataset (no meta/info.json); leave the
        # info-derived fields absent rather than guessing.
        print(f"Note: {info_path} not found; summary limited to name and link.")
    tasks_path = Path(local_dir) / "meta" / "tasks.parquet"
    if tasks_path.is_file():
        import tasks_reader

        rows, _ = tasks_reader.load(tasks_path)
        summary["tasks"] = rows
    return summary


def fetch_tasks(repo_id: str, token=None) -> list:
    """Fetch a dataset's task instructions from meta/tasks.parquet.

    tasks.parquet is a tiny file (a few KB) carrying the natural-language task
    string(s) the dataset was recorded against — the base "prompt". Fetched on
    the stats-only path so the dashboard can show prompts without a full pull.
    Returns [{"index", "task"}] (sorted), or [] if absent/unreadable.
    """
    from huggingface_hub import hf_hub_download

    try:
        path = hf_hub_download(
            repo_id=repo_id,
            filename="meta/tasks.parquet",
            repo_type="dataset",
            token=token,
        )
    except Exception:
        return []  # dataset has no tasks.parquet (or no access)
    import tasks_reader

    rows, _ = tasks_reader.load(path)
    return rows


def fetch_summary(repo_id: str, token=None) -> dict:
    """Summarize a dataset by fetching only small meta files (no full download).

    Used for the stats-only path: downloads meta/info.json (+ meta/tasks.parquet
    for the task prompt) instead of the whole (potentially huge) dataset. Falls
    back to name+link if info.json is absent.
    """
    from huggingface_hub import hf_hub_download

    summary = {
        "dataset_name": repo_id,
        "link": HF_DATASET_URL.format(repo_id=repo_id),
    }
    try:
        info_path = hf_hub_download(
            repo_id=repo_id,
            filename="meta/info.json",
            repo_type="dataset",
            token=token,
        )
    except Exception:
        return summary  # no info.json -> name+link only
    _apply_info(summary, json.loads(Path(info_path).read_text()))
    summary["tasks"] = fetch_tasks(repo_id, token)
    return summary


def discover_datasets_meta(org, token):
    """Return [{"id", "last_modified"}] for every dataset under an org/user.

    Ordered most-recently-updated first, matching the Hugging Face org page's
    default "Recently updated" sort. Datasets missing a timestamp sort last.
    """
    from huggingface_hub import list_datasets

    # Ask the Hub for its own "Recently updated" ranking when available; older
    # huggingface_hub versions (e.g. 1.23.x) do not expose a `direction` kwarg,
    # so we sort client-side as the source of truth and also pin timestamp-less
    # repos last.
    ds = list(list_datasets(author=org, token=token, sort="lastModified"))
    ds.sort(key=lambda d: (d.last_modified is not None, d.last_modified), reverse=True)
    out = []
    for d in ds:
        lm = d.last_modified
        out.append({"id": d.id, "last_modified": lm.isoformat() if lm else None})
    return out


def discover_datasets(org, token):
    """Return every dataset repo id under an org/user (recently-updated first)."""
    return [d["id"] for d in discover_datasets_meta(org, token)]


def fetch_uploader(repo_id, token=None):
    """Return uploader info from the dataset's HF commit history.

    `uploader` is the author of the earliest (initial) commit — i.e. who created
    the dataset. `uploaders` lists every distinct commit author. Degrades to an
    empty dict on any error (private repo, network, etc.).
    """
    from huggingface_hub import HfApi

    try:
        commits = HfApi().list_repo_commits(repo_id, repo_type="dataset", token=token)
    except Exception:
        return {}
    if not commits:
        return {}
    # Commits come newest-first; the last one is the initial commit.
    authors, seen = [], set()
    for c in commits:
        for a in (getattr(c, "authors", None) or []):
            if a not in seen:
                seen.add(a)
                authors.append(a)
    initial = commits[-1]
    creator = (getattr(initial, "authors", None) or [None])[0]
    last_at = getattr(commits[0], "created_at", None)
    return {
        "uploader": creator,
        "uploaders": authors,
        "last_commit_at": last_at.isoformat() if last_at else None,
    }


def pull_dataset(repo_id, day_dir, revision, token):
    """Download one dataset into <day_dir>/<dataset-name> and summarize it."""
    from huggingface_hub import snapshot_download

    local_dir = Path(day_dir) / repo_id.split("/")[-1]
    print(f"Downloading {repo_id} -> {local_dir}")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(local_dir),
        token=token,
    )
    return build_summary(repo_id, path)


def build_report(summaries, failures, now, org, requested):
    """Build the aggregate report dict (totals first, then per-dataset list)."""
    agg_episodes = sum(x.get("total_episodes") or 0 for x in summaries)
    agg_frames = sum(x.get("total_frames") or 0 for x in summaries)
    agg_hours = round(sum(x.get("duration_hours") or 0 for x in summaries), 3)
    report = {
        "total_datasets": len(summaries),
        "total_episodes": agg_episodes,
        "total_frames": agg_frames,
        "total_hours": agg_hours,
        "pulled_at": now.isoformat(timespec="seconds"),
        "date": now.strftime("%y%m%d"),
        "org": org,
        "requested": requested,
        "count": len(summaries),
        "datasets": summaries,
    }
    if failures:
        report["failures"] = failures
    return report


def run_pull(repo_ids, out_dir, org, revision=None, token=None, now=None,
             log=print, progress=None, write_summary=True,
             meta_map=None, with_uploader=True):
    """Pull every repo in `repo_ids` into a per-day folder and write a report.

    `log(msg)` receives human-readable progress lines (same text as the CLI).
    `progress(done, total)` is called before and after each dataset so a UI can
    drive a progress bar. Returns (report_dict, out_path_or_None).
    """
    now = now or dt.datetime.now()
    day_dir = Path(out_dir) / now.strftime("%y%m%d")
    day_dir.mkdir(parents=True, exist_ok=True)

    summaries, failures = [], []
    total = len(repo_ids)
    if progress:
        progress(0, total)
    for i, repo_id in enumerate(repo_ids, 1):
        log(f"[{i}/{total}] {repo_id}")
        try:
            s = pull_dataset(repo_id, day_dir, revision, token)
            _enrich(s, repo_id, meta_map, with_uploader, token)
            summaries.append(s)
        except Exception as exc:  # keep pulling the rest if one fails
            log(f"ERROR pulling {repo_id}: {exc}")
            failures.append({"dataset_name": repo_id, "error": str(exc)})
        if progress:
            progress(i, total)

    report = build_report(summaries, failures, now, org, total)
    out_path = None
    if write_summary:
        out_path = day_dir / f"pull_result_{now.strftime('%y%m%d_%H%M')}.json"
        out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        log(f"Wrote summary -> {out_path}")
    try:
        history_path = append_history(report)
        log(f"Updated history -> {history_path}")
    except OSError as exc:
        log(f"WARN: could not update {HISTORY_FILE}: {exc}")
    return report, out_path


def _enrich(summary, repo_id, meta_map, with_uploader, token):
    """Attach last_modified (from meta_map) and uploader fields to a summary."""
    if meta_map and repo_id in meta_map:
        summary["last_modified"] = meta_map[repo_id]
    if with_uploader:
        summary.update(fetch_uploader(repo_id, token))
    return summary


def collect_stats(repo_ids, org, token=None, now=None, log=print, progress=None,
                  meta_map=None, with_uploader=True):
    """Build a report from meta/info.json only — no dataset files downloaded.

    Same report shape as run_pull (totals + per-dataset list), so a UI can show
    it in the exact same dashboard. Per-dataset entries have no local_dir but do
    carry last_modified + uploader when meta_map/with_uploader are supplied.
    """
    now = now or dt.datetime.now()
    summaries, failures = [], []
    total = len(repo_ids)
    if progress:
        progress(0, total)
    for i, repo_id in enumerate(repo_ids, 1):
        log(f"[{i}/{total}] {repo_id}")
        try:
            s = fetch_summary(repo_id, token)
            _enrich(s, repo_id, meta_map, with_uploader, token)
            summaries.append(s)
        except Exception as exc:
            log(f"ERROR reading {repo_id}: {exc}")
            failures.append({"dataset_name": repo_id, "error": str(exc)})
        if progress:
            progress(i, total)
    return build_report(summaries, failures, now, org, total)


def find_latest_report(out_dir):
    """Return the newest pull_result_*.json under out_dir/*/ (or None)."""
    files = sorted(Path(out_dir).glob("*/pull_result_*.json"))
    return files[-1] if files else None


# --------------------------------------------------------------------------- #
# Analytics helpers (pure functions over report dicts — used by the GUI)
# --------------------------------------------------------------------------- #
# The git-committed config file stores only hand-edited settings. Pull history is
# runtime data and stays in a local ignored file to avoid exposing or constantly
# changing dataset snapshots in commits.
CONFIG_FILE = str(Path(__file__).parent / "config.json")
HISTORY_FILE = str(Path(__file__).parent / "pull_history.local.json")
CHANGE_HISTORY_FILE = str(Path(__file__).parent / "hf_change_history.local.json")

# Per-dataset fields kept in the lightweight history (drop link/local_dir paths).
_HISTORY_DS_FIELDS = (
    "dataset_name", "total_episodes", "total_frames", "duration_hours",
    "fps", "robot_type", "total_tasks", "uploader", "last_modified",
)


def _load_json(path):
    """Read a JSON file; returns None if missing/corrupt."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def load_config(path=CONFIG_FILE):
    """Read the committed config; returns {} if missing/corrupt."""
    cfg = _load_json(path)
    return cfg if isinstance(cfg, dict) else {}


def load_uploader_names(path=CONFIG_FILE):
    """The hand-edited HF id -> Chinese name map from the config file."""
    return load_config(path).get("uploader_names", {}) or {}


def _trim_report(report):
    """A compact, path-free copy of a report for the local history file."""
    return {
        "pulled_at": report.get("pulled_at"),
        "date": report.get("date"),
        "org": report.get("org"),
        "total_datasets": report.get("total_datasets"),
        "total_episodes": report.get("total_episodes"),
        "total_frames": report.get("total_frames"),
        "total_hours": report.get("total_hours"),
        "datasets": [{k: d.get(k) for k in _HISTORY_DS_FIELDS}
                     for d in report.get("datasets", [])],
    }


def load_history_file(path=HISTORY_FILE):
    """Read local history; accepts the current list format and old dict format."""
    data = _load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return data.get("pull_history", []) or []
    return []


def load_hf_change_history(path=CHANGE_HISTORY_FILE):
    """Read local Hugging Face commit-diff history cache."""
    data = _load_json(path)
    if isinstance(data, dict):
        data.setdefault("version", 1)
        data.setdefault("repos", {})
        return data
    return {"version": 1, "repos": {}}


def save_hf_change_history(data, path=CHANGE_HISTORY_FILE):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return path


def append_history(report, path=HISTORY_FILE, legacy_config_file=CONFIG_FILE):
    """Fold a trimmed snapshot of `report` into the local history file.

    If the local file does not exist yet, seed it from the old config
    `pull_history` field for one-time backward compatibility.
    """
    hist = load_history_file(path)
    if not hist and not Path(path).exists():
        hist = load_config(legacy_config_file).get("pull_history", []) or []
    snap = _trim_report(report)
    hist = [h for h in hist if h.get("pulled_at") != snap.get("pulled_at")]
    hist.append(snap)
    hist.sort(key=lambda r: r.get("pulled_at") or "")
    Path(path).write_text(json.dumps(hist, indent=2, ensure_ascii=False) + "\n")
    return path


def load_history(out_dir, history_file=HISTORY_FILE, config_file=CONFIG_FILE):
    """Load pull snapshots oldest-first for trends / deltas.

    Merges the local history file, any legacy config["pull_history"], and any
    pulls/*/pull_result_*.json still on disk, deduping by pulled_at.
    """
    by_at = {}
    for r in load_config(config_file).get("pull_history", []) or []:
        by_at[r.get("pulled_at") or id(r)] = r
    for r in load_history_file(history_file):
        by_at[r.get("pulled_at") or id(r)] = r
    for f in sorted(Path(out_dir).glob("*/pull_result_*.json")):
        try:
            r = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        by_at.setdefault(r.get("pulled_at") or str(f), r)  # history file wins on ties
    history = list(by_at.values())
    history.sort(key=lambda r: r.get("pulled_at", ""))
    return history


def load_latest_local_report(out_dir, org=ORG):
    """Return the newest locally available report without network access.

    Priority: explicit pulls/*/pull_result_*.json, then local history, then a
    best-effort scan of downloaded pulls/*/<dataset>/meta/info.json directories.
    """
    latest = find_latest_report(out_dir)
    if latest:
        data = _load_json(latest)
        if isinstance(data, dict) and data.get("datasets"):
            return data, str(latest)

    history = load_history(out_dir)
    if history:
        report = history[-1]
        if isinstance(report, dict) and report.get("datasets"):
            return report, HISTORY_FILE

    summaries = []
    newest_by_leaf = {}
    for info in Path(out_dir).glob("*/*/meta/info.json"):
        dataset_dir = info.parent.parent
        prev = newest_by_leaf.get(dataset_dir.name)
        if prev is None or dataset_dir.stat().st_mtime > prev.stat().st_mtime:
            newest_by_leaf[dataset_dir.name] = dataset_dir
    for leaf, dataset_dir in sorted(newest_by_leaf.items()):
        summaries.append(build_summary(f"{org}/{leaf}", str(dataset_dir)))
    if summaries:
        latest_time = max((Path(s["local_dir"]).stat().st_mtime for s in summaries), default=None)
        now = dt.datetime.fromtimestamp(latest_time) if latest_time else dt.datetime.now()
        return build_report(summaries, [], now, org, len(summaries)), str(Path(out_dir))
    return None, None


def daily_series(history):
    """Collapse history to one snapshot per day (the day's last pull).

    `total_hours` is the absolute library total at that snapshot (already
    cumulative). `new_hours` is the day-over-day increase (this day's total
    minus the previous pulled day's total); the first day's `new_hours` equals
    its total. Returns a date-sorted list of {date, total_hours, new_hours,
    total_episodes, total_frames, total_datasets} for trend charts.
    """
    by_day = {}
    for r in history:  # history is oldest-first, so later pulls overwrite
        by_day[r.get("date", "")] = r
    series = []
    prev_total = None
    for date in sorted(k for k in by_day if k):
        r = by_day[date]
        total = r.get("total_hours", 0) or 0
        new_hours = total if prev_total is None else round(total - prev_total, 3)
        prev_total = total
        series.append({
            "date": date,
            "total_hours": total,
            "new_hours": new_hours,
            "total_episodes": r.get("total_episodes", 0) or 0,
            "total_frames": r.get("total_frames", 0) or 0,
            "total_datasets": r.get("total_datasets", 0) or 0,
        })
    return series


def daily_group_series(history, key_fn):
    """Per-group daily positive growth from the last snapshot of each day.

    Returns rows sorted by date oldest-first and hours descending within each day:
    {date, group, hours, episodes, datasets}. The first detailed day counts
    each dataset's full duration as that day's contribution. If the previous day
    has only aggregate totals and no dataset details, attribution for the next
    day is skipped because per-group growth cannot be derived safely.
    """
    by_day = {}
    for r in history:
        by_day[r.get("date", "")] = r
    rows = []
    prev_report = None
    prev = {}
    for date in sorted(k for k in by_day if k):
        report = by_day[date]
        datasets = report.get("datasets", []) or []
        aggregate_only_prior = bool(prev_report) and not prev
        if not aggregate_only_prior:
            groups = {}
            for dataset in datasets:
                name = dataset.get("dataset_name")
                if not name:
                    continue
                prior = prev.get(name)
                d_hours = round((dataset.get("duration_hours") or 0)
                                - (prior.get("duration_hours") or 0 if prior else 0), 3)
                d_episodes = (dataset.get("total_episodes") or 0) \
                    - (prior.get("total_episodes") or 0 if prior else 0)
                hours = max(0, d_hours)
                episodes = max(0, d_episodes)
                if hours <= 0 and episodes <= 0:
                    continue
                key = key_fn(dataset) or "—"
                group = groups.setdefault(
                    key, {"date": date, "group": key, "hours": 0.0,
                          "episodes": 0, "datasets": 0})
                group["hours"] += hours
                group["episodes"] += episodes
                group["datasets"] += 1
            day_rows = sorted(groups.values(), key=lambda g: g["hours"], reverse=True)
            for row in day_rows:
                row["hours"] = round(row["hours"], 3)
            rows.extend(day_rows)
        prev_report = report
        prev = {d.get("dataset_name"): d for d in datasets if d.get("dataset_name")}
    return rows


def daily_uploader_series(history):
    """Backward-compatible per-uploader daily growth helper."""
    return daily_group_series(history, lambda d: d.get("uploader") or "")


def _hf_update_date(dataset):
    """YYMMDD from a dataset's Hugging Face last_modified timestamp."""
    value = dataset.get("last_modified")
    if not value:
        return ""
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%y%m%d")
    except (TypeError, ValueError):
        return str(value)[:10].replace("-", "")[2:]


def hf_daily_group_series(datasets, key_fn):
    """Group current datasets by Hugging Face update day and dimension.

    Uses each dataset's `last_modified` from the Hub rather than local pull
    snapshots. Returns {date, group, hours, episodes, datasets} sorted by date
    oldest-first and hours descending within each date.
    """
    groups = {}
    for dataset in datasets or []:
        date = _hf_update_date(dataset)
        if not date:
            continue
        key = key_fn(dataset) or "—"
        group = groups.setdefault(
            (date, key), {"date": date, "group": key, "hours": 0.0,
                          "episodes": 0, "datasets": 0})
        group["hours"] += dataset.get("duration_hours") or 0
        group["episodes"] += dataset.get("total_episodes") or 0
        group["datasets"] += 1
    rows = list(groups.values())
    for row in rows:
        row["hours"] = round(row["hours"], 3)
    rows.sort(key=lambda row: (row["date"], -row["hours"], row["group"]))
    return rows


def hf_latest_update_date(datasets):
    """Newest Hugging Face last_modified date (YYMMDD) in current datasets."""
    dates = [_hf_update_date(dataset) for dataset in datasets or []]
    return max((date for date in dates if date), default="")


def hf_update_totals(datasets, date=None):
    """Totals for datasets whose Hugging Face update day equals `date`.

    If date is omitted, uses the newest HF update day in the dataset list.
    """
    date = date or hf_latest_update_date(datasets)
    totals = {"date": date, "hours": 0.0, "episodes": 0, "datasets": 0}
    if not date:
        return totals
    for dataset in datasets or []:
        if _hf_update_date(dataset) != date:
            continue
        totals["hours"] += dataset.get("duration_hours") or 0
        totals["episodes"] += dataset.get("total_episodes") or 0
        totals["datasets"] += 1
    totals["hours"] = round(totals["hours"], 2)
    return totals


def hf_update_group_totals(datasets, key_fn, date=None):
    """Group totals for one Hugging Face update day."""
    date = date or hf_latest_update_date(datasets)
    groups = {}
    if not date:
        return []
    for dataset in datasets or []:
        if _hf_update_date(dataset) != date:
            continue
        key = key_fn(dataset) or "—"
        group = groups.setdefault(
            key, {"date": date, "group": key, "hours": 0.0,
                  "episodes": 0, "datasets": 0})
        group["hours"] += dataset.get("duration_hours") or 0
        group["episodes"] += dataset.get("total_episodes") or 0
        group["datasets"] += 1
    rows = list(groups.values())
    for row in rows:
        row["hours"] = round(row["hours"], 2)
    rows.sort(key=lambda row: row["hours"], reverse=True)
    return rows


def hf_daily_group_delta_series(current_report, history, key_fn):
    """Incremental HF update rows grouped by last_modified day and dimension.

    Existing datasets contribute only their positive growth versus the previous
    detailed baseline snapshot. New datasets contribute their full current size.
    If no detailed baseline exists, returns [] rather than treating old data as
    new, because the true increment cannot be known safely.
    """
    prior = find_baseline(current_report, history)
    if not prior or not prior.get("datasets"):
        return []
    deltas = compute_deltas(current_report, history)
    groups = {}
    for dataset in current_report.get("datasets", []) if current_report else []:
        name = dataset.get("dataset_name")
        if not name:
            continue
        delta = deltas.get(name, {})
        hours = max(0, delta.get("d_hours", 0) or 0)
        episodes = max(0, delta.get("d_episodes", 0) or 0)
        if hours <= 0 and episodes <= 0:
            continue
        date = _hf_update_date(dataset)
        if not date:
            continue
        key = key_fn(dataset) or "—"
        group = groups.setdefault(
            (date, key), {"date": date, "group": key, "hours": 0.0,
                          "episodes": 0, "datasets": 0})
        group["hours"] += hours
        group["episodes"] += episodes
        group["datasets"] += 1
    rows = list(groups.values())
    for row in rows:
        row["hours"] = round(row["hours"], 3)
    rows.sort(key=lambda row: (row["date"], -row["hours"], row["group"]))
    return rows


def hf_update_delta_totals(current_report, history, date=None):
    """Incremental totals for one HF update day; defaults to newest delta day."""
    rows = hf_daily_group_delta_series(current_report, history, lambda dataset: "__all__")
    date = date or max((row["date"] for row in rows), default="")
    totals = {"date": date, "hours": 0.0, "episodes": 0, "datasets": 0}
    for row in rows:
        if row["date"] != date:
            continue
        totals["hours"] += row["hours"]
        totals["episodes"] += row["episodes"]
        totals["datasets"] += row["datasets"]
    totals["hours"] = round(totals["hours"], 2)
    return totals


def hf_update_delta_group_totals(current_report, history, key_fn, date=None):
    """Incremental group totals for one HF update day."""
    rows = hf_daily_group_delta_series(current_report, history, key_fn)
    date = date or max((row["date"] for row in rows), default="")
    out = [row for row in rows if row["date"] == date]
    out.sort(key=lambda row: row["hours"], reverse=True)
    return out


def _commit_value(commit, *names):
    for name in names:
        if isinstance(commit, dict) and commit.get(name) is not None:
            return commit.get(name)
        if hasattr(commit, name):
            value = getattr(commit, name)
            if value is not None:
                return value
    return None


def _commit_created_at(commit):
    value = _commit_value(commit, "created_at", "createdAt", "date")
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value) if value else ""


def _yymmdd_from_iso(value):
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).strftime("%y%m%d")
    except (TypeError, ValueError):
        text = str(value or "")[:10]
        return text.replace("-", "")[2:] if len(text) >= 10 else ""


def _commit_id(commit):
    return _commit_value(commit, "commit_id", "commitId", "oid", "id") or ""


def _summary_from_info(repo_id, info):
    summary = {"dataset_name": repo_id, "link": HF_DATASET_URL.format(repo_id=repo_id)}
    _apply_info(summary, info or {})
    return summary


def _fetch_info_at_revision(repo_id, revision, token=None):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=repo_id,
        filename="meta/info.json",
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _dataset_meta_for_change(dataset):
    return {
        "dataset_name": dataset.get("dataset_name"),
        "uploader": dataset.get("uploader"),
        "robot_type": dataset.get("robot_type"),
        "total_tasks": dataset.get("total_tasks"),
        "last_modified": dataset.get("last_modified"),
    }


def build_hf_change_rows(repo_id, dataset, token=None):
    """Build true HF data-growth rows from commit history and meta/info.json diffs."""
    from huggingface_hub import HfApi

    api = HfApi()
    try:
        commits = api.list_repo_commits(repo_id=repo_id, repo_type="dataset", token=token)
    except TypeError:
        commits = api.list_repo_commits(repo_id=repo_id, repo_type="dataset")
    commits = sorted(commits, key=lambda commit: _commit_created_at(commit) or "")
    prev = None
    rows = []
    meta = _dataset_meta_for_change(dataset)
    for commit in commits:
        commit_id = _commit_id(commit)
        created_at = _commit_created_at(commit)
        if not commit_id or not created_at:
            continue
        try:
            summary = _summary_from_info(repo_id, _fetch_info_at_revision(repo_id, commit_id, token))
        except Exception:
            continue
        frames = summary.get("total_frames") or 0
        episodes = summary.get("total_episodes") or 0
        hours = summary.get("duration_hours") or 0
        prev_frames = prev.get("total_frames") or 0 if prev else 0
        prev_episodes = prev.get("total_episodes") or 0 if prev else 0
        prev_hours = prev.get("duration_hours") or 0 if prev else 0
        d_frames = frames - prev_frames
        d_episodes = episodes - prev_episodes
        d_hours = round(hours - prev_hours, 3)
        prev = summary
        if d_frames <= 0 and d_episodes <= 0 and d_hours <= 0:
            continue
        date = _yymmdd_from_iso(created_at)
        if not date:
            continue
        row = {
            **meta,
            "date": date,
            "commit_id": commit_id,
            "created_at": created_at,
            "hours": max(0, d_hours),
            "episodes": max(0, d_episodes),
            "frames": max(0, d_frames),
        }
        rows.append(row)
    return rows, commits[-1] if commits else None


def update_hf_change_history(report, token=None, path=CHANGE_HISTORY_FILE, log=None, progress=None):
    """Update local HF commit-diff cache for datasets in `report`."""
    cache = load_hf_change_history(path)
    repos = cache.setdefault("repos", {})
    datasets = report.get("datasets", []) if report else []
    total = len(datasets)
    for index, dataset in enumerate(datasets, 1):
        repo_id = dataset.get("dataset_name")
        if not repo_id:
            continue
        cached = repos.get(repo_id, {})
        if cached.get("last_modified") == dataset.get("last_modified") and cached.get("changes"):
            if progress:
                progress(index, total)
            continue
        if log:
            log(f"HF 变更历史 [{index}/{total}] {repo_id}")
        try:
            changes, head = build_hf_change_rows(repo_id, dataset, token)
        except Exception as exc:
            cached["error"] = str(exc)
            repos[repo_id] = cached
            if progress:
                progress(index, total)
            continue
        repos[repo_id] = {
            "dataset_name": repo_id,
            "last_modified": dataset.get("last_modified"),
            "head_commit": _commit_id(head) if head else "",
            "updated_at": dt.datetime.now().isoformat(timespec="seconds"),
            "changes": changes,
        }
        if progress:
            progress(index, total)
    save_hf_change_history(cache, path)
    return cache


def hf_change_rows(change_history):
    rows = []
    for repo in (change_history or {}).get("repos", {}).values():
        rows.extend(repo.get("changes", []) or [])
    rows.sort(key=lambda row: (row.get("date") or "", row.get("created_at") or ""))
    return rows


def hf_change_daily_group_series(change_history, key_fn):
    groups = {}
    for row in hf_change_rows(change_history):
        date = row.get("date") or ""
        if not date:
            continue
        key = key_fn(row) or "—"
        group = groups.setdefault(
            (date, key), {"date": date, "group": key, "hours": 0.0,
                          "episodes": 0, "datasets": 0, "_dataset_names": set()})
        group["hours"] += row.get("hours") or 0
        group["episodes"] += row.get("episodes") or 0
        group["_dataset_names"].add(row.get("dataset_name") or row.get("commit_id") or id(row))
    out = list(groups.values())
    for row in out:
        row["hours"] = round(row["hours"], 3)
        row["datasets"] = len(row.pop("_dataset_names"))
    out.sort(key=lambda row: (row["date"], -row["hours"], row["group"]))
    return out


def hf_change_latest_date(change_history):
    return max((row.get("date") for row in hf_change_rows(change_history) if row.get("date")), default="")


def hf_change_totals(change_history, date=None):
    date = date or hf_change_latest_date(change_history)
    totals = {"date": date, "hours": 0.0, "episodes": 0, "datasets": 0}
    dataset_names = set()
    for row in hf_change_rows(change_history):
        if row.get("date") != date:
            continue
        totals["hours"] += row.get("hours") or 0
        totals["episodes"] += row.get("episodes") or 0
        dataset_names.add(row.get("dataset_name") or row.get("commit_id") or id(row))
    totals["hours"] = round(totals["hours"], 2)
    totals["datasets"] = len(dataset_names)
    return totals


def hf_change_group_totals(change_history, key_fn, date=None):
    date = date or hf_change_latest_date(change_history)
    rows = [row for row in hf_change_daily_group_series(change_history, key_fn)
            if row.get("date") == date]
    rows.sort(key=lambda row: row["hours"], reverse=True)
    return rows


def find_baseline(current_report, history):
    """Return the snapshot to diff `current_report` against: the last pull of the
    most recent *earlier day*.

    "今日新增" is measured against the previous pull DAY, not merely the previous
    pull — so multiple pulls on the same day all compare back to that earlier day.
    `history` is oldest-first, so the last entry whose date precedes the current
    report's date is that day's final pull. Returns None when no earlier day
    exists (the current report is the first ever).
    """
    cur_date = current_report.get("date") or ""
    prior = None
    for r in history:  # oldest-first; last match = newest earlier-day pull
        rd = r.get("date") or ""
        if rd and rd < cur_date:
            prior = r
    return prior


def compute_deltas(current_report, history):
    """Per-dataset growth of current_report vs the previous pull day's snapshot.

    Baseline = find_baseline(current_report, history). Returns
    {dataset_name: {d_episodes, d_frames, d_hours, is_new}}. With no earlier-day
    snapshot every dataset is marked is_new with its full totals as the delta.
    """
    prior = find_baseline(current_report, history)
    prev = {d["dataset_name"]: d for d in (prior.get("datasets", []) if prior else [])}
    # An aggregate-only baseline (backfilled history that has totals but no
    # per-dataset detail) can't attribute growth to individual datasets. Report
    # zero per-dataset deltas there instead of pretending everything is new; the
    # KPI falls back to the aggregate total difference (see main_app._new_totals).
    aggregate_only = bool(prior) and not prev
    deltas = {}
    for d in current_report.get("datasets", []):
        name = d["dataset_name"]
        if aggregate_only:
            deltas[name] = {"d_episodes": 0, "d_frames": 0, "d_hours": 0,
                            "is_new": False}
            continue
        p = prev.get(name)
        deltas[name] = {
            "d_episodes": (d.get("total_episodes") or 0) - (p.get("total_episodes") or 0 if p else 0),
            "d_frames": (d.get("total_frames") or 0) - (p.get("total_frames") or 0 if p else 0),
            "d_hours": round((d.get("duration_hours") or 0) - (p.get("duration_hours") or 0 if p else 0), 3),
            "is_new": p is None,
        }
    return deltas


def task_prefix(dataset_name):
    """Derive a task label: drop the owner and a trailing -MMDD date suffix.

    e.g. 'TacVerse/taccap-g1-pepper-0703' -> 'taccap-g1-pepper'.
    """
    name = dataset_name.split("/")[-1]
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 4:
        return parts[0]
    return name


def rollup(datasets, key_fn):
    """Group datasets by key_fn(d) and sum count/episodes/frames/hours.

    Returns a list of {group, count, episodes, frames, hours, pct_hours} sorted
    by hours descending. pct_hours is each group's share of total hours.
    """
    groups = {}
    for d in datasets:
        key = key_fn(d) or "—"
        g = groups.setdefault(
            key, {"group": key, "count": 0, "episodes": 0, "frames": 0, "hours": 0.0})
        g["count"] += 1
        g["episodes"] += d.get("total_episodes") or 0
        g["frames"] += d.get("total_frames") or 0
        g["hours"] += d.get("duration_hours") or 0
    total_hours = sum(g["hours"] for g in groups.values()) or 1
    rows = sorted(groups.values(), key=lambda g: g["hours"], reverse=True)
    for g in rows:
        g["hours"] = round(g["hours"], 3)
        g["pct_hours"] = round(100 * g["hours"] / total_hours, 1)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pull HF datasets into date-stamped folders."
    )
    parser.add_argument(
        "--org",
        default=ORG,
        help=f"Pull every dataset under this org/user (default: {ORG})",
    )
    parser.add_argument(
        "--repo-id",
        action="append",
        dest="repo_ids",
        metavar="OWNER/NAME",
        help="Pull only these datasets (repeat); overrides --org discovery",
    )
    parser.add_argument(
        "--out-dir",
        default="pulls",
        help="Base directory; a per-day <YYMMDD> subfolder is created inside",
    )
    parser.add_argument("--revision", default=None, help="Branch, tag, or commit")
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF access token (defaults to $HF_TOKEN or the cached login token)",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Download only; skip writing the summary file",
    )
    args = parser.parse_args()

    normalize_proxy_env()

    try:
        import huggingface_hub  # noqa: F401  (imported for the clear error below)
    except ImportError:
        sys.exit(
            "huggingface_hub is not installed.\n"
            "Install it with:  pip install huggingface_hub"
        )

    meta_map = None
    if args.repo_ids:
        repo_ids = args.repo_ids
    else:
        print(f"Discovering datasets under '{args.org}' ...")
        meta = discover_datasets_meta(args.org, args.token)
        repo_ids = [m["id"] for m in meta]
        meta_map = {m["id"]: m["last_modified"] for m in meta}
        print(f"Found {len(repo_ids)} datasets.")
    if not repo_ids:
        sys.exit(f"No datasets to pull (org '{args.org}' returned nothing).")

    report, _ = run_pull(
        repo_ids,
        out_dir=args.out_dir,
        org=args.org,
        revision=args.revision,
        token=args.token,
        write_summary=not args.no_summary,
        meta_map=meta_map,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if report.get("failures") else 0


if __name__ == "__main__":
    raise SystemExit(main())
