"""Read-only access to a dataset's native LeRobot task instructions.

Every LeRobot dataset ships the natural-language task string(s) it was
recorded against in `<dataset>/meta/tasks.parquet` (columns `task_index` +
`task`). This is the base "prompt" — it exists as soon as the dataset is
pulled, with no dependency on the viewer or its language-annotation editor.

Kept Qt-free and pandas-free (pyarrow only, imported lazily) so it stays light
and reusable by later post-processing. Mirrors annotations_reader's resolution
strategy so the dashboard can locate files the same way for both sources.
"""

from pathlib import Path

TASKS_REL = Path("meta") / "tasks.parquet"


def resolve_path(dataset, out_dir="pulls"):
    """Locate a dataset's tasks.parquet on disk, or None.

    Resolution order:
      1. `dataset["local_dir"]/meta/tasks.parquet` (pulled records).
      2. newest (by mtime) `pulls/*/<leaf>/meta/tasks.parquet`.
    """
    local_dir = (dataset or {}).get("local_dir")
    if local_dir:
        p = Path(local_dir) / TASKS_REL
        if p.is_file():
            return p

    name = (dataset or {}).get("dataset_name") or ""
    leaf = name.split("/")[-1]
    if not leaf:
        return None
    candidates = [
        p for p in Path(out_dir).glob(f"*/{leaf}/{TASKS_REL}") if p.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load(path):
    """Read task rows from tasks.parquet. Returns (tasks, error).

    `tasks` is a list of {"index": int, "task": str} sorted by index (empty on
    any failure); `error` is a human-readable string when reading failed, else
    None. pyarrow is imported here (not at module load) so a missing/broken
    pyarrow degrades to a friendly message instead of breaking the app import.
    """
    if not path:
        return [], None
    try:
        import pyarrow.parquet as pq
    except Exception as exc:  # pyarrow missing or broken
        return [], f"缺少 pyarrow，无法读取 task: {exc}"
    try:
        table = pq.read_table(path)
    except Exception as exc:
        return [], f"tasks.parquet 解析失败: {exc}"

    cols = table.to_pydict()
    texts = cols.get("task")
    if not isinstance(texts, list):
        # Unexpected schema (no `task` column) — nothing to show, not an error.
        return [], None
    idxs = cols.get("task_index")
    if not isinstance(idxs, list) or len(idxs) != len(texts):
        idxs = list(range(len(texts)))

    rows = []
    for i, text in enumerate(texts):
        if text is None:
            continue
        idx = idxs[i]
        rows.append({"index": idx if isinstance(idx, int) else i, "task": str(text)})
    rows.sort(key=lambda r: r["index"] if isinstance(r["index"], int) else 0)
    return rows, None
