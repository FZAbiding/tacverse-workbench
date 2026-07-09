"""Read-only access to a dataset's viewer-authored language annotations.

The viewer (xense_lerobot_viewer) persists its "Language annotations" (Prompt
atoms) into `<dataset>/meta/lerobot_annotations.json` (lerobot v3.1 schema).
workbench only *reads* this file to surface Prompts in the dashboard — it never
writes it (editing stays in the viewer). Keep this module Qt-free so it can be
reused by later post-processing.

File shape::

    {
      "version": 2,
      "episodes": { "<ep_index>": { "atoms": [Atom, ...] }, ... },
      "updated_at": "..."
    }

Each Atom::

    { "role", "content", "style", "timestamp", "camera", "tool_calls" }
"""

import json
from pathlib import Path

ANNOTATIONS_REL = Path("meta") / "lerobot_annotations.json"

# style -> Chinese label (matches the viewer's own grouping semantics).
STYLE_CN = {
    "task_aug": "任务改写",
    "subtask": "子任务",
    "plan": "计划",
    "memory": "记忆",
    "interjection": "插话",
    "vqa": "视觉问答",
    None: "语音",  # style=null: tool-call-only (speech) atoms
}

# Display order: persistent styles first, then events, then speech/other.
STYLE_ORDER = ["task_aug", "subtask", "plan", "memory", "interjection", "vqa", None]

# Event styles carry a meaningful per-frame timestamp worth showing.
_EVENT_STYLES = {"interjection", "vqa", None}


def resolve_path(dataset, out_dir="pulls"):
    """Locate a dataset's annotations file on disk, or None.

    Resolution order:
      1. `dataset["local_dir"]/meta/lerobot_annotations.json` (pulled records).
      2. newest (by mtime) `pulls/*/<leaf>/meta/lerobot_annotations.json`.
    """
    local_dir = (dataset or {}).get("local_dir")
    if local_dir:
        p = Path(local_dir) / ANNOTATIONS_REL
        if p.is_file():
            return p

    name = (dataset or {}).get("dataset_name") or ""
    leaf = name.split("/")[-1]
    if not leaf:
        return None
    candidates = [
        p for p in Path(out_dir).glob(f"*/{leaf}/{ANNOTATIONS_REL}") if p.is_file()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load(path):
    """Parse the annotations file. Returns (doc, error).

    `doc` is `{"episodes": {...}, "updated_at": str|None}` (empty on any
    failure); `error` is a human-readable string when parsing failed, else None.
    """
    empty = {"episodes": {}, "updated_at": None}
    if not path:
        return empty, None
    try:
        raw = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        return empty, f"读取失败: {exc}"
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as exc:
        return empty, f"标注文件解析失败: {exc}"
    episodes = parsed.get("episodes") if isinstance(parsed, dict) else None
    if not isinstance(episodes, dict):
        episodes = {}
    return {"episodes": episodes, "updated_at": (parsed or {}).get("updated_at")}, None


def _atoms(entry):
    atoms = (entry or {}).get("atoms")
    return atoms if isinstance(atoms, list) else []


def episodes_with_atoms(doc):
    """Episode keys that have at least one atom, sorted numerically."""
    keys = [k for k, entry in doc.get("episodes", {}).items() if _atoms(entry)]

    def sort_key(k):
        try:
            return (0, int(k))
        except (ValueError, TypeError):
            return (1, k)

    return sorted(keys, key=sort_key)


def atoms_for_episode(doc, episode):
    return _atoms(doc.get("episodes", {}).get(str(episode)))


def group_by_style(atoms):
    """Group atoms by style into an ordered list of (style, [atoms]).

    Only non-empty groups are returned, in STYLE_ORDER. Any unknown style is
    appended after the known ones.
    """
    buckets = {}
    for a in atoms:
        buckets.setdefault(a.get("style"), []).append(a)
    ordered = [(s, buckets.pop(s)) for s in STYLE_ORDER if s in buckets]
    ordered += [(s, buckets[s]) for s in buckets]  # any unknown styles left
    return ordered


def style_label(style):
    return STYLE_CN.get(style, style if style else "语音")


def is_event_style(style):
    return style in _EVENT_STYLES


def atom_text(atom):
    """Best-effort display text for one atom (content or a speech summary)."""
    content = atom.get("content")
    if content:
        return str(content)
    calls = atom.get("tool_calls")
    if isinstance(calls, list) and calls:
        names = [
            (c.get("function") or {}).get("name")
            for c in calls
            if isinstance(c, dict)
        ]
        names = [n for n in names if n]
        if names:
            return "· " + ", ".join(names)
    return "(空)"
