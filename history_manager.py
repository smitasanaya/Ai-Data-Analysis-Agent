"""
Phase D (new): Search History
Persists past questions/answers to a local JSON file so history survives
across app restarts, without needing a full database. Each dataset gets
its own history bucket (keyed by a hash of its columns + row count) so
history doesn't mix between unrelated files.
"""

import os
import json
import hashlib
from datetime import datetime

HISTORY_FILE = "search_history.json"


def _dataset_key(schema_info: dict) -> str:
    """Build a stable key for a dataset based on its shape/columns."""
    col_names = ",".join(c["name"] for c in schema_info["columns"])
    raw = f"{schema_info['n_rows']}_{schema_info['n_cols']}_{col_names}"
    return hashlib.md5(raw.encode()).hexdigest()[:10]


def _load_all() -> dict:
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_all(data: dict):
    with open(HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def add_entry(schema_info: dict, question: str, answer: str):
    """Record a question/answer pair to this dataset's history."""
    key = _dataset_key(schema_info)
    data = _load_all()
    data.setdefault(key, [])
    data[key].insert(0, {  # newest first
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "question": question,
        "answer": answer[:500],  # keep entries reasonably sized
    })
    data[key] = data[key][:50]  # cap history length per dataset
    _save_all(data)


def get_history(schema_info: dict) -> list:
    """Return this dataset's history, newest first."""
    key = _dataset_key(schema_info)
    return _load_all().get(key, [])


def clear_history(schema_info: dict):
    """Clear history for this dataset only."""
    key = _dataset_key(schema_info)
    data = _load_all()
    data.pop(key, None)
    _save_all(data)
