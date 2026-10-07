"""Small file helpers shared by fixtures, runner, report, ensemble and notebooks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

# Values of the "gold" key in a fixtures sidecar. Fixtures written from your
# own text or files have no gold spans (GOLD_NONE). Fixtures whose gold spans
# are LLM silver labels say GOLD_SILVER. A sidecar without the key belongs to
# a dataset sample with real gold. fixtures, silver, report and nb all read
# these names so the value is spelled in one place.
GOLD_NONE = "none"
GOLD_SILVER = "silver"


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_jsonl(path: str | Path) -> list[dict]:
    return list(iter_jsonl(path))


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    """Write one JSON object per line in UTF-8 and return the row count.

    A row holding a lone UTF-16 surrogate (text read from a JSON file with a
    stray "\\ud83d" escape) cannot be written as UTF-8. That row is written
    with ASCII escapes instead, so it reads back exactly as it was.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            line = json.dumps(row, ensure_ascii=False)
            try:
                line.encode("utf-8")
            except UnicodeEncodeError:
                line = json.dumps(row)
            f.write(line + "\n")
            n += 1
    return n


def index_by_id(records: Iterable[dict], key: str = "id") -> dict[str, dict]:
    return {r[key]: r for r in records}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def meta_path(fixtures: str | Path) -> Path:
    """Sidecar metadata written next to a fixtures file by `materialize`."""
    p = Path(fixtures)
    return p.with_name(p.name + ".meta.json")


def read_meta(fixtures: str | Path) -> dict | None:
    p = meta_path(fixtures)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
