"""Read raw records from local files: JSONL, JSON, CSV/TSV, Parquet.

CSV/TSV comes in two shapes, detected from the header:

- **one row per record**: a `text` column plus an optional `spans` column
  holding a JSON list of `{start, end, label}` (and optional `id`,
  `language`).
- **one row per span**: `text`, `label`, `start`, `end` columns (plus
  optional `id`). Rows sharing an `id` (or, without one, the same `text`)
  are grouped into one record; a row with an empty label contributes just
  the text.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

FORMATS = ("jsonl", "json", "csv", "tsv", "parquet")


def infer_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower().lstrip(".")
    if suffix in ("jsonl", "ndjson"):
        return "jsonl"
    if suffix in FORMATS:
        return suffix
    raise ValueError(
        f"can't infer format from {path!r}; pass format= one of {FORMATS}"
    )


def read_records(path: str | Path, format: str | None = None) -> list[dict]:
    path = Path(path)
    fmt = format or infer_format(path)
    if fmt == "jsonl":
        return _read_jsonl(path)
    if fmt == "json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            # {"data": [...]} / {"records": [...]} wrappers
            for key in ("data", "records", "examples", "rows"):
                if isinstance(data.get(key), list):
                    return data[key]
            raise ValueError(f"{path}: JSON object has no list under data/records/examples/rows")
        return list(data)
    if fmt in ("csv", "tsv"):
        return _read_csv(path, delimiter="\t" if fmt == "tsv" else ",")
    if fmt == "parquet":
        import pyarrow.parquet as pq  # installed with `datasets`

        return pq.read_table(path).to_pylist()
    raise ValueError(f"unknown format {fmt!r}; expected one of {FORMATS}")


def _read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: invalid JSON ({e.msg})") from e
    return out


def _read_csv(path: Path, *, delimiter: str) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f, delimiter=delimiter))
    if not rows:
        return []
    cols = set(rows[0])
    if "text" not in cols:
        raise ValueError(f"{path}: CSV needs a `text` column (found {sorted(cols)})")

    if {"label", "start", "end"} <= cols and "spans" not in cols:
        return _group_span_rows(rows)

    out = []
    for row in rows:
        rec: dict = dict(row)
        raw = row.get("spans")
        rec["spans"] = json.loads(raw) if raw else []
        out.append(rec)
    return out


def _group_span_rows(rows: list[dict]) -> list[dict]:
    records: dict[str, dict] = {}
    for row in rows:
        key = row.get("id") or row["text"]
        rec = records.get(key)
        if rec is None:
            rec = {k: v for k, v in row.items() if k not in ("label", "start", "end")}
            rec["spans"] = []
            records[key] = rec
        if row.get("label"):
            rec["spans"].append(
                {"label": row["label"], "start": int(row["start"]), "end": int(row["end"])}
            )
    return list(records.values())
