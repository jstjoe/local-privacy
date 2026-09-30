"""Materialize a deterministic sample from any registered dataset into a
fixtures file, and validate fixtures files.

Fixture record (one JSON object per line; see `datasets.base`):

    {"id": str, "text": str, "language": iso2|null,
     "gold_spans": [{"label": coarse, "fine_label": canonical,
                     "raw_label": str, "start": int, "end": int}]}

`materialize` also writes `<fixtures>.meta.json` recording the dataset,
sample parameters and annotated labels, so the runner and report know what
the fixtures contain without being told again.

Examples:

    python -m opf_eval.fixtures --dataset openpii_nano --out data/nano_100.jsonl --n 100
    python -m opf_eval.fixtures --file my_data.jsonl --out data/mine.jsonl     # all records
    python -m opf_eval.fixtures validate data/mine.jsonl
    python -m opf_eval.fixtures list
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import taxonomy
from .datasets import (
    DATASETS,
    DEFAULT_DATASET,
    annotated_labels,
    describe as describe_datasets,
    get as get_dataset_config,
    load_raw,
    register_dataset,
)
from .io import file_sha256, meta_path, read_meta, write_jsonl

META_VERSION = 2


def materialize(
    out_path: Path,
    n: int | None,
    *,
    dataset: str = DEFAULT_DATASET,
    split: str | None = None,
    seed: int = 42,
) -> int:
    """Sample `n` records from the named dataset and write fixture JSONL.

    `n=None` writes every record in source order; otherwise records are
    shuffled with `seed` and the first `n` kept (same sample every time).
    Records with no usable text are skipped, and gold spans whose offsets
    fall outside the text are dropped (counted in the meta sidecar).
    Returns the number of records written.
    """
    out_path = Path(out_path)
    cfg = get_dataset_config(dataset)
    raw = load_raw(cfg, split)
    indices = list(range(len(raw)))
    if n is not None:
        rng = random.Random(seed)
        rng.shuffle(indices)
        indices = indices[: min(n, len(indices))]

    dropped = Counter()

    def records():
        for rec in cfg.loader(raw[int(i)] for i in indices):
            kept = []
            for s in rec["gold_spans"]:
                if 0 <= s["start"] < s["end"] <= len(rec["text"]):
                    kept.append(s)
                else:
                    dropped["out_of_bounds"] += 1
            rec["gold_spans"] = kept
            yield rec

    rows = list(records())
    written = write_jsonl(out_path, rows)
    meta = {
        "version": META_VERSION,
        "dataset": cfg.name,
        "source": cfg.source,
        "split": split or (None if cfg.path else cfg.default_split),
        "hf_revision": cfg.hf_revision,
        "vocab_key": cfg.vocab_key,
        "seed": seed if n is not None else None,
        "n_requested": n,
        "n_written": written,
        "labels": sorted(annotated_labels(cfg, rows, level="fine")),
        "dropped_spans": dict(dropped),
        "sha256": file_sha256(out_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path(out_path).write_text(json.dumps(meta, indent=2))
    return written


def ensure_fixtures(
    out_path: Path,
    n: int | None,
    *,
    dataset: str = DEFAULT_DATASET,
    split: str | None = None,
    seed: int = 42,
) -> tuple[Path, bool]:
    """Materialize unless `out_path` already holds exactly this sample.

    Reuses the file only when its meta sidecar matches (dataset, split, n,
    seed) and the file is unchanged since it was written. Returns
    (path, reused).
    """
    out_path = Path(out_path)
    meta = read_meta(out_path)
    cfg = get_dataset_config(dataset)
    want_split = split or (None if cfg.path else cfg.default_split)
    if (
        out_path.exists()
        and meta is not None
        and meta.get("dataset") == dataset
        and meta.get("split") == want_split
        and meta.get("n_requested") == n
        and meta.get("seed") == (seed if n is not None else None)
        and meta.get("sha256") == file_sha256(out_path)
    ):
        return out_path, True
    materialize(out_path, n, dataset=dataset, split=split, seed=seed)
    return out_path, False


# ------------------------------------------------------------- validation


@dataclass
class FixtureReport:
    path: Path
    n_records: int = 0
    n_spans: int = 0
    labels: Counter = field(default_factory=Counter)
    languages: Counter = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [
            f"{self.path}: {self.n_records} records, {self.n_spans} gold spans"
            f" — {'OK' if self.ok else f'{len(self.errors)} error(s)'}"
            + (f", {len(self.warnings)} warning(s)" if self.warnings else ""),
        ]
        if self.labels:
            lines.append("labels: " + ", ".join(f"{k}={v}" for k, v in self.labels.most_common()))
        if self.languages:
            lines.append("languages: " + ", ".join(f"{k}={v}" for k, v in self.languages.most_common()))
        for e in self.errors[:20]:
            lines.append(f"  ERROR  {e}")
        for w in self.warnings[:20]:
            lines.append(f"  warn   {w}")
        hidden = max(0, len(self.errors) - 20) + max(0, len(self.warnings) - 20)
        if hidden:
            lines.append(f"  ... {hidden} more")
        return "\n".join(lines)


def validate_fixtures(path: Path, *, max_issues: int = 200) -> FixtureReport:
    """Check a fixtures file against the record contract.

    Errors (the runner/report would misbehave): bad JSON, missing/duplicate
    id, non-string text, spans with non-integer or out-of-range offsets.
    Warnings (scoring would silently ignore something): labels outside the
    canonical taxonomy, overlapping gold spans, duplicate spans.
    """
    rep = FixtureReport(Path(path))

    def err(msg: str) -> None:
        if len(rep.errors) < max_issues:
            rep.errors.append(msg)

    def warn(msg: str) -> None:
        if len(rep.warnings) < max_issues:
            rep.warnings.append(msg)

    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                err(f"line {lineno}: invalid JSON ({e.msg})")
                continue
            rep.n_records += 1
            rid = rec.get("id")
            where = f"line {lineno} (id={rid!r})"
            if rid is None:
                err(f"line {lineno}: missing `id`")
            elif str(rid) in seen_ids:
                err(f"{where}: duplicate id")
            else:
                seen_ids.add(str(rid))
            text = rec.get("text")
            if not isinstance(text, str):
                err(f"{where}: `text` must be a string")
                continue
            if rec.get("language"):
                rep.languages[rec["language"]] += 1
            spans = rec.get("gold_spans", [])
            if not isinstance(spans, list):
                err(f"{where}: `gold_spans` must be a list")
                continue
            prev: list[tuple[int, int]] = []
            for s in spans:
                rep.n_spans += 1
                start, end, label = s.get("start"), s.get("end"), s.get("label")
                if not isinstance(start, int) or not isinstance(end, int):
                    err(f"{where}: span {s} needs integer start/end")
                    continue
                if not 0 <= start < end <= len(text):
                    err(f"{where}: span [{start}, {end}) outside text of length {len(text)}")
                    continue
                fine = s.get("fine_label") or label
                rep.labels[fine] += 1
                if not taxonomy.is_known(fine):
                    warn(f"{where}: label {fine!r} is not canonical — it won't be scored")
                if (start, end) in prev:
                    warn(f"{where}: duplicate span [{start}, {end})")
                elif any(start < e and s_ < end for s_, e in prev):
                    warn(f"{where}: span [{start}, {end}) overlaps another gold span")
                prev.append((start, end))
    return rep


# -------------------------------------------------------------------- CLI


def _cmd_materialize(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m opf_eval.fixtures")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=None,
                    help="sample size (default: 5000 for registered datasets, all records for --file)")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--dataset", default=None, choices=sorted(DATASETS),
                     help=f"registered dataset (default: {DEFAULT_DATASET})")
    src.add_argument("--file", type=Path, default=None,
                     help="local .jsonl/.json/.csv/.tsv/.parquet with canonical labels")
    ap.add_argument("--format", default=None, help="reader for --file (inferred from suffix)")
    ap.add_argument("--label-map", type=Path, default=None,
                    help="JSON file {raw_label: canonical_label} for --file")
    ap.add_argument("--split", default=None, help="HF split name (default = dataset's preferred)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    if args.file:
        name = f"file:{args.file.stem}"
        register_dataset(
            name,
            path=args.file,
            format=args.format,
            label_map=json.loads(args.label_map.read_text()) if args.label_map else None,
            overwrite=True,
        )
        n = args.n
    else:
        name = args.dataset or DEFAULT_DATASET
        n = args.n if args.n is not None else 5000
    written = materialize(args.out, n, dataset=name, split=args.split, seed=args.seed)
    print(f"wrote {written} examples from {get_dataset_config(name).source} to {args.out}")
    rep = validate_fixtures(args.out)
    print(rep.summary())
    return 0 if rep.ok else 1


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["validate"]:
        reports = [validate_fixtures(Path(p)) for p in argv[1:]]
        for rep in reports:
            print(rep.summary())
        raise SystemExit(0 if all(r.ok for r in reports) else 1)
    if argv[:1] == ["list"]:
        for row in describe_datasets():
            print(f"{row['name']:<20} {row['labels']:>2} labels  {row['source']:<48} {row['description']}")
        raise SystemExit(0)
    raise SystemExit(_cmd_materialize(argv))


if __name__ == "__main__":
    main()
