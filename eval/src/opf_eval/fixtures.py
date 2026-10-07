"""Materialize a deterministic sample from any registered dataset into a
fixtures file, and validate fixtures files.

Fixture record (one JSON object per line; see `datasets.base`):

    {"id": str, "text": str, "language": iso2|null,
     "gold_spans": [{"label": coarse, "fine_label": canonical,
                     "raw_label": str, "start": int, "end": int}]}

`materialize` also writes `<fixtures>.meta.json` recording the dataset,
sample parameters and annotated labels, so the runner and report know what
the fixtures contain without being told again.

Unlabeled fixtures come from your own text or files instead of a dataset.
`from_texts` and `from_documents` write records with `gold_spans: []` and a
sidecar whose `"gold"` is `"none"`. The runner treats them like any other
fixtures. Precision and recall then come from silver labels or reviews
(see plan 13) rather than from gold. Records made from documents also carry
`doc_id` and `offset`, so `map_spans_to_documents` can put each detected span
back on its page or cell.

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
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import taxonomy
from .datasets import (
    DATASETS,
    DEFAULT_DATASET,
    annotated_labels,
    load_raw,
    register_dataset,
)
from .datasets import (
    describe as describe_datasets,
)
from .datasets import (
    get as get_dataset_config,
)
from .documents import Document, Segment, chunk
from .io import GOLD_NONE, file_sha256, iter_jsonl, meta_path, read_meta, write_jsonl

META_VERSION = 2
UNLABELED = GOLD_NONE  # meta["gold"] for fixtures with no gold spans


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
    # Matching needs only the sidecar, so a custom dataset registered in an
    # earlier notebook/session doesn't have to be registered again to reuse it.
    if (
        out_path.exists()
        and meta is not None
        and meta.get("dataset") == dataset
        and (split is None or meta.get("split") == split)
        and meta.get("n_requested") == n
        and meta.get("seed") == (seed if n is not None else None)
        and meta.get("sha256") == file_sha256(out_path)
    ):
        return out_path, True
    materialize(out_path, n, dataset=dataset, split=split, seed=seed)
    return out_path, False


# ------------------------------------------------------- unlabeled fixtures


def default_target_labels(level: str = "fine") -> list[str]:
    """Every canonical label at `level`, sorted.

    "fine" returns the coarse categories and their fine sub-types together
    because some detectors only emit the coarse label for a category.
    """
    taxonomy.check_level(level)
    return sorted(taxonomy.ALL_LABELS if level == "fine" else taxonomy.COARSE_LABELS)


def _target_labels(labels: Iterable[str] | None) -> list[str]:
    if labels is None:
        return default_target_labels("fine")
    labels = sorted(set(labels))
    if not labels:
        raise ValueError("labels must not be empty")
    unknown = [lbl for lbl in labels if not taxonomy.is_known(lbl)]
    if unknown:
        raise ValueError(
            f"unknown label(s) {unknown}; use canonical labels such as {', '.join(taxonomy.COARSE_LABELS[:5])}"
        )
    return labels


def documents_path(fixtures: str | Path) -> Path:
    """Sidecar with each document's segment map, written by `from_documents`."""
    p = Path(fixtures)
    return p.with_name(p.name + ".documents.jsonl")


def is_unlabeled(fixtures: str | Path) -> bool:
    """True when the fixtures' sidecar says they have no gold spans."""
    meta = read_meta(fixtures)
    return bool(meta) and meta.get("gold") == UNLABELED


def _write_unlabeled(
    out_path: Path,
    rows: list[dict],
    *,
    source: str,
    labels: list[str],
    documents: dict[str, dict],
    extra: dict | None = None,
) -> Path:
    written = write_jsonl(out_path, rows)
    meta = {
        "version": META_VERSION,
        "dataset": f"custom:{out_path.stem}",
        "source": source,
        "gold": UNLABELED,
        "vocab_key": None,
        "labels": labels,
        "n_written": written,
        "documents": documents,
        **(extra or {}),
        "sha256": file_sha256(out_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path(out_path).write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return out_path


def from_texts(
    texts: Iterable[str] | Mapping[str, str],
    out_path: str | Path,
    *,
    labels: Iterable[str] | None = None,
    language: str | None = None,
) -> Path:
    """Write unlabeled fixtures from plain strings.

    A sequence gets ids "text-0001", "text-0002", ... in order. A mapping
    uses its keys as ids. Blank texts are left out but keep their number, so
    ids do not shift when one is removed. `labels` is the set detectors are
    asked to find and silver labels or reviews are scored on (default: every
    canonical label).
    """
    out_path = Path(out_path)
    target = _target_labels(labels)
    if isinstance(texts, Mapping):
        items = [(str(k), v) for k, v in texts.items()]
    else:
        items = [(f"text-{i:04d}", t) for i, t in enumerate(texts, 1)]
    rows, skipped = [], 0
    for rid, text in items:
        if not isinstance(text, str):
            raise TypeError(f"text {rid!r} must be a str, not {type(text).__name__}")
        if not text.strip():
            skipped += 1
            continue
        rows.append({"id": rid, "text": text, "language": language, "gold_spans": []})
    documents_path(out_path).unlink(missing_ok=True)  # stale map from an earlier from_documents
    return _write_unlabeled(
        out_path, rows, source="texts", labels=target, documents={},
        extra={"n_blank_skipped": skipped},
    )


def from_documents(
    docs: Iterable[Document],
    out_path: str | Path,
    *,
    max_chars: int = 4000,
    labels: Iterable[str] | None = None,
    language: str | None = None,
) -> Path:
    """Write unlabeled fixtures from parsed documents, one record per chunk.

    Each document is split with `documents.chunk(doc, max_chars=...)` and
    chunks holding only whitespace are left out of the records. The segment
    sidecar keeps their text under `"blank"` so `load_documents` can still
    rebuild every document in full.
    Records are `{"id": "<doc id>#<k>", "text", "language", "gold_spans": [],
    "doc_id", "offset"}` where `offset` is where the chunk starts in the
    document's text. The sidecar lists every document with its source, kind,
    chunk count and parser warnings. A second sidecar
    (`documents_path(out_path)`) keeps each document's segment map so spans
    can be located after a restart without parsing the files again.
    """
    out_path = Path(out_path)
    target = _target_labels(labels)
    rows: list[dict] = []
    documents: dict[str, dict] = {}
    seg_rows: list[dict] = []
    for doc in docs:
        if doc.id in documents:
            raise ValueError(f"duplicate document id {doc.id!r} ({doc.source})")
        # A whitespace-only chunk holds no PII and some detector APIs reject
        # blank input. `chunk` avoids them where it can; drop the rest.
        ranges, blank = [], []
        for a, b in chunk(doc, max_chars=max_chars):
            (ranges if doc.text[a:b].strip() else blank).append((a, b))
        for k, (start, end) in enumerate(ranges):
            rows.append({
                "id": f"{doc.id}#{k}",
                "text": doc.text[start:end],
                "language": language,
                "gold_spans": [],
                "doc_id": doc.id,
                "offset": start,
            })
        documents[doc.id] = {
            "source": doc.source,
            "kind": doc.kind,
            "n_chunks": len(ranges),
            "warnings": list(doc.warnings),
        }
        seg_rows.append({
            "id": doc.id,
            "source": doc.source,
            "kind": doc.kind,
            "length": len(doc.text),
            "segments": [s.to_json() for s in doc.segments],
            "blank": [[a, doc.text[a:b]] for a, b in blank],
            "warnings": list(doc.warnings),
        })
    _write_unlabeled(
        out_path, rows, source="files", labels=target, documents=documents,
        extra={"max_chars": max_chars},
    )
    write_jsonl(documents_path(out_path), seg_rows)
    return out_path


def read_document_map(fixtures: str | Path) -> dict[str, dict]:
    """Chunk id -> `{"doc_id", "source", "kind", "offset"}` for every record
    that came from a document. Records from `from_texts` are not included."""
    meta = read_meta(fixtures) or {}
    docs = meta.get("documents") or {}
    out: dict[str, dict] = {}
    for rec in iter_jsonl(fixtures):
        doc_id = rec.get("doc_id")
        if doc_id is None:
            continue
        info = docs.get(doc_id, {})
        out[rec["id"]] = {
            "doc_id": doc_id,
            "source": info.get("source"),
            "kind": info.get("kind"),
            "offset": int(rec.get("offset", 0)),
        }
    return out


def load_documents(fixtures: str | Path) -> dict[str, Document]:
    """Rebuild the documents behind `from_documents` fixtures, keyed by id.

    The text is the chunks put back together with the whitespace-only chunks
    that the sidecar keeps under `"blank"`. Together they cover each document
    with no gaps. The segments come from the `.documents.jsonl` sidecar.
    Raises FileNotFoundError when there is no sidecar and ValueError when the
    chunks no longer fit the sidecar, for example after the fixtures file was
    edited by hand.
    """
    side = documents_path(fixtures)
    if not side.exists():
        raise FileNotFoundError(f"{side} not found — fixtures were not written by from_documents")
    pieces: dict[str, list[tuple[int, str]]] = {}
    for rec in iter_jsonl(fixtures):
        if rec.get("doc_id") is not None:
            pieces.setdefault(rec["doc_id"], []).append((int(rec.get("offset", 0)), rec["text"]))
    out: dict[str, Document] = {}
    for row in iter_jsonl(side):
        text, pos = [], 0
        blank = [(int(offset), piece) for offset, piece in row.get("blank") or []]
        for offset, piece in sorted(pieces.get(row["id"], []) + blank):
            if offset != pos:
                raise ValueError(f"document {row['id']!r}: chunk at offset {offset} does not follow {pos}")
            text.append(piece)
            pos += len(piece)
        if pos != row["length"]:
            raise ValueError(f"document {row['id']!r}: chunks give {pos} chars, sidecar says {row['length']}")
        out[row["id"]] = Document(
            id=row["id"],
            source=row["source"],
            kind=row["kind"],
            text="".join(text),
            segments=[Segment.from_json(s) for s in row["segments"]],
            warnings=list(row.get("warnings") or []),
        )
    return out


def map_spans_to_documents(
    fixtures: str | Path,
    run_dir: str | Path,
    detector: str,
    docs: Mapping[str, Document] | Iterable[Document] | None = None,
) -> list[dict]:
    """Put one detector's spans back on whole documents.

    Reads `run_dir/raw_<detector>.jsonl` and shifts every span on a document
    chunk by the chunk's offset. Rows are
    `{"doc_id", "source", "detector", "label", "fine_label", "text",
    "doc_start", "doc_end", "where", "chunk_id"}` sorted by document and
    position. `where` lists the page/cell/line locations the span touches.
    It is filled from `docs` when given, else from the `.documents.jsonl`
    sidecar when present, else left empty. A sidecar that no longer fits the
    fixtures gives a warning and empty `where` lists instead of an error.
    Spans on records that did not come from a document are skipped.
    """
    doc_map = read_document_map(fixtures)
    if docs is None:
        try:
            docs = load_documents(fixtures)
        except FileNotFoundError:
            docs = {}
        except ValueError as e:
            warnings.warn(f"page and line locations are left out because {e}", stacklevel=2)
            docs = {}
    elif not isinstance(docs, Mapping):
        docs = {d.id: d for d in docs}

    order = {doc_id: i for i, doc_id in enumerate(dict.fromkeys(m["doc_id"] for m in doc_map.values()))}
    rows: list[dict] = []
    for raw in iter_jsonl(Path(run_dir) / f"raw_{detector}.jsonl"):
        info = doc_map.get(raw.get("id"))
        if info is None:
            continue
        doc = docs.get(info["doc_id"])
        for span in raw.get("spans") or []:
            start = info["offset"] + int(span["start"])
            end = info["offset"] + int(span["end"])
            text = span.get("text")
            if text is None and doc is not None:
                text = doc.text[start:end]
            rows.append({
                "doc_id": info["doc_id"],
                "source": info["source"],
                "detector": detector,
                "label": span.get("label"),
                "fine_label": span.get("fine_label") or span.get("label"),
                "text": text,
                "doc_start": start,
                "doc_end": end,
                "where": doc.locate(start, end) if doc is not None else [],
                "chunk_id": raw["id"],
            })
    rows.sort(key=lambda r: (order.get(r["doc_id"], len(order)), r["doc_start"], r["doc_end"], str(r["label"])))
    return rows


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
    unlabeled: bool = False  # sidecar says gold == "none" (from_texts / from_documents)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        gold = "unlabeled (no gold spans)" if self.unlabeled else f"{self.n_spans} gold spans"
        lines = [
            f"{self.path}: {self.n_records} records, {gold}"
            f" — {'OK' if self.ok else f'{len(self.errors)} error(s)'}"
            + (f", {len(self.warnings)} warning(s)" if self.warnings else ""),
        ]
        if self.labels:
            lines.append("labels: " + ", ".join(f"{k}={v}" for k, v in self.labels.most_common()))
        if self.languages:
            lines.append(
                "languages: "
                + ", ".join(f"{k}={v}" for k, v in self.languages.most_common())
            )
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

    Empty gold is always valid. When the sidecar marks the fixtures as
    unlabeled, `unlabeled` is set and gold spans found anyway are a warning.
    Document chunks must carry a non-negative integer `offset`.
    """
    rep = FixtureReport(Path(path))
    rep.unlabeled = is_unlabeled(path)

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
            if "doc_id" in rec:
                offset = rec.get("offset")
                if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
                    err(f"{where}: document chunk needs a non-negative integer `offset`")
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
    if rep.unlabeled and rep.n_spans:
        warn(f"the sidecar marks these fixtures unlabeled but they hold {rep.n_spans} gold spans")
    return rep


# -------------------------------------------------------------------- CLI


def _cmd_materialize(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m opf_eval.fixtures")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=None,
                    help="sample size (default: 5000 for registered datasets, "
                         "all records for --file)")
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
            print(
                f"{row['name']:<20} {row['labels']:>2} labels  "
                f"{row['source']:<48} {row['description']}"
            )
        raise SystemExit(0)
    raise SystemExit(_cmd_materialize(argv))


if __name__ == "__main__":
    main()
