"""Materialize a deterministic sample from PII-Masking-300k as JSONL fixtures.

Schema per line:
    {"id": str, "text": str, "gold_spans": [{"label": canonical, "start": int, "end": int, "raw_label": str}]}

PII-Masking-300k stores spans as BIO-tagged tokens with offsets. We project to
character-level spans + canonical labels via taxonomy.pii300k_to_canonical.

Usage:
    python -m opf_eval.fixtures --out data/sample.jsonl --n 5000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .taxonomy import pii300k_to_canonical


DATASET_NAME = "ai4privacy/pii-masking-300k"
DEFAULT_SPLIT = "train"


def _coalesce_token_spans(
    text: str,
    tokens: list[str],
    labels: list[str],
    offsets: list[tuple[int, int]] | None = None,
) -> list[dict]:
    """BIO -> contiguous canonical spans."""
    spans: list[dict] = []
    cur_label: str | None = None
    cur_raw: str | None = None
    cur_start: int | None = None
    cur_end: int | None = None
    for i, raw in enumerate(labels):
        if raw in ("O", "0", None):
            tag = None
        else:
            tag = raw[2:] if raw[1:2] == "-" else raw
        canonical = pii300k_to_canonical(tag) if tag else None
        if offsets is not None:
            start, end = offsets[i]
        else:
            # fall back to running concat — assumes whitespace separator
            start = sum(len(t) + 1 for t in tokens[:i])
            end = start + len(tokens[i])
        if canonical and canonical == cur_label:
            cur_end = end
            continue
        if cur_label is not None and cur_start is not None and cur_end is not None:
            spans.append(
                {
                    "label": cur_label,
                    "raw_label": cur_raw,
                    "start": cur_start,
                    "end": cur_end,
                }
            )
        if canonical:
            cur_label = canonical
            cur_raw = tag
            cur_start = start
            cur_end = end
        else:
            cur_label = cur_raw = cur_start = cur_end = None
    if cur_label is not None and cur_start is not None and cur_end is not None:
        spans.append(
            {"label": cur_label, "raw_label": cur_raw, "start": cur_start, "end": cur_end}
        )
    return spans


def _record_text(record: dict) -> str:
    for key in ("source_text", "unmasked_text", "text", "raw_text"):
        v = record.get(key)
        if isinstance(v, str):
            return v
    raise KeyError(f"no text column found in record: {sorted(record)}")


def _record_spans(record: dict, text: str) -> list[dict]:
    """Try several known PII-Masking-300k schemas, return canonical spans."""
    # Variant A: token-level BIO with offsets
    tokens = record.get("tokens") or record.get("source_tokens")
    labels = record.get("ner_tags_str") or record.get("ner_tags") or record.get("labels")
    offsets = record.get("token_char_spans") or record.get("offsets")
    if tokens and labels and len(tokens) == len(labels):
        if offsets and len(offsets) == len(tokens):
            offs = [tuple(o) for o in offsets]
        else:
            offs = None
        return _coalesce_token_spans(text, list(tokens), list(labels), offs)

    # Variant B: span-level "privacy_mask" list of {value,label,start,end}
    raw_spans = record.get("privacy_mask") or record.get("spans")
    if isinstance(raw_spans, list):
        out: list[dict] = []
        for s in raw_spans:
            if not isinstance(s, dict):
                continue
            label = s.get("label") or s.get("entity_type") or s.get("type")
            start = s.get("start") or s.get("start_index")
            end = s.get("end") or s.get("end_index")
            if label is None or start is None or end is None:
                continue
            canonical = pii300k_to_canonical(label)
            if not canonical:
                continue
            out.append(
                {
                    "label": canonical,
                    "raw_label": label,
                    "start": int(start),
                    "end": int(end),
                }
            )
        return out

    return []


def materialize(out_path: Path, n: int, split: str = DEFAULT_SPLIT, seed: int = 42) -> int:
    from datasets import load_dataset

    ds = load_dataset(DATASET_NAME, split=split, streaming=False)
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[:n]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w") as f:
        for idx in indices:
            rec = ds[int(idx)]
            try:
                text = _record_text(rec)
                gold = _record_spans(rec, text)
            except (KeyError, ValueError):
                continue
            record_out = {
                "id": str(idx),
                "text": text,
                "gold_spans": gold,
                "language": rec.get("language"),
            }
            f.write(json.dumps(record_out, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n = materialize(args.out, args.n, split=args.split, seed=args.seed)
    print(f"wrote {n} examples to {args.out}")


if __name__ == "__main__":
    main()
