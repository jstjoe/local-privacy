"""Materialize a deterministic sample from any registered ai4privacy dataset
into our fixture JSONL shape.

Schema per line:
    {"id": str, "text": str, "language": iso2|null,
     "gold_spans": [{"label": canonical, "raw_label": str, "start": int, "end": int}]}

Use `--dataset` to pick the source (default: pii_masking_300k for back-compat).

Examples:

    python -m opf_eval.fixtures --out data/sample_1k.jsonl --n 1000
    python -m opf_eval.fixtures --dataset openpii_nano --out data/openpii_nano.jsonl --n 1000
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .datasets import DATASETS, DEFAULT_DATASET, get as get_dataset_config


def materialize(
    out_path: Path,
    n: int,
    *,
    dataset: str = DEFAULT_DATASET,
    split: str | None = None,
    seed: int = 42,
) -> int:
    """Sample `n` records from the named dataset and write fixture JSONL.

    Returns the count actually written (records with no usable text are skipped).
    """
    from datasets import load_dataset  # lazy — only needed at materialize time

    cfg = get_dataset_config(dataset)
    ds = load_dataset(cfg.hf_id, split=split or cfg.default_split, streaming=False)
    indices = list(range(len(ds)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    indices = indices[: min(n, len(indices))]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    raw_iter = (ds[int(i)] for i in indices)
    with out_path.open("w") as f:
        for record in cfg.loader(raw_iter):
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    return written


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        choices=sorted(DATASETS),
        help=f"Source dataset (default: {DEFAULT_DATASET}).",
    )
    ap.add_argument("--split", default=None, help="HF split name (default = dataset's preferred)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n = materialize(
        args.out, args.n, dataset=args.dataset, split=args.split, seed=args.seed
    )
    cfg = get_dataset_config(args.dataset)
    print(f"wrote {n} examples from {cfg.hf_id} to {args.out}")


if __name__ == "__main__":
    main()
