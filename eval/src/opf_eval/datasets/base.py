"""Dataset registry entry + fixture record contract.

A dataset is: where the records come from (a Hugging Face dataset id or a
local file), how a raw record becomes a fixture record (the `loader`), and
which taxonomy vocabulary its gold labels are written in (`vocab_key`).

Fixture record (one JSON object per line in a fixtures file):

    {"id": str, "text": str, "language": iso2 | null,
     "gold_spans": [{"label": coarse, "fine_label": canonical,
                     "raw_label": str, "start": int, "end": int}]}

Offsets are Python string indices into `text`, end-exclusive.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

Loader = Callable[[Iterable[dict]], Iterable[dict]]


@dataclass(frozen=True)
class DatasetConfig:
    name: str                    # short name on the CLI ("pii_masking_300k")
    vocab_key: str               # taxonomy vocabulary the gold labels use
    loader: Loader               # raw records -> fixture records
    hf_id: str | None = None     # "ai4privacy/pii-masking-300k"
    default_split: str = "train"
    hf_revision: str | None = None  # pin a dataset revision for reproducibility
    path: str | None = None      # local file (jsonl / json / csv / parquet)
    format: str | None = None    # reader for `path`; inferred from the suffix
    # Canonical labels the dataset annotates. Default: every label the vocab
    # can produce (or, for the identity `canonical` vocab, the labels observed
    # in the gold spans). Set it when a vocab covers more than is annotated.
    labels: tuple[str, ...] | None = None
    description: str = ""
    url: str = ""
    license: str = ""
    languages: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.hf_id and not self.path:
            raise ValueError(f"dataset {self.name!r} needs an hf_id or a path")

    @property
    def source(self) -> str:
        return self.path if self.path else f"hf://{self.hf_id}"
