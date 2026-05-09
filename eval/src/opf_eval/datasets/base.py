"""Dataset registry + loader contract.

Each ai4privacy dataset has its own annotation vocabulary; the loader maps
raw HF records into our canonical-span shape using a per-dataset reverse
function. Languages are normalized to ISO codes (`"en"`, `"nl"`, `"hu"`)
so detectors that route per-language (Presidio) get a consistent input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


@dataclass(frozen=True)
class DatasetConfig:
    name: str                # short name on the CLI ("pii_masking_300k")
    hf_id: str               # "ai4privacy/pii-masking-300k"
    default_split: str
    vocab_key: str           # which CANONICAL_MAP column to use
    loader: Callable[[Iterable[dict]], Iterable[dict]]
    """loader: maps raw HF records into our fixture record shape:
        {"id": str, "text": str, "gold_spans": [...], "language": iso2}
    """
