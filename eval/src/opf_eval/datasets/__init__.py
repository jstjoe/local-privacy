"""Dataset registry for the eval harness."""

from __future__ import annotations

from .. import taxonomy
from .base import DatasetConfig
from .loaders import make_loader


DATASETS: dict[str, DatasetConfig] = {
    "pii_masking_300k": DatasetConfig(
        name="pii_masking_300k",
        hf_id="ai4privacy/pii-masking-300k",
        default_split="train",
        vocab_key="pii300k",
        loader=make_loader(taxonomy.pii300k_to_canonical),
    ),
    "pii_masking_200k": DatasetConfig(
        name="pii_masking_200k",
        hf_id="ai4privacy/pii-masking-200k",
        default_split="train",
        vocab_key="pii200k",
        loader=make_loader(lambda lbl: taxonomy.dataset_to_canonical("pii200k", lbl)),
    ),
    "pii_masking_400k": DatasetConfig(
        name="pii_masking_400k",
        hf_id="ai4privacy/pii-masking-400k",
        default_split="train",
        vocab_key="openpii",
        loader=make_loader(lambda lbl: taxonomy.dataset_to_canonical("openpii", lbl)),
    ),
    "openpii_nano": DatasetConfig(
        name="openpii_nano",
        hf_id="ai4privacy/openpii-masking-nano-1k",
        default_split="train",
        vocab_key="openpii",
        loader=make_loader(lambda lbl: taxonomy.dataset_to_canonical("openpii", lbl)),
    ),
    "openpii_mini": DatasetConfig(
        name="openpii_mini",
        hf_id="ai4privacy/openpii-masking-mini-10k",
        default_split="train",
        vocab_key="openpii",
        loader=make_loader(lambda lbl: taxonomy.dataset_to_canonical("openpii", lbl)),
    ),
}

DEFAULT_DATASET = "pii_masking_300k"


def get(name: str) -> DatasetConfig:
    if name not in DATASETS:
        raise KeyError(
            f"unknown dataset {name!r}; available: {sorted(DATASETS)}"
        )
    return DATASETS[name]


def names() -> list[str]:
    return sorted(DATASETS)


__all__ = ["DATASETS", "DEFAULT_DATASET", "DatasetConfig", "get", "names"]
