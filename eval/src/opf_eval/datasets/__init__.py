"""Dataset registry for the eval harness.

Built-in datasets are the ai4privacy family on Hugging Face. Register your
own — a local file or another HF dataset — with `register_dataset`:

    from opf_eval import datasets

    # Already labelled with canonical names (EMAIL, GOV_ID, ...):
    datasets.register_dataset("support_tickets", path="data/tickets.jsonl")

    # Your own label names, mapped onto the taxonomy:
    datasets.register_dataset(
        "claims",
        path="data/claims.csv",
        label_map={"patient_name": "PERSON", "mrn": "MEDICAL_ID", "dob": "DATE_OF_BIRTH"},
    )

Then materialize fixtures from it like any built-in
(`fixtures.materialize(out, n, dataset="claims")`).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .. import taxonomy
from .base import DatasetConfig, Loader
from .loaders import SCHEMAS, bio_loader, privacy_mask_loader, spans_loader
from .readers import read_records

_AI4PRIVACY_URL = "https://huggingface.co/datasets/"

DATASETS: dict[str, DatasetConfig] = {}


def _register(cfg: DatasetConfig, *, overwrite: bool = False) -> DatasetConfig:
    if cfg.name in DATASETS and not overwrite:
        raise ValueError(f"dataset {cfg.name!r} already registered (pass overwrite=True)")
    taxonomy.get_vocab(cfg.vocab_key)  # fail fast on an unknown vocab
    DATASETS[cfg.name] = cfg
    return cfg


def _ai4privacy(name: str, hf_id: str, vocab: str, description: str) -> None:
    _register(DatasetConfig(
        name=name,
        hf_id=hf_id,
        vocab_key=vocab,
        loader=privacy_mask_loader(vocab),
        description=description,
        url=_AI4PRIVACY_URL + hf_id,
    ))


_ai4privacy("pii_masking_300k", "ai4privacy/pii-masking-300k", "pii300k",
            "ai4privacy PII-Masking-300k — numbered name components, 6 languages")
_ai4privacy("pii_masking_200k", "ai4privacy/pii-masking-200k", "pii200k",
            "ai4privacy PII-Masking-200k — richest legacy vocabulary (15 categories)")
_ai4privacy("pii_masking_400k", "ai4privacy/pii-masking-400k", "openpii",
            "ai4privacy PII-Masking-400k — OpenPII vocabulary")
_ai4privacy("openpii_nano", "ai4privacy/openpii-masking-nano-1k", "openpii",
            "ai4privacy OpenPII nano (1k) — smoke tests / CI")
_ai4privacy("openpii_mini", "ai4privacy/openpii-masking-mini-10k", "openpii",
            "ai4privacy OpenPII mini (10k)")

DEFAULT_DATASET = "pii_masking_300k"


def register_dataset(
    name: str,
    *,
    path: str | Path | None = None,
    hf_id: str | None = None,
    split: str = "train",
    hf_revision: str | None = None,
    format: str | None = None,
    schema: str = "spans",
    label_map: dict[str, str] | None = None,
    vocab: str | None = None,
    labels: Iterable[str] | None = None,
    loader: Loader | None = None,
    description: str = "",
    url: str = "",
    license: str = "",
    languages: Iterable[str] = (),
    overwrite: bool = False,
    **loader_kwargs: Any,
) -> DatasetConfig:
    """Register a dataset so fixtures, runner and report can use it by name.

    Source: `path` (local .jsonl / .json / .csv / .tsv / .parquet) or
    `hf_id` (+ `split`, `hf_revision`).

    Labels — pick one:
      - neither `label_map` nor `vocab`: gold labels are already canonical
        names (`PERSON`, `EMAIL`, `GOV_ID`, ... — see `taxonomy.ALL_LABELS`).
      - `label_map`: your raw label -> canonical label. Registered as the
        vocabulary `ds:<name>`. Unmapped labels are dropped.
      - `vocab`: reuse an already-registered vocabulary.

    `labels` declares which canonical labels are annotated (defaults to what
    the vocabulary can produce, or what's observed in the gold for the
    identity vocab). Declare it when your vocab maps labels you didn't
    annotate, so detectors aren't charged for predicting them.

    schema: `spans` (text + char-offset spans), `privacy_mask` (ai4privacy),
    or `bio` (tokens + BIO tags). `loader_kwargs` go to the schema's loader
    (e.g. `text_field="body"`, `spans_field="pii"`, `tags_field="ner_tags"`).
    Pass `loader=` to bypass the schemas entirely.
    """
    if label_map is not None and vocab is not None:
        raise ValueError("pass label_map or vocab, not both")
    if label_map is not None:
        vocab = f"ds:{name}"
        taxonomy.register_vocab(
            vocab, label_map, kind="dataset", description=description, overwrite=True
        )
    vocab = vocab or taxonomy.CANONICAL_VOCAB
    if loader is None:
        if schema not in SCHEMAS:
            raise ValueError(f"unknown schema {schema!r}; expected one of {sorted(SCHEMAS)}")
        loader = SCHEMAS[schema](vocab, **loader_kwargs)
    if labels is not None:
        labels = tuple(labels)
        unknown = [lbl for lbl in labels if not taxonomy.is_known(lbl)]
        if unknown:
            raise ValueError(f"unknown canonical label(s) in labels=: {unknown}")
    return _register(
        DatasetConfig(
            name=name,
            vocab_key=vocab,
            loader=loader,
            hf_id=hf_id,
            default_split=split,
            hf_revision=hf_revision,
            path=str(path) if path is not None else None,
            format=format,
            labels=labels,
            description=description,
            url=url,
            license=license,
            languages=tuple(languages),
        ),
        overwrite=overwrite,
    )


def get(name: str) -> DatasetConfig:
    if name not in DATASETS:
        raise KeyError(
            f"unknown dataset {name!r}; available: {sorted(DATASETS)}. "
            f"Register your own with opf_eval.datasets.register_dataset(...)."
        )
    return DATASETS[name]


def names() -> list[str]:
    return sorted(DATASETS)


def describe() -> list[dict[str, Any]]:
    """One row per registered dataset — handy for a notebook table."""
    return [
        {
            "name": c.name,
            "source": c.source,
            "vocab": c.vocab_key,
            "labels": len(annotated_labels(c, level="coarse")),
            "description": c.description,
        }
        for c in (DATASETS[n] for n in names())
    ]


def load_raw(cfg: DatasetConfig, split: str | None = None) -> list[dict] | Any:
    """Raw records for a dataset: a list (local files) or an indexable HF Dataset."""
    if cfg.path:
        return read_records(cfg.path, cfg.format)
    from datasets import load_dataset  # lazy — only needed at materialize time

    return load_dataset(
        cfg.hf_id, split=split or cfg.default_split, revision=cfg.hf_revision, streaming=False
    )


def annotated_labels(
    cfg: DatasetConfig,
    records: Iterable[dict] | None = None,
    *,
    level: str = "fine",
) -> set[str]:
    """Canonical labels this dataset annotates, at `level`.

    Explicit `cfg.labels` win. Otherwise it's everything the vocabulary can
    produce — except for the identity `canonical` vocab, which could produce
    anything, so the labels observed in `records`' gold spans are used.
    """
    if cfg.labels is not None:
        fine = set(cfg.labels)
    elif cfg.vocab_key == taxonomy.CANONICAL_VOCAB and records is not None:
        fine = observed_labels(records)
    else:
        fine = taxonomy.vocab_labels(cfg.vocab_key, "fine")
    return {taxonomy.project(lbl, level) for lbl in fine}


def observed_labels(records: Iterable[dict]) -> set[str]:
    """Fine canonical labels present in fixture records' gold spans."""
    return {
        s.get("fine_label") or s["label"]
        for r in records
        for s in r.get("gold_spans") or []
        if taxonomy.is_known(s.get("fine_label") or s["label"])
    }


__all__ = [
    "DATASETS",
    "DEFAULT_DATASET",
    "DatasetConfig",
    "annotated_labels",
    "bio_loader",
    "describe",
    "get",
    "load_raw",
    "names",
    "observed_labels",
    "privacy_mask_loader",
    "register_dataset",
    "spans_loader",
]
