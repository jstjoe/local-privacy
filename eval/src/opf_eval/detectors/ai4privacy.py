"""ai4privacy multilingual ModernBERT-based PII anonymiser.

Model: `ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii`
(despite the `llama-` prefix in the name, the actual base is
`answerdotai/ModernBERT-base` per the model card).

Trained on `ai4privacy/open-pii-masking-500k-ai4privacy` — same OpenPII
vocabulary used by our `pii_masking_400k` / `openpii_nano` / `openpii_mini`
datasets, so labels map directly via `taxonomy.dataset_to_canonical("openpii", …)`.

8 languages: fr, en, de, te, hi, it, es, nl. MIT license. ~150M params.
"""

from __future__ import annotations

import time
from typing import Literal

from transformers import pipeline

from ..taxonomy import dataset_to_canonical
from .base import DetectorResult, Span


DEFAULT_MODEL = "ai4privacy/llama-ai4privacy-multilingual-categorical-anonymiser-openpii"
AggregationStrategy = Literal["none", "simple", "first", "average", "max"]


class Ai4PrivacyDetector:
    name = "ai4privacy_modernbert"

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        aggregation_strategy: AggregationStrategy = "simple",
        device: str = "cpu",
    ) -> None:
        """
        aggregation_strategy: how the HF pipeline merges sub-token predictions.
            `simple` is the default — empirically best on this model: produces
            ~3× more entities with cleaner boundaries than `first` (which over-
            extends spans into trailing whitespace + punctuation). `average`
            and `max` give cleaner boundaries but lower recall.
        device: torch device — `"cpu"`, `"cuda"`, or `"mps"`. Translated to
            the HF pipeline's int convention internally.
        """
        # HF pipeline takes an int (-1 = CPU, 0..N = CUDA index) or a string
        # for non-CUDA accelerators. Translate from our unified naming.
        hf_device: int | str = -1
        if device == "cuda":
            hf_device = 0
        elif device == "mps":
            hf_device = "mps"
        self._pipe = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy=aggregation_strategy,
            device=hf_device,
        )

    def close(self) -> None:
        """Drop the HF pipeline so its model weights can be released.
        Called by the runner between detector iterations to free VRAM."""
        self._pipe = None  # type: ignore[assignment]

    def detect(self, text: str, **_context: object) -> DetectorResult:
        t0 = time.perf_counter()
        try:
            entities = self._pipe(text)
        except Exception as e:  # noqa: BLE001
            return {"spans": [], "latency_ms": (time.perf_counter() - t0) * 1000, "error": repr(e)}
        latency_ms = (time.perf_counter() - t0) * 1000
        spans: list[Span] = []
        for ent in entities or []:
            raw = ent.get("entity_group") or ent.get("entity") or ""
            if not raw or raw.upper() == "O":
                continue
            # Strip BIO prefix if pipeline returned per-token labels rather
            # than aggregated groups (e.g. when aggregation_strategy="none").
            bare = raw.split("-", 1)[1] if "-" in raw and raw[1:2] == "-" else raw
            canonical = dataset_to_canonical("openpii", bare)
            if not canonical:
                continue
            start, end = _trim_boundaries(text, int(ent["start"]), int(ent["end"]))
            if end <= start:
                continue
            spans.append(
                {
                    "label": canonical,
                    "raw_label": bare,
                    "start": start,
                    "end": end,
                    "text": text[start:end],
                }
            )
        return {"spans": spans, "latency_ms": latency_ms, "error": None}


_TRIM_CHARS = " \t\n\r,;:\"'`()[]{}<>"


def _trim_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """The ai4privacy pipeline often includes trailing whitespace/punctuation
    in span boundaries (subword token alignment quirk). Trim conservatively."""
    while start < end and text[start] in _TRIM_CHARS:
        start += 1
    while end > start and text[end - 1] in _TRIM_CHARS:
        end -= 1
    return start, end
