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
        aggregation_strategy: AggregationStrategy = "first",
        device: int | str = -1,
    ) -> None:
        """
        aggregation_strategy: how the HF pipeline merges sub-token predictions.
            `first` is the default — use the first sub-token's score for the
            aggregated entity. `simple` over-merges, `average` is more lenient.
        device: -1 = CPU; 0 = first CUDA device; "mps" for Apple Silicon.
        """
        self._pipe = pipeline(
            "token-classification",
            model=model_name,
            aggregation_strategy=aggregation_strategy,
            device=device,
        )

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
            start = int(ent["start"])
            end = int(ent["end"])
            spans.append(
                {
                    "label": canonical,
                    "raw_label": bare,
                    "start": start,
                    "end": end,
                    "text": ent.get("word") or text[start:end],
                }
            )
        return {"spans": spans, "latency_ms": latency_ms, "error": None}
