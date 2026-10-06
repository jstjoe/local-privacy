"""Generic Hugging Face `token-classification` detector.

Wraps `transformers.pipeline("token-classification")` for any fine-tuned
BERT / DeBERTa / ModernBERT / XLM-R style PII model. Raw entity-group labels
are mapped to canonical labels through a taxonomy vocabulary, so adding a new
checkpoint is: register a vocab for its label names, then
`registry.register_hf_token_classifier(name, model_id, vocab=...)`.

`Ai4PrivacyDetector` (in `ai4privacy.py`) is this class with ai4privacy's
defaults.
"""

from __future__ import annotations

import time
from typing import Literal

from transformers import pipeline

from .base import DetectorResult, Span, error_result, make_span


AggregationStrategy = Literal["none", "simple", "first", "average", "max"]


class HFTokenClassifierDetector:
    name = "hf_token_classifier"

    def __init__(
        self,
        *,
        model_name: str,
        vocab: str,
        name: str | None = None,
        aggregation_strategy: AggregationStrategy = "simple",
        device: str = "cpu",
    ) -> None:
        """
        model_name: Hugging Face model id.
        vocab: taxonomy vocabulary mapping the model's entity-group labels to
            canonical labels. Labels outside it are dropped.
        name: detector name for raw_<name>.jsonl and report tables.
        aggregation_strategy: how the HF pipeline merges sub-token predictions.
            `simple` is the default. `first` tends to over-extend spans into
            trailing whitespace and punctuation; `average` / `max` give cleaner
            boundaries but lower recall on some models.
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
        self._vocab = vocab
        if name is not None:
            self.name = name
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
            return error_result(t0, e)
        latency_ms = (time.perf_counter() - t0) * 1000
        spans: list[Span] = []
        for ent in entities or []:
            raw = ent.get("entity_group") or ent.get("entity") or ""
            if not raw or raw.upper() == "O":
                continue
            # Strip BIO prefix if pipeline returned per-token labels rather
            # than aggregated groups (e.g. when aggregation_strategy="none").
            bare = raw.split("-", 1)[1] if "-" in raw and raw[1:2] == "-" else raw
            start, end = _trim_boundaries(text, int(ent["start"]), int(ent["end"]))
            if end <= start:
                continue
            span = make_span(self._vocab, bare, start, end, text[start:end], keep_unmapped=False)
            if span is not None:
                spans.append(span)
        return {"spans": spans, "latency_ms": latency_ms, "error": None}


_TRIM_CHARS = " \t\n\r,;:\"'`()[]{}<>"


def _trim_boundaries(text: str, start: int, end: int) -> tuple[int, int]:
    """HF pipelines often include leading/trailing whitespace or punctuation
    in span boundaries (sub-word alignment quirk). Trim conservatively."""
    while start < end and text[start] in _TRIM_CHARS:
        start += 1
    while end > start and text[end - 1] in _TRIM_CHARS:
        end -= 1
    return start, end
