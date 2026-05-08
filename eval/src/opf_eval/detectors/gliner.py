"""GLiNER detector — small open-weight zero-shot NER (Apache 2.0).

Default model: urchade/gliner_multi_pii-v1 — multilingual, pre-trained on PII.
First load downloads ~500 MB to ~/.cache/huggingface/.

Same deployment story as OPF (local, fine-tunable, free) but smaller and
prompt-driven instead of having a fixed label vocabulary.
"""

from __future__ import annotations

import time

from gliner import GLiNER

from ..taxonomy import gliner_prompts, gliner_to_canonical
from .base import DetectorResult, Span


class GLiNERDetector:
    name = "gliner"

    def __init__(
        self,
        *,
        model_name: str = "urchade/gliner_multi_pii-v1",
        threshold: float = 0.5,
    ) -> None:
        """
        model_name: HuggingFace model id. Default is the multilingual PII variant.
        threshold: confidence cutoff (0-1). Default 0.5; lower = more recall, more FPs.
        """
        self._model = GLiNER.from_pretrained(model_name)
        self._labels = gliner_prompts()
        self._threshold = threshold

    def detect(self, text: str, **_context: object) -> DetectorResult:
        t0 = time.perf_counter()
        try:
            entities = self._model.predict_entities(
                text, self._labels, threshold=self._threshold
            )
        except Exception as e:  # noqa: BLE001
            return {"spans": [], "latency_ms": (time.perf_counter() - t0) * 1000, "error": repr(e)}
        latency_ms = (time.perf_counter() - t0) * 1000
        spans: list[Span] = []
        for ent in entities:
            raw = ent.get("label", "")
            canonical = gliner_to_canonical(raw) or raw.upper()
            start = int(ent["start"])
            end = int(ent["end"])
            spans.append(
                {
                    "label": canonical,
                    "raw_label": raw,
                    "start": start,
                    "end": end,
                    "text": ent.get("text") or text[start:end],
                }
            )
        return {"spans": spans, "latency_ms": latency_ms, "error": None}
