"""GLiNER detector — small open-weight zero-shot NER (Apache 2.0).

Default model: urchade/gliner_multi_pii-v1 — multilingual, pre-trained on PII.
First load downloads ~500 MB to ~/.cache/huggingface/.

Same deployment story as OPF (local, fine-tunable, free) but smaller and
prompt-driven instead of having a fixed label vocabulary.
"""

from __future__ import annotations

import time
from typing import Callable

from gliner import GLiNER

from ..taxonomy import gliner_prompts, gliner_to_canonical
from .base import DetectorResult, Span


LabelMapper = Callable[[str], "str | None"]


class GLiNERDetector:
    name = "gliner"

    def __init__(
        self,
        *,
        model_name: str = "urchade/gliner_multi_pii-v1",
        threshold: float = 0.5,
        prompts: list[str] | None = None,
        label_to_canonical: LabelMapper | None = None,
        name: str | None = None,
        device: str = "cpu",
    ) -> None:
        """
        model_name: HuggingFace model id. Default is the multilingual PII variant.
        threshold: confidence cutoff (0-1). Default 0.5; lower = more recall, more FPs.
        prompts: explicit prompt list to feed the model. Default = full set
            (`gliner_prompts()`). Pass a restricted subset to focus the model
            on a dataset's annotated labels.
        label_to_canonical: callback mapping raw model label -> canonical
            label. Defaults to `gliner_to_canonical` (the generic GLiNER
            prompt vocabulary). Override for variants like Gretel that emit
            their own snake_case labels.
        name: override the registered detector name (for raw_<name>.jsonl
            output paths and report tables). Defaults to "gliner".
        device: torch device — `"cpu"`, `"cuda"`, or `"mps"`. The 570M
            gliner_nvidia base benefits ~10× from GPU.
        """
        self._model = GLiNER.from_pretrained(model_name)
        if device != "cpu":
            try:
                self._model = self._model.to(device)
            except Exception:  # noqa: BLE001
                # Some GLiNER bi-encoder variants don't allow .to(); fall back
                # to CPU rather than crash the run.
                pass
        self._labels = prompts if prompts is not None else gliner_prompts()
        self._threshold = threshold
        self._to_canonical = label_to_canonical or gliner_to_canonical
        if name is not None:
            self.name = name

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
            canonical = self._to_canonical(raw) or raw.upper()
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
