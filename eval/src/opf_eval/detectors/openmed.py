"""OpenMed PII detector — DeBERTa-based, multilingual via per-language models.

Wraps `openmed.extract_pii` which routes to the right per-language checkpoint
in `openmed.DEFAULT_PII_MODELS` based on the `lang` arg. English uses the
44M-param `OpenMed-PII-SuperClinical-Small-44M-v1`; non-English uses
larger 434M-568M variants. Snake_case labels — see `taxonomy.openmed`.

Note: `OpenMed/privacy-filter-multilingual` (a single-model OPF-architecture
alternative the user might find on HF) is currently broken — its config has
`model_type=openai_privacy_filter` which isn't registered in transformers,
and the openmed library's loader can't handle it either. The default
DEFAULT_PII_MODELS family used here works cleanly via standard HF loading.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ..taxonomy import openmed_to_canonical
from .base import DetectorResult, Span

if TYPE_CHECKING:
    from openmed import ModelLoader


# Languages OpenMed has a per-language PII checkpoint for. Anything else
# falls back to the English model (and likely under-performs).
_SUPPORTED_LANGS = {"en", "fr", "de", "it", "es", "nl", "hi", "te", "pt"}


class OpenMedDetector:
    name = "openmed"

    def __init__(
        self,
        *,
        confidence_threshold: float = 0.5,
        default_lang: str = "en",
        device: str = "cpu",
    ) -> None:
        """
        confidence_threshold: drop predictions below this score (0..1).
        default_lang: language code used when the fixture record doesn't
            carry a `language` hint.
        device: torch device passed through to OpenMedConfig — `"cpu"`,
            `"cuda"`, or `"mps"`. Big speedup on the larger 434–568M
            non-English checkpoints.
        """
        # Lazy import — keeps the openmed dep optional at module import time.
        import openmed

        self._openmed = openmed
        self._loader_cls = openmed.ModelLoader
        self._config_cls = openmed.OpenMedConfig
        self._loaders: dict[str, "ModelLoader"] = {}
        self._threshold = confidence_threshold
        self._default_lang = default_lang
        self._device = device

    def close(self) -> None:
        """Drop every cached per-language loader so their weights can be
        released. Called by the runner between detector iterations to free
        VRAM."""
        self._loaders.clear()

    def _get_loader(self, lang: str) -> "ModelLoader":
        if lang not in self._loaders:
            cfg = self._config_cls(device=self._device) if self._device != "cpu" else None
            self._loaders[lang] = self._loader_cls(config=cfg)
        return self._loaders[lang]

    def detect(self, text: str, **context: object) -> DetectorResult:
        # Fixtures carry ISO codes; legacy 300k carries full names — translate.
        raw_lang = str(context.get("language") or "")
        lang = raw_lang.lower() if len(raw_lang) == 2 else self._default_lang
        if lang not in _SUPPORTED_LANGS:
            lang = self._default_lang

        t0 = time.perf_counter()
        try:
            result = self._openmed.extract_pii(
                text,
                confidence_threshold=self._threshold,
                lang=lang,
                loader=self._get_loader(lang),
            )
        except Exception as e:  # noqa: BLE001
            return {
                "spans": [],
                "latency_ms": (time.perf_counter() - t0) * 1000,
                "error": repr(e),
            }
        latency_ms = (time.perf_counter() - t0) * 1000
        spans: list[Span] = []
        for ent in result.entities or []:
            raw = ent.label
            if not raw or raw.upper() == "O":
                continue
            canonical = openmed_to_canonical(raw)
            if not canonical:
                continue
            spans.append(
                {
                    "label": canonical,
                    "raw_label": raw,
                    "start": int(ent.start),
                    "end": int(ent.end),
                    "text": ent.text,
                }
            )
        return {"spans": spans, "latency_ms": latency_ms, "error": None}
