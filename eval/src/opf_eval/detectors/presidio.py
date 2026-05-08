"""Microsoft Presidio detector — regex + spaCy NER baseline (Apache 2.0).

Multi-language by default: routes per call based on the `language` context
field passed by the runner. Falls back to English when no language hint
is provided or the detected language isn't loaded.

Required spaCy models (install once with the commands below):
    python -m spacy download en_core_web_lg
    python -m spacy download nl_core_news_lg
    python -m spacy download fr_core_news_lg
    python -m spacy download de_core_news_lg
    python -m spacy download it_core_news_lg
    python -m spacy download es_core_news_lg

To save ~3 GB, swap _lg → _sm. Smaller models hit lower NER quality.
"""

from __future__ import annotations

import time

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from ..taxonomy import presidio_to_canonical
from .base import DetectorResult, Span


# Map PII-Masking-300k's full-name `language` field -> (ISO code, spaCy model).
# Order is the default `languages` list when caller doesn't override.
LANGUAGE_MODELS: dict[str, tuple[str, str]] = {
    "English": ("en", "en_core_web_lg"),
    "Dutch": ("nl", "nl_core_news_lg"),
    "French": ("fr", "fr_core_news_lg"),
    "German": ("de", "de_core_news_lg"),
    "Italian": ("it", "it_core_news_lg"),
    "Spanish": ("es", "es_core_news_lg"),
}


class PresidioDetector:
    name = "presidio"

    def __init__(
        self,
        *,
        languages: list[str] | None = None,
        score_threshold: float = 0.0,
    ) -> None:
        """
        languages: subset of LANGUAGE_MODELS keys to load. Default ["English"]
            — empirically the best baseline (multi-lang lost ~24 F1 on ACCOUNT
            because country-specific regex recognizers like US_SSN are gated to
            language='en' and stop firing when text is routed to other langs).
            Pass list(LANGUAGE_MODELS.keys()) to opt into the multilingual NLP
            engine for the NER-driven categories (PERSON, ADDRESS).
        score_threshold: drop predictions with confidence below this (0..1).
        """
        loaded = languages or ["English"]
        models = [
            {"lang_code": LANGUAGE_MODELS[name][0], "model_name": LANGUAGE_MODELS[name][1]}
            for name in loaded
            if name in LANGUAGE_MODELS
        ]
        provider = NlpEngineProvider(
            nlp_configuration={"nlp_engine_name": "spacy", "models": models}
        )
        nlp_engine = provider.create_engine()
        supported = [m["lang_code"] for m in models]
        self._analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine, supported_languages=supported
        )
        self._lang_to_code = {name: LANGUAGE_MODELS[name][0] for name in loaded}
        self._supported_codes = set(supported)
        self._default_code = "en" if "en" in self._supported_codes else supported[0]
        self._score_threshold = score_threshold
        # Warm each language so first detect() doesn't pay the lazy-load cost.
        for code in supported:
            self._analyzer.analyze(text="warmup", language=code)

    def detect(self, text: str, **context: object) -> DetectorResult:
        # PII-Masking-300k passes language as full name ("Dutch", "English", ...).
        lang_full = str(context.get("language") or "")
        lang_code = self._lang_to_code.get(lang_full, self._default_code)
        if lang_code not in self._supported_codes:
            lang_code = self._default_code
        t0 = time.perf_counter()
        try:
            results = self._analyzer.analyze(
                text=text,
                language=lang_code,
                score_threshold=self._score_threshold,
            )
        except Exception as e:  # noqa: BLE001
            return {"spans": [], "latency_ms": (time.perf_counter() - t0) * 1000, "error": repr(e)}
        latency_ms = (time.perf_counter() - t0) * 1000
        spans: list[Span] = []
        for r in results:
            canonical = presidio_to_canonical(r.entity_type) or r.entity_type.upper()
            spans.append(
                {
                    "label": canonical,
                    "raw_label": r.entity_type,
                    "start": int(r.start),
                    "end": int(r.end),
                    "text": text[r.start : r.end],
                }
            )
        return {"spans": spans, "latency_ms": latency_ms, "error": None}
