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

import logging
import time
from contextlib import contextmanager

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from ..taxonomy import presidio_to_canonical
from .base import DetectorResult, Span


@contextmanager
def _silence_presidio_registry_warnings():
    """Suppress the noisy per-recognizer "language not supported" WARNINGs
    that presidio-analyzer emits during AnalyzerEngine construction —
    one per built-in non-English recognizer when the registry is set up
    English-only. They're informational, not actionable; not worth
    surfacing to the demo audience."""
    logger = logging.getLogger("presidio-analyzer")
    prev_level = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(prev_level)


# Keyed on ISO 639-1 code (matches what the fixtures emit; new ai4privacy
# datasets already use ISO and legacy 300k is normalised to ISO at fixture-
# write time). The `display` form is for documentation / error messages.
LANGUAGE_MODELS: dict[str, tuple[str, str]] = {
    # iso  -> (spaCy model, display)
    "en": ("en_core_web_lg", "English"),
    "nl": ("nl_core_news_lg", "Dutch"),
    "fr": ("fr_core_news_lg", "French"),
    "de": ("de_core_news_lg", "German"),
    "it": ("it_core_news_lg", "Italian"),
    "es": ("es_core_news_lg", "Spanish"),
}

# Legacy full-name aliases for backwards compatibility with old fixtures
# that stored language as "English" / "Dutch" / etc. New fixtures use ISO.
_LEGACY_FULLNAME_TO_ISO = {
    display: iso for iso, (_, display) in LANGUAGE_MODELS.items()
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
        languages: subset of LANGUAGE_MODELS keys (ISO codes) to load.
            Default `["en"]` — empirically the best baseline (multi-lang lost
            ~24 F1 on ACCOUNT because country-specific regex recognizers like
            US_SSN are gated to language='en' and stop firing when text is
            routed to other langs). Pass `list(LANGUAGE_MODELS.keys())` to opt
            into the multilingual NLP engine for the NER-driven categories
            (PERSON, ADDRESS).
        score_threshold: drop predictions with confidence below this (0..1).
        """
        loaded = languages or ["en"]
        models = [
            {"lang_code": code, "model_name": LANGUAGE_MODELS[code][0]}
            for code in loaded
            if code in LANGUAGE_MODELS
        ]
        # spaCy NER emits a number of label types (FAC, CARDINAL, PRODUCT,
        # WORK_OF_ART, ORG, etc.) that Presidio doesn't translate to its
        # entity vocabulary. By default Presidio logs a WARNING per record
        # and keeps them anyway — we drop them at scoring time so they're
        # noise. List them here so Presidio drops them upstream instead.
        provider = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": models,
                "ner_model_configuration": {
                    "labels_to_ignore": [
                        "O", "ORG", "ORGANIZATION", "FAC", "FACILITY",
                        "GPE", "EVENT", "LANGUAGE", "LAW", "MONEY",
                        "NORP", "ORDINAL", "PERCENT", "PRODUCT", "QUANTITY",
                        "WORK_OF_ART", "CARDINAL", "MISC",
                    ],
                },
            }
        )
        nlp_engine = provider.create_engine()
        supported = [m["lang_code"] for m in models]
        with _silence_presidio_registry_warnings():
            self._analyzer = AnalyzerEngine(
                nlp_engine=nlp_engine, supported_languages=supported
            )
        self._supported_codes = set(supported)
        self._default_code = "en" if "en" in self._supported_codes else supported[0]
        self._score_threshold = score_threshold
        # Warm each language so first detect() doesn't pay the lazy-load cost.
        for code in supported:
            self._analyzer.analyze(text="warmup", language=code)

    def detect(self, text: str, **context: object) -> DetectorResult:
        # New fixtures pass ISO ("en", "nl", "hu"); legacy fixtures pass full
        # names ("English"). Translate the latter for back-compat.
        lang_raw = str(context.get("language") or "")
        lang_code = (
            _LEGACY_FULLNAME_TO_ISO.get(lang_raw)
            or (lang_raw.lower() if len(lang_raw) == 2 else None)
            or self._default_code
        )
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
