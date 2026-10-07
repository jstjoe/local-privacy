"""Registrations for every detector shipped with the harness.

Imported by `registry` on first use. Factories import their heavy
dependencies lazily so listing detectors stays cheap.
"""

from __future__ import annotations

from ..taxonomy import canonical_to_skyflow_request_types, parent
from .base import Detector
from .registry import (
    BuildContext,
    DetectorSpec,
    register_detector,
    register_gliner_model,
)

# ------------------------------------------------------------------- OPF


def _opf(ctx: BuildContext) -> Detector:
    from .opf import OPFDetector

    return OPFDetector(
        device=ctx.device,  # type: ignore[arg-type]
        decode_mode=ctx.options.get("decode_mode", "viterbi"),
        viterbi_calibration_path=ctx.opf_calibration_path,
    )


def _opf_calibrated(ctx: BuildContext) -> Detector:
    if not ctx.opf_calibration_path:
        raise ValueError("opf_calibrated requires --opf-calibration-path")
    return _opf(ctx)


register_detector(DetectorSpec(
    name="opf", vocab="opf", factory=_opf, requires=("opf",),
    description="OpenAI Privacy Filter (default Viterbi decoder)",
    license="Apache-2.0", tags=("local",),
))
register_detector(DetectorSpec(
    name="opf_calibrated", vocab="opf", factory=_opf_calibrated, requires=("opf",),
    description="OPF with a custom Viterbi calibration JSON (--opf-calibration-path)",
    license="Apache-2.0", tags=("local",), in_api=False,
))


# -------------------------------------------------------------- Presidio


def _presidio(multilang: bool):
    def factory(ctx: BuildContext) -> Detector:
        from .presidio import LANGUAGE_MODELS, PresidioDetector

        # Multilang needs every `<lang>_core_news_lg` spaCy model installed.
        languages = list(LANGUAGE_MODELS) if multilang else ["en"]
        return PresidioDetector(languages=languages)

    return factory


register_detector(DetectorSpec(
    name="presidio", vocab="presidio", factory=_presidio(False),
    requires=("presidio_analyzer", "spacy"),
    description="Microsoft Presidio, English (regex + spaCy NER)",
    license="MIT", tags=("local", "rules"),
))
register_detector(DetectorSpec(
    name="presidio_multilang", vocab="presidio", factory=_presidio(True),
    requires=("presidio_analyzer", "spacy"),
    description="Presidio with all 6 spaCy language models",
    license="MIT", tags=("local", "rules"),
))


# ---------------------------------------------------------------- GLiNER

register_gliner_model(
    "gliner", "urchade/gliner_multi_pii-v1",
    description="GLiNER multilingual PII; prompts restricted to the dataset's categories",
    license="Apache-2.0",
)
register_gliner_model(
    "gliner_nvidia", "nvidia/gliner-PII",
    description="NVIDIA gliner-PII (gliner_large-v2.1 base, 570M)",
    license="NVIDIA Open Model License",
)
# Gretel models were trained on a fixed snake_case label set; always prompt
# with all of it rather than a dataset-restricted subset.
register_gliner_model(
    "gliner_gretel_small", "gretelai/gretel-gliner-bi-small-v1.0",
    vocab="gretel", dataset_aware=False,
    description="Gretel bi-encoder GLiNER small (English)",
    license="Apache-2.0",
)
register_gliner_model(
    "gliner_gretel_large", "gretelai/gretel-gliner-bi-large-v1.0",
    vocab="gretel", dataset_aware=False,
    description="Gretel bi-encoder GLiNER large (English)",
    license="Apache-2.0",
)


# ------------------------------------------------- HF token classifiers


def _ai4privacy(ctx: BuildContext) -> Detector:
    from .ai4privacy import Ai4PrivacyDetector

    return Ai4PrivacyDetector(device=ctx.device)


register_detector(DetectorSpec(
    name="ai4privacy_modernbert", vocab="openpii", factory=_ai4privacy,
    requires=("transformers",),
    description="ai4privacy ModernBERT-base OpenPII model, 8 languages",
    license="MIT", tags=("hf", "local"),
))


def _openmed(ctx: BuildContext) -> Detector:
    from .openmed import OpenMedDetector

    return OpenMedDetector(device=ctx.device)


register_detector(DetectorSpec(
    name="openmed", vocab="openmed", factory=_openmed, requires=("openmed",),
    description="OpenMed PII, per-language DeBERTa models",
    license="Apache-2.0", tags=("hf", "local"),
))


# --------------------------------------------------------------- Skyflow


def _skyflow(constrained: bool):
    def factory(ctx: BuildContext) -> Detector:
        from .skyflow import SkyflowDetector

        types = list(ctx.skyflow_entity_types) if ctx.skyflow_entity_types else None
        if constrained and types is None and ctx.dataset_labels is not None:
            # Ask only for the entity types covering what the dataset annotates.
            types = canonical_to_skyflow_request_types(
                sorted({parent(lbl) for lbl in ctx.dataset_labels})
            )
        return SkyflowDetector(entity_types=types)

    return factory


register_detector(DetectorSpec(
    name="skyflow", vocab="skyflow", factory=_skyflow(True), requires=("httpx",),
    remote=True,
    description="Skyflow Detect API; entity types derived from the dataset's categories",
    license="commercial API", tags=("remote",),
))
register_detector(DetectorSpec(
    name="skyflow_full", vocab="skyflow", factory=_skyflow(False), requires=("httpx",),
    remote=True, in_api=False,
    description="Skyflow Detect API, unconstrained (~70 entity types)",
    license="commercial API", tags=("remote",),
))
