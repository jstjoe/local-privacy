"""Detector registry — lazy-loaded factories with per-detector locks.

In-process detectors (OPF, GLiNER, Presidio) hold a single instance per worker;
the asyncio.Lock serializes inference on that instance. Skyflow is a stateless
HTTP proxy and the lock is essentially a no-op (httpx.Client is thread-safe).

Detectors are loaded on first use unless EAGER_LOAD lists them. Loading OPF
takes 5-30s and ~2.8 GB; defer until needed.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Callable

from opf_eval.detectors.base import Detector
from opf_eval.taxonomy import CANONICAL_MAP


logger = logging.getLogger("opf_api.registry")


@dataclass
class DetectorEntry:
    name: str
    factory: Callable[[], Detector]
    proxy: bool = False
    instance: Detector | None = None
    load_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    call_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def loaded(self) -> bool:
        return self.instance is not None

    async def get(self) -> Detector:
        """Return the detector instance, loading it under load_lock if needed."""
        if self.instance is not None:
            return self.instance
        async with self.load_lock:
            if self.instance is None:
                logger.info("loading detector: %s", self.name)
                # Run the (potentially slow, blocking) factory off the event loop.
                self.instance = await asyncio.to_thread(self.factory)
                logger.info("detector ready: %s", self.name)
        return self.instance


def _opf_factory() -> Detector:
    from opf_eval.detectors.opf import OPFDetector

    return OPFDetector(
        device=os.environ.get("OPF_DEVICE", "cpu"),  # type: ignore[arg-type]
        decode_mode=os.environ.get("OPF_DECODE_MODE", "viterbi"),  # type: ignore[arg-type]
    )


def _skyflow_factory(entity_types: list[str] | None = None) -> Callable[[], Detector]:
    def make() -> Detector:
        from opf_eval.detectors.skyflow import SkyflowDetector

        return SkyflowDetector(entity_types=entity_types)

    return make


def _presidio_factory(*, multilang: bool) -> Callable[[], Detector]:
    def make() -> Detector:
        from opf_eval.detectors.presidio import LANGUAGE_MODELS, PresidioDetector

        languages = list(LANGUAGE_MODELS.keys()) if multilang else ["en"]
        return PresidioDetector(languages=languages)

    return make


def _device() -> str:
    return os.environ.get("OPF_DEVICE", "cpu")


def _gliner_factory() -> Detector:
    from opf_eval.detectors.gliner import GLiNERDetector

    return GLiNERDetector(device=_device())  # type: ignore[arg-type]


def _gliner_nvidia_factory() -> Detector:
    from opf_eval.detectors.gliner import GLiNERDetector

    return GLiNERDetector(
        model_name="nvidia/gliner-PII",
        threshold=0.3,
        name="gliner_nvidia",
        device=_device(),  # type: ignore[arg-type]
    )


def _gliner_gretel_factory(*, size: str) -> Callable[[], Detector]:
    def make() -> Detector:
        from opf_eval.detectors.gliner import GLiNERDetector
        from opf_eval.taxonomy import gretel_prompts, gretel_to_canonical

        return GLiNERDetector(
            model_name=f"gretelai/gretel-gliner-bi-{size}-v1.0",
            threshold=0.7,
            prompts=gretel_prompts(),
            label_to_canonical=gretel_to_canonical,
            name=f"gliner_gretel_{size}",
            device=_device(),  # type: ignore[arg-type]
        )

    return make


def _ai4privacy_factory() -> Detector:
    from opf_eval.detectors.ai4privacy import Ai4PrivacyDetector

    return Ai4PrivacyDetector(device=_device())  # type: ignore[arg-type]


def build_default_registry() -> dict[str, DetectorEntry]:
    """Construct the registry. Optional backends (gliner, presidio,
    ai4privacy_modernbert) are registered only if their imports succeed.
    Skyflow is always available since it's just an HTTP client (creds
    checked at first call)."""
    reg: dict[str, DetectorEntry] = {
        "opf": DetectorEntry("opf", _opf_factory),
        "skyflow": DetectorEntry("skyflow", _skyflow_factory(None), proxy=True),
    }
    try:
        import presidio_analyzer  # noqa: F401

        reg["presidio"] = DetectorEntry("presidio", _presidio_factory(multilang=False))
        reg["presidio_multilang"] = DetectorEntry(
            "presidio_multilang", _presidio_factory(multilang=True)
        )
    except ImportError:
        logger.info("presidio not installed; skipping registration")
    try:
        import gliner  # noqa: F401

        reg["gliner"] = DetectorEntry("gliner", _gliner_factory)
        reg["gliner_nvidia"] = DetectorEntry("gliner_nvidia", _gliner_nvidia_factory)
        reg["gliner_gretel_small"] = DetectorEntry(
            "gliner_gretel_small", _gliner_gretel_factory(size="small")
        )
        reg["gliner_gretel_large"] = DetectorEntry(
            "gliner_gretel_large", _gliner_gretel_factory(size="large")
        )
    except ImportError:
        logger.info("gliner not installed; skipping registration")
    try:
        import transformers  # noqa: F401

        reg["ai4privacy_modernbert"] = DetectorEntry(
            "ai4privacy_modernbert", _ai4privacy_factory
        )
    except ImportError:
        logger.info("transformers not installed; ai4privacy_modernbert skipped")
    return reg


_SOURCE_KEY = {
    "opf": "opf",
    "skyflow": "skyflow",
    "presidio": "presidio",
    "presidio_multilang": "presidio",
    "gliner": "gliner",
    "gliner_nvidia": "gliner",
    "gliner_gretel_small": "gretel",
    "gliner_gretel_large": "gretel",
    "ai4privacy_modernbert": "openpii",
}


def detector_categories(name: str) -> list[str]:
    """Canonical categories a detector can produce.

    Variants (presidio_multilang) share their parent's coverage. Returns
    empty list for unknown names.
    """
    source = _SOURCE_KEY.get(name)
    if source is None:
        return []
    return [
        canonical
        for canonical, by_source in CANONICAL_MAP.items()
        if by_source.get(source)
    ]
