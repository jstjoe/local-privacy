"""Detector registry — lazy-loaded factories with per-detector locks.

Which detectors exist, how to build them and what labels they produce all
come from the eval harness registry (`opf_eval.detectors.registry`); this
module adds the service concerns: lazy loading off the event loop and one
lock per instance.

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
from collections.abc import Callable
from dataclasses import dataclass, field

from opf_eval.detectors import registry as detector_registry
from opf_eval.detectors.base import Detector
from opf_eval.taxonomy import CANONICAL_LABELS, detector_supported_canonicals

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


def _device() -> str:
    return os.environ.get("OPF_DEVICE", "cpu")


def _options(name: str) -> dict:
    """Env-driven per-detector options for the service."""
    if name in ("opf", "opf_calibrated"):
        return {"decode_mode": os.environ.get("OPF_DECODE_MODE", "viterbi")}
    return {}


def _factory(name: str) -> Callable[[], Detector]:
    def make() -> Detector:
        # No dataset in the service: every detector gets its full vocabulary
        # (all GLiNER prompts, unconstrained Skyflow entity types).
        return detector_registry.build(name, device=_device(), options=_options(name))

    return make


def build_default_registry() -> dict[str, DetectorEntry]:
    """One entry per detector in the eval harness registry that is exposed to
    the API (`in_api`) and whose dependencies are installed. Nothing is
    loaded until first use (or EAGER_LOAD)."""
    reg: dict[str, DetectorEntry] = {}
    for spec in detector_registry.specs():
        if not spec.in_api:
            continue
        if not spec.available():
            logger.info("%s: missing %s; skipping registration", spec.name, spec.missing())
            continue
        reg[spec.name] = DetectorEntry(spec.name, _factory(spec.name), proxy=spec.remote)
    return reg


def detector_categories(name: str) -> list[str]:
    """Canonical (coarse) categories a detector can produce, in taxonomy
    order. Empty list for unknown names."""
    supported = detector_supported_canonicals(name)
    return [c for c in CANONICAL_LABELS if c in supported]
