"""PII detectors.

Build detectors by name through the registry — it knows every shipped
detector, what it needs installed, and which label vocabulary it speaks:

    from opf_eval.detectors import registry
    registry.names()                      # every registered detector
    det = registry.build("gliner", device="cuda")
    det.detect("Email alice@example.com")

The concrete classes are importable too (`from opf_eval.detectors import
OPFDetector`), but are loaded lazily: importing this package never pulls in
torch, transformers, GLiNER or Presidio.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from . import registry
from .base import Detector, DetectorResult, Span, make_span

_LAZY = {
    "Ai4PrivacyDetector": ".ai4privacy",
    "GLiNERDetector": ".gliner",
    "HFTokenClassifierDetector": ".hf_token",
    "OpenMedDetector": ".openmed",
    "OPFDetector": ".opf",
    "PresidioDetector": ".presidio",
    "SkyflowDetector": ".skyflow",
}

if TYPE_CHECKING:
    from .ai4privacy import Ai4PrivacyDetector
    from .gliner import GLiNERDetector
    from .hf_token import HFTokenClassifierDetector
    from .openmed import OpenMedDetector
    from .opf import OPFDetector
    from .presidio import PresidioDetector
    from .skyflow import SkyflowDetector


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Ai4PrivacyDetector",
    "Detector",
    "DetectorResult",
    "GLiNERDetector",
    "HFTokenClassifierDetector",
    "OPFDetector",
    "OpenMedDetector",
    "PresidioDetector",
    "SkyflowDetector",
    "Span",
    "make_span",
    "registry",
]
