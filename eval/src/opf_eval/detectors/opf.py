from __future__ import annotations

import os
import time
from typing import Literal

# Triton has no stable Apple Silicon support — force vanilla PyTorch MoE before
# the OPF runtime imports its kernels.
os.environ.setdefault("OPF_MOE_TRITON", "0")

from opf._api import OPF  # noqa: E402

from ..taxonomy import opf_to_canonical
from .base import DetectorResult, Span


class OPFDetector:
    name = "opf"

    def __init__(
        self,
        *,
        device: Literal["cpu", "cuda", "mps"] = "cpu",
        decode_mode: Literal["viterbi", "argmax"] = "viterbi",
        viterbi_calibration_path: str | None = None,
    ) -> None:
        self._opf = OPF(
            device=device,  # type: ignore[arg-type]  # mps coerced via str()
            decode_mode=decode_mode,
            output_mode="typed",
        )
        if viterbi_calibration_path:
            self._opf.set_viterbi_decoder(calibration_path=viterbi_calibration_path)
        # Warm load so first detect() doesn't include a multi-second cold start.
        self._opf.get_runtime()

    def close(self) -> None:
        """Drop the OPF runtime so its weights can be released. Called by
        the runner between detector iterations to free VRAM."""
        self._opf = None  # type: ignore[assignment]

    def detect(self, text: str, **_context: object) -> DetectorResult:
        t0 = time.perf_counter()
        try:
            result = self._opf.redact(text)
        except Exception as e:  # noqa: BLE001
            return {"spans": [], "latency_ms": (time.perf_counter() - t0) * 1000, "error": repr(e)}
        latency_ms = (time.perf_counter() - t0) * 1000
        spans: list[Span] = []
        for s in result.detected_spans:  # type: ignore[union-attr]
            canonical = opf_to_canonical(s.label) or s.label.upper()
            spans.append(
                {
                    "label": canonical,
                    "raw_label": s.label,
                    "start": int(s.start),
                    "end": int(s.end),
                    "text": s.text,
                }
            )
        return {"spans": spans, "latency_ms": latency_ms, "error": None}
