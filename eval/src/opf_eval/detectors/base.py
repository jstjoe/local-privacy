from __future__ import annotations

import time
from typing import Protocol, TypedDict

try:
    from typing import NotRequired
except ImportError:  # Python 3.10
    from typing_extensions import NotRequired

from ..taxonomy import parent, to_canonical


class Span(TypedDict):
    label: str          # coarse canonical label (what reports score by default)
    raw_label: str      # the detector's / dataset's own label
    start: int
    end: int
    text: str
    fine_label: NotRequired[str]  # most specific canonical label; see taxonomy


class DetectorResult(TypedDict):
    spans: list[Span]
    latency_ms: float
    error: str | None


class Detector(Protocol):
    name: str

    def detect(self, text: str, **context: object) -> DetectorResult:
        """Detect spans in `text`.

        Detectors may accept optional context kwargs (e.g. `language`) that
        the runner passes through from the fixture record. Detectors that
        don't need context should still accept and ignore extra kwargs.
        """
        ...


def make_span(
    vocab: str,
    raw_label: str,
    start: int,
    end: int,
    text: str,
    *,
    keep_unmapped: bool = True,
) -> Span | None:
    """Build a Span from a raw detector label via the `vocab` vocabulary.

    `label` is the coarse canonical label and `fine_label` the most specific
    one. Raw labels the vocabulary doesn't know are kept with their
    upper-cased raw label (scoring ignores them, but they stay visible in the
    raw output) unless `keep_unmapped=False`, which drops them.
    """
    fine = to_canonical(vocab, raw_label)
    if fine is None:
        if not keep_unmapped:
            return None
        fine = raw_label.upper()
    return {
        "label": parent(fine),
        "fine_label": fine,
        "raw_label": raw_label,
        "start": int(start),
        "end": int(end),
        "text": text,
    }


def error_result(t0: float, exc: BaseException) -> DetectorResult:
    """The result a detector returns when inference raised."""
    return {"spans": [], "latency_ms": (time.perf_counter() - t0) * 1000, "error": repr(exc)}
