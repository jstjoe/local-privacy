from __future__ import annotations

from typing import Protocol, TypedDict


class Span(TypedDict):
    label: str
    raw_label: str
    start: int
    end: int
    text: str


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
