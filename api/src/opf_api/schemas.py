from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DecodeMode = Literal["viterbi", "argmax"]

# Four sanitization modes, in increasing strength of identity preservation:
#   redact        -> "********" (fixed-length asterisks; no information leaks)
#   label         -> "[EMAIL]" (default; category label only)
#   label_number  -> "[EMAIL_1]" (per-request counters; duplicates reuse numbers)
#   label_token   -> "[EMAIL_jRc7QGn]" (deterministic Skyflow vault token)
SanitizeMode = Literal["redact", "label", "label_number", "label_token"]


class DetectRequest(BaseModel):
    """Base request for /v1/detect — no text-rewriting fields."""

    text: str
    detector: str | None = Field(
        default=None,
        description="Detector name from /v1/detectors. None = use DEFAULT_DETECTOR env.",
    )
    categories: list[str] | None = Field(
        default=None,
        description=(
            "Canonical categories to keep. Valid: PERSON, EMAIL, PHONE, "
            "ADDRESS, URL, DATE, ACCOUNT, SECRET, USERNAME, DEMOGRAPHIC, "
            "ORGANIZATION, OCCUPATION, MONEY, VEHICLE, PHYSICAL. None = keep all."
        ),
    )
    decode_mode: DecodeMode | None = Field(
        default=None,
        description="OPF-only. Ignored by other detectors.",
    )


class SanitizeRequest(DetectRequest):
    """Request for /v1/sanitize — DetectRequest plus a `mode` field that
    picks how detected spans are rewritten."""

    mode: SanitizeMode = Field(
        default="label",
        description=(
            "`redact` -> `********` (fixed asterisks). "
            "`label` -> `[EMAIL]` (default). "
            "`label_number` -> `[EMAIL_1]` (per-request counter, duplicates reuse number). "
            "`label_token` -> `[EMAIL_jRc7QGn]` (deterministic 7-char Skyflow vault token)."
        ),
    )


class SanitizedSpan(BaseModel):
    label: str
    raw_label: str
    start: int
    end: int
    text: str
    replacement: str


class SummaryOut(BaseModel):
    """Per-response detection summary. `by_label` maps canonical label -> count."""

    span_count: int
    by_label: dict[str, int]


class SanitizeResponse(BaseModel):
    schema_version: int
    detector: str
    mode: SanitizeMode
    text: str
    detected_spans: list[SanitizedSpan]
    sanitized_text: str
    summary: SummaryOut
    warning: str | None = None


class SpanOut(BaseModel):
    """Plain span for /v1/detect — no replacement text."""

    label: str
    raw_label: str
    start: int
    end: int
    text: str


class DetectResponse(BaseModel):
    schema_version: int
    detector: str
    text: str
    detected_spans: list[SpanOut]
    summary: SummaryOut
    warning: str | None = None


class DetectorInfo(BaseModel):
    name: str
    categories: list[str]
    loaded: bool
    proxy: bool


class DetectorsResponse(BaseModel):
    default: str
    detectors: list[DetectorInfo]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    default_detector: str
    loaded_detectors: list[str]
    schema_version: int
