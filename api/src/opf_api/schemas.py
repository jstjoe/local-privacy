from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .limits import text_max_length


DecodeMode = Literal["viterbi", "argmax"]
PlaceholderFormat = Literal["bracket", "opf_native"]


class RedactRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=text_max_length())
    detector: str | None = Field(
        default=None,
        description="Detector name from /v1/detectors. None = use DEFAULT_DETECTOR env.",
    )
    categories: list[str] | None = Field(
        default=None,
        description=(
            "Canonical categories to keep (PERSON, EMAIL, PHONE, ADDRESS, URL, "
            "DATE, ACCOUNT, SECRET, USERNAME, DEMOGRAPHIC). None = keep all."
        ),
    )
    decode_mode: DecodeMode | None = Field(
        default=None,
        description="OPF-only. Ignored by other detectors.",
    )
    placeholder_format: PlaceholderFormat = Field(
        default="bracket",
        description="`bracket` = `[CATEGORY]`. `opf_native` only valid for OPF.",
    )


class SpanOut(BaseModel):
    label: str
    raw_label: str
    start: int
    end: int
    text: str
    placeholder: str | None = None


class RedactResponse(BaseModel):
    schema_version: int
    detector: str
    text: str
    detected_spans: list[SpanOut]
    redacted_text: str
    summary: dict
    warning: str | None = None


class DetectResponse(BaseModel):
    schema_version: int
    detector: str
    text: str
    detected_spans: list[SpanOut]
    summary: dict
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


class ReadyResponse(BaseModel):
    status: Literal["ok", "not_ready"]
    eager_loaded: list[str]
    eager_pending: list[str]
    skyflow_credentials: bool | None = None
    skyflow_reason: str | None = None
    schema_version: int


# --- Legacy v0 schemas for back-compat /redact, /detect, /health ---

LegacyOutputMode = Literal["typed", "redacted"]


class LegacyRedactRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=text_max_length())
    categories: list[str] | None = Field(
        default=None,
        description="Legacy: raw OPF labels (private_email, etc.). None = keep all.",
    )
    decode_mode: DecodeMode | None = None
    output_mode: LegacyOutputMode | None = None


class LegacySpanOut(BaseModel):
    label: str
    start: int
    end: int
    text: str
    placeholder: str


class LegacyRedactResponse(BaseModel):
    schema_version: int
    summary: dict
    text: str
    detected_spans: list[LegacySpanOut]
    redacted_text: str
    warning: str | None = None


class LegacyDetectResponse(BaseModel):
    schema_version: int
    summary: dict
    text: str
    detected_spans: list[LegacySpanOut]
    warning: str | None = None


class LegacyHealthResponse(BaseModel):
    status: Literal["ok"]
    checkpoint_path: str
    schema_version: int
    decode_mode: DecodeMode
    output_mode: LegacyOutputMode
