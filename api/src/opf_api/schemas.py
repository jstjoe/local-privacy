from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


DecodeMode = Literal["viterbi", "argmax"]
OutputMode = Literal["typed", "redacted"]


class RedactRequest(BaseModel):
    text: str
    categories: list[str] | None = Field(
        default=None,
        description="Optional allow-list of OPF labels (e.g. private_email). None = keep all.",
    )
    decode_mode: DecodeMode | None = None
    output_mode: OutputMode | None = None


class SpanOut(BaseModel):
    label: str
    start: int
    end: int
    text: str
    placeholder: str


class RedactResponse(BaseModel):
    schema_version: int
    summary: dict
    text: str
    detected_spans: list[SpanOut]
    redacted_text: str
    warning: str | None = None


class DetectResponse(BaseModel):
    schema_version: int
    summary: dict
    text: str
    detected_spans: list[SpanOut]
    warning: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok"]
    checkpoint_path: str
    schema_version: int
    decode_mode: DecodeMode
    output_mode: OutputMode
