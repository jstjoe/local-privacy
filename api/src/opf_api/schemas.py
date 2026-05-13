from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from opf_eval.taxonomy import CANONICAL_LABELS


DecodeMode = Literal["viterbi", "argmax"]

# Four sanitization modes, in increasing strength of identity preservation:
#   redact        -> "********" (fixed-length asterisks; no information leaks)
#   label         -> "[EMAIL]" (default; category label only)
#   label_number  -> "[EMAIL_1]" (per-request counters; duplicates reuse numbers)
#   label_token   -> "[EMAIL_jRc7QGn]" (deterministic Skyflow vault token)
SanitizeMode = Literal["redact", "label", "label_number", "label_token"]


class CanonicalLabel(str, Enum):
    """Canonical PII category. Every detector's raw output maps into this taxonomy."""

    PERSON = "PERSON"
    EMAIL = "EMAIL"
    PHONE = "PHONE"
    ADDRESS = "ADDRESS"
    URL = "URL"
    DATE = "DATE"
    ACCOUNT = "ACCOUNT"
    SECRET = "SECRET"
    USERNAME = "USERNAME"
    DEMOGRAPHIC = "DEMOGRAPHIC"
    ORGANIZATION = "ORGANIZATION"
    OCCUPATION = "OCCUPATION"
    MONEY = "MONEY"
    VEHICLE = "VEHICLE"
    PHYSICAL = "PHYSICAL"


# Guardrail: enum must mirror opf_eval.taxonomy.CANONICAL_LABELS exactly.
# Failing here at import time is loud and immediate.
assert {m.value for m in CanonicalLabel} == set(CANONICAL_LABELS), (
    "CanonicalLabel enum drifted from opf_eval.taxonomy.CANONICAL_LABELS: "
    f"enum={sorted(m.value for m in CanonicalLabel)} "
    f"taxonomy={sorted(CANONICAL_LABELS)}"
)


class OpfOptions(BaseModel):
    """Options that only apply when `detector` is `opf`."""

    model_config = ConfigDict(extra="forbid")  # catch typos like `decod_mode`

    decode_mode: DecodeMode = Field(
        default="viterbi",
        description=(
            "OPF decode strategy. `viterbi` (default) maximises sequence probability; "
            "`argmax` picks the most likely label per token independently. "
            "**Currently advisory** — the OPF detector reads its decode mode from the "
            "server's `OPF_DECODE_MODE` env at startup; per-request override is "
            "reserved for a follow-up that extends `OPFDetector.detect()`."
        ),
        examples=["viterbi"],
    )


class DetectorOptions(BaseModel):
    """Per-detector options namespace. Each key carries options for that one
    detector. Only the entry matching the top-level `detector` is consulted —
    other keys are accepted but ignored, so a client can carry one options
    blob across detector swaps without restructuring it."""

    opf: OpfOptions | None = Field(
        default=None,
        description="Options for the `opf` detector. Ignored by other detectors.",
    )


class DetectRequest(BaseModel):
    """Base request for `/v1/detect` — no text-rewriting fields."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Email joe@example.com about the trip to Elgin, TX.",
                    "detector": "presidio",
                    "categories": ["EMAIL"],
                }
            ]
        }
    )

    text: str = Field(
        ...,
        description="Free-form text to scan for PII.",
        examples=["Email joe@example.com about the trip to Elgin, TX."],
    )
    detector: str | None = Field(
        default=None,
        description=(
            "Detector name from `GET /v1/detectors`. Omit to use the server's "
            "`DEFAULT_DETECTOR` (set via env, typically `opf`)."
        ),
        examples=["presidio", "opf", "gliner"],
    )
    categories: list[CanonicalLabel] | None = Field(
        default=None,
        description=(
            "Canonical categories to keep. Omit or set to `null` to keep every "
            "category the detector produces. An empty list `[]` is a deliberate "
            "\"match nothing\" filter and returns zero spans — use `null` if you "
            "mean \"no filter\". Values outside the canonical taxonomy are "
            "rejected with `422`."
        ),
        examples=[["EMAIL", "PHONE"]],
    )
    options: DetectorOptions | None = Field(
        default=None,
        description=(
            "Per-detector options namespace. The entry matching the top-level "
            "`detector` is consulted; other entries are accepted but ignored."
        ),
        examples=[{"opf": {"decode_mode": "argmax"}}],
    )


class SanitizeRequest(DetectRequest):
    """Request for `/v1/sanitize` — `DetectRequest` plus a `mode` field that picks
    how detected spans are rewritten."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "text": "Email alice@x.com or call +1-415-555-0100.",
                    "detector": "presidio",
                    "mode": "label_token",
                }
            ]
        }
    )

    mode: SanitizeMode = Field(
        default="label",
        description=(
            "How to rewrite each detected span. "
            "`redact` -> `********` (fixed asterisks). "
            "`label` -> `[EMAIL]` (default). "
            "`label_number` -> `[EMAIL_1]` (per-request counter; duplicates reuse the number). "
            "`label_token` -> `[EMAIL_jRc7QGn]` (deterministic 7-char Skyflow vault token; "
            "requires `SKYFLOW_TOKEN_VAULT_*` env)."
        ),
        examples=["label", "label_token"],
    )


class SanitizedSpan(BaseModel):
    """One detected span plus the string it was rewritten to in `sanitized_text`."""

    label: str = Field(
        ...,
        description="Canonical label for the span (e.g. `EMAIL`, `PERSON`).",
        examples=["EMAIL"],
    )
    raw_label: str = Field(
        ...,
        description="Detector's native label before mapping into the canonical taxonomy.",
        examples=["EMAIL_ADDRESS"],
    )
    start: int = Field(..., description="Inclusive character offset in `text`.", examples=[6])
    end: int = Field(..., description="Exclusive character offset in `text`.", examples=[17])
    text: str = Field(
        ..., description="Substring of the original input.", examples=["alice@x.com"]
    )
    replacement: str = Field(
        ...,
        description="The string this span was rewritten to in `sanitized_text`.",
        examples=["[EMAIL_MGaE1Bo]"],
    )


class SummaryOut(BaseModel):
    """Per-response detection summary. `by_label` maps canonical label to count."""

    span_count: int = Field(
        ..., description="Total number of spans detected.", examples=[2]
    )
    by_label: dict[str, int] = Field(
        ...,
        description="Count of spans per canonical label.",
        examples=[{"EMAIL": 1, "PHONE": 1}],
    )


class SanitizeResponse(BaseModel):
    """Response from `/v1/sanitize`."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "detector": "presidio",
                    "mode": "label_token",
                    "text": "Email alice@x.com or call +1-415-555-0100.",
                    "sanitized_text": "Email [EMAIL_MGaE1Bo] or call [PHONE_vRXiWKZ].",
                    "detected_spans": [
                        {
                            "label": "EMAIL",
                            "raw_label": "EMAIL_ADDRESS",
                            "start": 6,
                            "end": 17,
                            "text": "alice@x.com",
                            "replacement": "[EMAIL_MGaE1Bo]",
                        },
                        {
                            "label": "PHONE",
                            "raw_label": "PHONE_NUMBER",
                            "start": 27,
                            "end": 42,
                            "text": "+1-415-555-0100",
                            "replacement": "[PHONE_vRXiWKZ]",
                        },
                    ],
                    "summary": {"span_count": 2, "by_label": {"EMAIL": 1, "PHONE": 1}},
                    "warning": None,
                }
            ]
        }
    )

    detector: str = Field(
        ...,
        description="Detector that produced the spans.",
        examples=["presidio"],
    )
    mode: SanitizeMode = Field(
        ..., description="Echo of the request `mode`.", examples=["label_token"]
    )
    text: str = Field(
        ..., description="Echo of the request `text`.", examples=["Email alice@x.com."]
    )
    detected_spans: list[SanitizedSpan] = Field(
        ..., description="Spans detected by the chosen detector, with their replacements."
    )
    sanitized_text: str = Field(
        ...,
        description=(
            "`text` with each detected span replaced by its `replacement`. "
            "Overlapping spans: the earlier-starting span wins; later overlaps are skipped "
            "here (they still appear in `detected_spans`)."
        ),
        examples=["Email [EMAIL_MGaE1Bo]."],
    )
    summary: SummaryOut
    warning: str | None = Field(
        default=None,
        description="Non-fatal warning surfaced by the detector backend, if any.",
    )


class SpanOut(BaseModel):
    """Plain span for `/v1/detect` — no replacement text."""

    label: str = Field(..., description="Canonical label.", examples=["EMAIL"])
    raw_label: str = Field(
        ...,
        description="Detector's native label before mapping into the canonical taxonomy.",
        examples=["EMAIL_ADDRESS"],
    )
    start: int = Field(..., description="Inclusive character offset in `text`.", examples=[6])
    end: int = Field(..., description="Exclusive character offset in `text`.", examples=[21])
    text: str = Field(
        ..., description="Substring of the original input.", examples=["joe@example.com"]
    )


class DetectResponse(BaseModel):
    """Response from `/v1/detect`."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "detector": "presidio",
                    "text": "Email joe@example.com about the trip to Elgin, TX.",
                    "detected_spans": [
                        {
                            "label": "EMAIL",
                            "raw_label": "EMAIL_ADDRESS",
                            "start": 6,
                            "end": 21,
                            "text": "joe@example.com",
                        }
                    ],
                    "summary": {"span_count": 1, "by_label": {"EMAIL": 1}},
                    "warning": None,
                }
            ]
        }
    )

    detector: str = Field(..., description="Detector that produced the spans.")
    text: str = Field(..., description="Echo of the request `text`.")
    detected_spans: list[SpanOut] = Field(
        ..., description="Spans detected by the chosen detector."
    )
    summary: SummaryOut
    warning: str | None = Field(
        default=None,
        description="Non-fatal warning surfaced by the detector backend, if any.",
    )


class DetectorInfo(BaseModel):
    """One row in the detector registry."""

    name: str = Field(..., description="Registry key.", examples=["opf"])
    categories: list[CanonicalLabel] = Field(
        ...,
        description="Canonical categories this detector can produce.",
        examples=[["PERSON", "EMAIL", "PHONE"]],
    )
    loaded: bool = Field(
        ...,
        description=(
            "`true` once the detector has been initialised. Flips on first use, "
            "or at startup if listed in `EAGER_LOAD`."
        ),
        examples=[True],
    )
    proxy: bool = Field(
        ...,
        description="`true` if the detector calls an external service (e.g. Skyflow).",
        examples=[False],
    )


class DetectorsResponse(BaseModel):
    """Response from `/v1/detectors`."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "default": "opf",
                    "detectors": [
                        {
                            "name": "gliner",
                            "categories": ["PERSON", "EMAIL"],
                            "loaded": False,
                            "proxy": False,
                        },
                        {
                            "name": "opf",
                            "categories": ["PERSON", "EMAIL", "PHONE"],
                            "loaded": True,
                            "proxy": False,
                        },
                        {
                            "name": "skyflow",
                            "categories": ["PERSON", "EMAIL"],
                            "loaded": False,
                            "proxy": True,
                        },
                    ],
                }
            ]
        }
    )

    default: str = Field(
        ...,
        description="Detector used when a request omits the `detector` field.",
        examples=["opf"],
    )
    detectors: list[DetectorInfo]


class HealthResponse(BaseModel):
    """Response from `/v1/health`."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "default_detector": "opf",
                    "loaded_detectors": ["opf"],
                }
            ]
        }
    )

    status: Literal["ok"] = Field(..., description="Always `ok` when this endpoint responds.")
    default_detector: str = Field(
        ..., description="Detector used when requests omit `detector`.", examples=["opf"]
    )
    loaded_detectors: list[str] = Field(
        ...,
        description="Names of detectors that have been initialised so far.",
        examples=[["opf"]],
    )
