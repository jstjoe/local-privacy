from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from opf._api import OPF, DecodeOptions  # noqa: F401
from opf._core.runtime import build_detection_summary

from .schemas import (
    DetectResponse,
    HealthResponse,
    RedactRequest,
    RedactResponse,
)


router = APIRouter()


def _redact_text(text: str, spans) -> str:
    """Re-implement opf._api._redact_text (private) to avoid importing _-prefixed names."""
    if not spans:
        return text
    pieces: list[str] = []
    cursor = 0
    for s in spans:
        pieces.append(text[cursor : s.start])
        pieces.append(s.placeholder)
        cursor = s.end
    pieces.append(text[cursor:])
    return "".join(pieces)


async def _run_redact(request: Request, body: RedactRequest):
    """Shared inference path. Holds the model lock for the duration of one call."""
    opf: OPF = request.app.state.opf
    lock = request.app.state.lock

    if body.output_mode and body.output_mode != opf._output_mode:  # type: ignore[attr-defined]
        # The OPF runtime caches output_mode; switching per-request requires
        # a runtime rebuild. Reject for v1 to keep concurrency simple.
        raise HTTPException(
            status_code=400,
            detail="per-request output_mode override is not supported in v1",
        )

    decode = (
        DecodeOptions(decode_mode=body.decode_mode) if body.decode_mode else None
    )
    async with lock:
        result = opf.redact(body.text, decode=decode)

    spans = list(result.detected_spans)
    if body.categories is not None:
        allowed = set(body.categories)
        spans = [s for s in spans if s.label in allowed]
        redacted_text = _redact_text(result.text, spans)
    else:
        redacted_text = result.redacted_text

    return result, spans, redacted_text


@router.post("/redact", response_model=RedactResponse)
async def redact(request: Request, body: RedactRequest) -> RedactResponse:
    result, spans, redacted_text = await _run_redact(request, body)
    summary = build_detection_summary(
        output_mode=request.app.state.opf._output_mode,  # type: ignore[attr-defined]
        labels=[s.label for s in spans],
        decoded_mismatch=result.summary.get("decoded_mismatch", False),
    )
    return RedactResponse(
        schema_version=result.schema_version,
        summary=dict(summary),
        text=result.text,
        detected_spans=[
            {
                "label": s.label,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "placeholder": s.placeholder,
            }
            for s in spans
        ],
        redacted_text=redacted_text,
        warning=result.warning,
    )


@router.post("/detect", response_model=DetectResponse)
async def detect(request: Request, body: RedactRequest) -> DetectResponse:
    result, spans, _ = await _run_redact(request, body)
    summary = build_detection_summary(
        output_mode=request.app.state.opf._output_mode,  # type: ignore[attr-defined]
        labels=[s.label for s in spans],
        decoded_mismatch=result.summary.get("decoded_mismatch", False),
    )
    return DetectResponse(
        schema_version=result.schema_version,
        summary=dict(summary),
        text=result.text,
        detected_spans=[
            {
                "label": s.label,
                "start": s.start,
                "end": s.end,
                "text": s.text,
                "placeholder": s.placeholder,
            }
            for s in spans
        ],
        warning=result.warning,
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    opf: OPF = request.app.state.opf
    return HealthResponse(
        status="ok",
        checkpoint_path=opf._checkpoint,  # type: ignore[attr-defined]
        schema_version=1,
        decode_mode=opf._decoder_config.decode_mode,  # type: ignore[attr-defined]
        output_mode=opf._output_mode,  # type: ignore[attr-defined]
    )
