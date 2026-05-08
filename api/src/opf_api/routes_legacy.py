"""Legacy v0 routes — `/redact`, `/detect`, `/health` — deprecated.

Always route to OPF (regardless of DEFAULT_DETECTOR) since the v0 contract
exposed OPF-native labels (`private_email`) and placeholders (`<PRIVATE_EMAIL>`).
Sets a `Deprecation: true` response header — clients should migrate to /v1/.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request, Response

from opf._api import DecodeOptions
from opf._core.runtime import build_detection_summary

from .schemas import (
    LegacyDetectResponse,
    LegacyHealthResponse,
    LegacyRedactRequest,
    LegacyRedactResponse,
    LegacySpanOut,
)


legacy_router = APIRouter()
DEPRECATION_HEADER = {"Deprecation": "true", "Link": '</v1/redact>; rel="successor-version"'}


def _redact_text_opf(text: str, spans) -> str:
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


async def _legacy_run(request: Request, body: LegacyRedactRequest):
    registry = request.app.state.registry
    entry = registry.get("opf")
    if entry is None:
        raise HTTPException(status_code=503, detail="OPF detector not registered")
    detector = await entry.get()
    opf = detector._opf  # type: ignore[attr-defined]

    if body.output_mode and body.output_mode != opf._output_mode:  # type: ignore[attr-defined]
        raise HTTPException(
            status_code=400,
            detail="per-request output_mode override is not supported",
        )
    decode = DecodeOptions(decode_mode=body.decode_mode) if body.decode_mode else None

    async with entry.call_lock:
        result = await asyncio.to_thread(opf.redact, body.text, decode=decode)

    spans = list(result.detected_spans)
    if body.categories is not None:
        allowed = set(body.categories)
        spans = [s for s in spans if s.label in allowed]
        redacted_text = _redact_text_opf(result.text, spans)
    else:
        redacted_text = result.redacted_text
    return result, spans, redacted_text


@legacy_router.post("/redact", response_model=LegacyRedactResponse, deprecated=True)
async def legacy_redact(
    request: Request, body: LegacyRedactRequest, response: Response
) -> LegacyRedactResponse:
    result, spans, redacted_text = await _legacy_run(request, body)
    response.headers.update(DEPRECATION_HEADER)
    summary = build_detection_summary(
        output_mode=request.app.state.registry["opf"].instance._opf._output_mode,  # type: ignore[union-attr,attr-defined]
        labels=[s.label for s in spans],
        decoded_mismatch=result.summary.get("decoded_mismatch", False),
    )
    return LegacyRedactResponse(
        schema_version=result.schema_version,
        summary=dict(summary),
        text=result.text,
        detected_spans=[
            LegacySpanOut(
                label=s.label, start=s.start, end=s.end, text=s.text, placeholder=s.placeholder
            )
            for s in spans
        ],
        redacted_text=redacted_text,
        warning=result.warning,
    )


@legacy_router.post("/detect", response_model=LegacyDetectResponse, deprecated=True)
async def legacy_detect(
    request: Request, body: LegacyRedactRequest, response: Response
) -> LegacyDetectResponse:
    result, spans, _ = await _legacy_run(request, body)
    response.headers.update(DEPRECATION_HEADER)
    summary = build_detection_summary(
        output_mode=request.app.state.registry["opf"].instance._opf._output_mode,  # type: ignore[union-attr,attr-defined]
        labels=[s.label for s in spans],
        decoded_mismatch=result.summary.get("decoded_mismatch", False),
    )
    return LegacyDetectResponse(
        schema_version=result.schema_version,
        summary=dict(summary),
        text=result.text,
        detected_spans=[
            LegacySpanOut(
                label=s.label, start=s.start, end=s.end, text=s.text, placeholder=s.placeholder
            )
            for s in spans
        ],
        warning=result.warning,
    )


@legacy_router.get("/health", response_model=LegacyHealthResponse, deprecated=True)
async def legacy_health(request: Request, response: Response) -> LegacyHealthResponse:
    entry = request.app.state.registry.get("opf")
    if entry is None or not entry.loaded:
        raise HTTPException(status_code=503, detail="OPF not loaded; eager-load it via EAGER_LOAD")
    opf = entry.instance._opf  # type: ignore[union-attr,attr-defined]
    response.headers.update(DEPRECATION_HEADER)
    return LegacyHealthResponse(
        status="ok",
        checkpoint_path=opf._checkpoint,  # type: ignore[attr-defined]
        schema_version=1,
        decode_mode=opf._decoder_config.decode_mode,  # type: ignore[attr-defined]
        output_mode=opf._output_mode,  # type: ignore[attr-defined]
    )
