from __future__ import annotations

import asyncio
from collections import Counter
from typing import Callable, Protocol

from fastapi import APIRouter, HTTPException, Request

from opf_eval.detectors.base import Span
from opf_eval.taxonomy import CANONICAL_LABELS
from opf_eval.transforms import (
    VaultTokenError,
    label_numbered_renderer,
    placeholder_renderer,
    splice_spans,
    vault_token_renderer,
)

from .registry import DetectorEntry, detector_categories
from .schemas import (
    DetectorInfo,
    DetectorsResponse,
    DetectResponse,
    HealthResponse,
    RedactRequest,
    RedactResponse,
    SpanOut,
    TokenizeRequest,
    TokenizeResponse,
    TokenSpanOut,
)
from .vault_tokens import TokenVaultClient


router = APIRouter()


_VALID_CATEGORIES = set(CANONICAL_LABELS)


def _resolve_detector(request: Request, name: str | None) -> tuple[str, DetectorEntry]:
    chosen = name or request.app.state.default_detector
    registry: dict[str, DetectorEntry] = request.app.state.registry
    entry = registry.get(chosen)
    if entry is None:
        raise HTTPException(
            status_code=400,
            detail=f"unknown detector {chosen!r}; available: {sorted(registry)}",
        )
    return chosen, entry


def _validate_categories(categories: list[str] | None) -> set[str] | None:
    if categories is None:
        return None
    bad = [c for c in categories if c not in _VALID_CATEGORIES]
    if bad:
        raise HTTPException(
            status_code=400,
            detail=(
                f"unknown canonical categories: {bad}. "
                f"Valid: {sorted(_VALID_CATEGORIES)}"
            ),
        )
    return set(categories)


def _placeholder_for(span: Span, fmt: str) -> str:
    return placeholder_renderer(fmt)(span)


def _redact_text(text: str, spans: list[Span], fmt: str) -> str:
    return splice_spans(text, spans, placeholder_renderer(fmt))


class _DetectInput(Protocol):
    text: str
    detector: str | None
    categories: list[str] | None
    decode_mode: object


async def _run_detect(request: Request, body: _DetectInput) -> tuple[str, list[Span]]:
    name, entry = _resolve_detector(request, body.detector)
    if body.decode_mode is not None and name != "opf":
        # Silently no-op rather than 400 — plan 06 says ignore for non-OPF.
        pass

    detector = await entry.get()
    async with entry.call_lock:
        result = await asyncio.to_thread(detector.detect, body.text)

    if result.get("error"):
        raise HTTPException(status_code=502, detail=f"{name}: {result['error']}")

    spans: list[Span] = list(result.get("spans") or [])
    allow = _validate_categories(body.categories)
    if allow is not None:
        spans = [s for s in spans if s["label"] in allow]
    return name, spans


def _to_span_out(span: Span, fmt: str, *, with_placeholder: bool) -> SpanOut:
    return SpanOut(
        label=span["label"],
        raw_label=span["raw_label"],
        start=span["start"],
        end=span["end"],
        text=span["text"],
        placeholder=_placeholder_for(span, fmt) if with_placeholder else None,
    )


def _check_opf_native(body: RedactRequest, detector_name: str) -> None:
    if body.placeholder_format == "opf_native" and detector_name != "opf":
        raise HTTPException(
            status_code=400,
            detail="placeholder_format='opf_native' is only valid with detector='opf'",
        )


@router.post("/redact", response_model=RedactResponse)
async def redact(request: Request, body: RedactRequest) -> RedactResponse:
    name, _ = _resolve_detector(request, body.detector)
    _check_opf_native(body, name)
    name, spans = await _run_detect(request, body)
    fmt = body.placeholder_format
    redacted = _redact_text(body.text, spans, fmt)
    by_label = Counter(s["label"] for s in spans)
    return RedactResponse(
        schema_version=request.app.state.schema_version,
        detector=name,
        text=body.text,
        detected_spans=[_to_span_out(s, fmt, with_placeholder=True) for s in spans],
        redacted_text=redacted,
        summary={"span_count": len(spans), "by_label": dict(by_label)},
        warning=None,
    )


@router.post("/detect", response_model=DetectResponse)
async def detect(request: Request, body: RedactRequest) -> DetectResponse:
    name, _ = _resolve_detector(request, body.detector)
    _check_opf_native(body, name)
    name, spans = await _run_detect(request, body)
    by_label = Counter(s["label"] for s in spans)
    return DetectResponse(
        schema_version=request.app.state.schema_version,
        detector=name,
        text=body.text,
        detected_spans=[_to_span_out(s, body.placeholder_format, with_placeholder=False) for s in spans],
        summary={"span_count": len(spans), "by_label": dict(by_label)},
        warning=None,
    )


def _build_vault_token_renderer_raising_http(
    spans: list[Span],
    client: TokenVaultClient,
) -> Callable[[Span], str]:
    """Thin wrapper that converts shared module's `VaultTokenError` into
    the route-level HTTPException(502). Keeps the shared helper free of
    FastAPI types."""
    try:
        return vault_token_renderer(spans, client)
    except VaultTokenError as e:
        raise HTTPException(status_code=502, detail=f"vault_token: {e}")


@router.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(request: Request, body: TokenizeRequest) -> TokenizeResponse:
    name, spans = await _run_detect(request, body)

    fmt = body.token_format
    if fmt == "label":
        render: Callable[[Span], str] = lambda s: f"[{s['label']}]"  # noqa: E731
    elif fmt == "label_numbered":
        render = label_numbered_renderer()
    elif fmt == "vault_token":
        client: TokenVaultClient | None = getattr(
            request.app.state, "token_vault_client", None
        )
        if client is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "token_format='vault_token' requires SKYFLOW_TOKEN_VAULT_URL "
                    "and SKYFLOW_TOKEN_VAULT_ID env vars to be set"
                ),
            )
        # TokenVaultClient uses httpx.Client (sync) so the Skyflow round-trip
        # would block the event loop if called directly. Same pattern as the
        # detector call in _run_detect.
        render = await asyncio.to_thread(
            _build_vault_token_renderer_raising_http, spans, client
        )
    else:  # pragma: no cover — pydantic Literal blocks other values
        raise HTTPException(status_code=400, detail=f"unknown token_format: {fmt!r}")

    # Render in sorted order first so label_numbered's per-call counter
    # assigns numbers in document order, then splice using the same renderer
    # (idempotent — duplicate (label, text) returns the assigned number).
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    out_spans = [
        TokenSpanOut(
            label=s["label"],
            raw_label=s["raw_label"],
            start=s["start"],
            end=s["end"],
            text=s["text"],
            token=render(s),
        )
        for s in ordered
    ]
    tokenized = splice_spans(body.text, spans, render)

    by_label = Counter(s.label for s in out_spans)
    return TokenizeResponse(
        schema_version=request.app.state.schema_version,
        detector=name,
        text=body.text,
        detected_spans=out_spans,
        tokenized_text=tokenized,
        summary={"span_count": len(out_spans), "by_label": dict(by_label)},
        warning=None,
    )


@router.get("/detectors", response_model=DetectorsResponse)
async def list_detectors(request: Request) -> DetectorsResponse:
    registry: dict[str, DetectorEntry] = request.app.state.registry
    items = [
        DetectorInfo(
            name=name,
            categories=detector_categories(name),
            loaded=entry.loaded,
            proxy=entry.proxy,
        )
        for name, entry in sorted(registry.items())
    ]
    return DetectorsResponse(
        default=request.app.state.default_detector, detectors=items
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    registry: dict[str, DetectorEntry] = request.app.state.registry
    loaded = sorted(name for name, e in registry.items() if e.loaded)
    return HealthResponse(
        status="ok",
        default_detector=request.app.state.default_detector,
        loaded_detectors=loaded,
        schema_version=request.app.state.schema_version,
    )
