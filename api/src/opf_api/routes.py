from __future__ import annotations

import asyncio
from collections import Counter
from typing import Callable, Protocol

from fastapi import APIRouter, HTTPException, Request

from opf_eval.detectors.base import Span
from opf_eval.taxonomy import CANONICAL_LABELS
from opf_eval.transforms import (
    VaultTokenError,
    label_number_renderer,
    label_renderer,
    label_token_renderer,
    redact_renderer,
    splice_spans,
)

from .registry import DetectorEntry, detector_categories
from .schemas import (
    DetectorInfo,
    DetectorsResponse,
    DetectResponse,
    HealthResponse,
    SanitizedSpan,
    SanitizeRequest,
    SanitizeResponse,
    SpanOut,
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


class _DetectInput(Protocol):
    text: str
    detector: str | None
    categories: list[str] | None
    decode_mode: object


async def _run_detect(request: Request, body: _DetectInput) -> tuple[str, list[Span]]:
    name, entry = _resolve_detector(request, body.detector)
    if body.decode_mode is not None and name != "opf":
        # Silently no-op rather than 400 — ignore for non-OPF.
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


@router.post("/detect", response_model=DetectResponse)
async def detect(request: Request, body: SanitizeRequest) -> DetectResponse:
    """Detect-only: returns spans, no text rewriting."""
    name, spans = await _run_detect(request, body)
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    out_spans = [
        SpanOut(
            label=s["label"],
            raw_label=s["raw_label"],
            start=s["start"],
            end=s["end"],
            text=s["text"],
        )
        for s in ordered
    ]
    by_label = Counter(s.label for s in out_spans)
    return DetectResponse(
        schema_version=request.app.state.schema_version,
        detector=name,
        text=body.text,
        detected_spans=out_spans,
        summary={"span_count": len(out_spans), "by_label": dict(by_label)},
        warning=None,
    )


def _build_label_token_renderer_raising_http(
    spans: list[Span],
    client: TokenVaultClient,
) -> Callable[[Span], str]:
    """Thin wrapper that converts the shared module's `VaultTokenError`
    into a 502 HTTPException. Keeps the shared helper free of FastAPI."""
    try:
        return label_token_renderer(spans, client)
    except VaultTokenError as e:
        raise HTTPException(status_code=502, detail=f"label_token: {e}")


@router.post("/sanitize", response_model=SanitizeResponse)
async def sanitize(request: Request, body: SanitizeRequest) -> SanitizeResponse:
    """Detect + rewrite the input text under the chosen `mode`.

    Modes:
      - `redact`       -> fixed-length asterisks (`********`)
      - `label`        -> `[EMAIL]` (default)
      - `label_number` -> `[EMAIL_1]`, per-request counter
      - `label_token`  -> `[EMAIL_jRc7QGn]`, deterministic Skyflow vault token
    """
    name, spans = await _run_detect(request, body)
    mode = body.mode

    if mode == "redact":
        render: Callable[[Span], str] = redact_renderer()
    elif mode == "label":
        render = label_renderer()
    elif mode == "label_number":
        render = label_number_renderer()
    elif mode == "label_token":
        client: TokenVaultClient | None = getattr(
            request.app.state, "token_vault_client", None
        )
        if client is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "mode='label_token' requires SKYFLOW_TOKEN_VAULT_URL "
                    "and SKYFLOW_TOKEN_VAULT_ID env vars to be set"
                ),
            )
        # TokenVaultClient uses httpx.Client (sync) — wrap in to_thread so
        # the Skyflow round-trip doesn't block the event loop.
        render = await asyncio.to_thread(
            _build_label_token_renderer_raising_http, spans, client
        )
    else:  # pragma: no cover — pydantic Literal blocks other values
        raise HTTPException(status_code=400, detail=f"unknown mode: {mode!r}")

    # Render in sorted order first so label_number's per-call counter assigns
    # numbers in document order, then splice using the same renderer
    # (idempotent — duplicate (label, text) returns the assigned number).
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    out_spans = [
        SanitizedSpan(
            label=s["label"],
            raw_label=s["raw_label"],
            start=s["start"],
            end=s["end"],
            text=s["text"],
            replacement=render(s),
        )
        for s in ordered
    ]
    sanitized = splice_spans(body.text, spans, render)

    by_label = Counter(s.label for s in out_spans)
    return SanitizeResponse(
        schema_version=request.app.state.schema_version,
        detector=name,
        mode=mode,
        text=body.text,
        detected_spans=out_spans,
        sanitized_text=sanitized,
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
