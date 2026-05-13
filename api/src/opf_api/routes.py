from __future__ import annotations

import asyncio
from collections import Counter
from typing import Callable, Protocol

from fastapi import APIRouter, HTTPException, Request

from opf_eval.detectors.base import Span
from opf_eval.transforms import (
    VaultTokenError,
    label_number_renderer,
    label_renderer,
    label_token_renderer,
    redact_renderer,
    splice_pieces,
)

from .registry import DetectorEntry, detector_categories
from .schemas import (
    CanonicalLabel,
    DetectorInfo,
    DetectorsResponse,
    DetectorOptions,
    FindRequest,
    FindResponse,
    HealthResponse,
    ReplacedSpan,
    ReplaceRequest,
    ReplaceResponse,
    SpanOut,
)
from .vault_tokens import TokenVaultClient


router = APIRouter()


_ERROR_400_EXAMPLE = {
    "description": "Unknown detector or missing config for the chosen mode.",
    "content": {
        "application/json": {
            "examples": {
                "unknown_detector": {
                    "summary": "Unknown detector",
                    "value": {
                        "detail": "unknown detector 'foo'; available: ['gliner', 'opf', 'presidio']"
                    },
                },
                "label_token_unconfigured": {
                    "summary": "label_token mode without vault env",
                    "value": {
                        "detail": "mode='label_token' requires SKYFLOW_TOKEN_VAULT_URL and SKYFLOW_TOKEN_VAULT_ID env vars to be set"
                    },
                },
            }
        }
    },
}

_ERROR_422_EXAMPLE = {
    "description": (
        "Request body failed Pydantic validation. Common causes: a value in "
        "`categories` that isn't a canonical label, an unknown key in "
        "`options.opf`, or a wrong type on any field."
    ),
    "content": {
        "application/json": {
            "examples": {
                "bad_category": {
                    "summary": "Non-canonical value in `categories`",
                    "value": {
                        "detail": [
                            {
                                "type": "enum",
                                "loc": ["body", "categories", 0],
                                "msg": "Input should be 'PERSON', 'EMAIL', 'PHONE', ...",
                                "input": "FOO",
                            }
                        ]
                    },
                },
                "unknown_opf_option": {
                    "summary": "Unknown key in `options.opf`",
                    "value": {
                        "detail": [
                            {
                                "type": "extra_forbidden",
                                "loc": ["body", "options", "opf", "decod_mode"],
                                "msg": "Extra inputs are not permitted",
                                "input": "argmax",
                            }
                        ]
                    },
                },
            }
        }
    },
}

_ERROR_502_EXAMPLE = {
    "description": "Detector backend failure or vault call failure.",
    "content": {
        "application/json": {
            "examples": {
                "detector_failure": {
                    "summary": "Detector backend errored",
                    "value": {"detail": "skyflow: upstream 503"},
                },
                "vault_failure": {
                    "summary": "Token vault call failed",
                    "value": {"detail": "label_token: vault insert failed: 401"},
                },
            }
        }
    },
}


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


class _FindInput(Protocol):
    text: str
    detector: str | None
    categories: list[CanonicalLabel] | None
    options: DetectorOptions | None


async def _run_find(request: Request, body: _FindInput) -> tuple[str, list[Span]]:
    name, entry = _resolve_detector(request, body.detector)

    detector = await entry.get()
    async with entry.call_lock:
        result = await asyncio.to_thread(detector.detect, body.text)

    if result.get("error"):
        raise HTTPException(status_code=502, detail=f"{name}: {result['error']}")

    spans: list[Span] = list(result.get("spans") or [])
    if body.categories is not None:
        # Empty list is a deliberate "match nothing" filter, not "no filter".
        # Omit the field (or send null) to keep every category. CanonicalLabel
        # inherits str, so membership lookup against the raw detector label
        # works without coercion.
        allow = set(body.categories)
        spans = [s for s in spans if s["label"] in allow]
    return name, spans


@router.post(
    "/find",
    response_model=FindResponse,
    tags=["Find"],
    summary="Find sensitive data in text",
    description=(
        "Run the chosen detector over `text` and return canonical-labelled spans.\n\n"
        "No rewriting is performed — call `/api/replace` for that.\n\n"
        "Filter detector output to a subset of canonical labels with `categories`.\n"
        "OPF-only: pass `options.opf.decode_mode` to override the default Viterbi decoding."
    ),
    response_description="Detected spans plus per-label counts.",
    responses={
        400: _ERROR_400_EXAMPLE,
        422: _ERROR_422_EXAMPLE,
        502: _ERROR_502_EXAMPLE,
    },
)
async def find(request: Request, body: FindRequest) -> FindResponse:
    """Find-only: returns spans, no text rewriting."""
    name, spans = await _run_find(request, body)
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
    return FindResponse(
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


@router.post(
    "/replace",
    response_model=ReplaceResponse,
    tags=["Replace"],
    summary="Replace sensitive data in text",
    description=(
        "Find spans, then rewrite each one under the chosen `mode`. Four modes, "
        "in increasing strength of identity preservation:\n\n"
        "| `mode` | Looks like | What it preserves |\n"
        "|---|---|---|\n"
        "| `redact` | `********` | Nothing — fixed 8-char asterisk run regardless of span length. |\n"
        "| `label` | `[EMAIL]` | Category only. Default. |\n"
        "| `label_number` | `[EMAIL_1]` | Identity **within one request** via per-label counter; "
        "duplicate `(label, text)` reuses its number. Dropped-overlap spans "
        "(`replaced=false`) still consume a counter slot, so the kept sequence "
        "may skip numbers — see the overlap notes below. |\n"
        "| `label_token` | `[EMAIL_MGaE1Bo]` | Identity **across requests and detectors** via a "
        "Skyflow vault. Deterministic — same plaintext maps to the same 7-char token forever. |\n\n"
        "**Overlapping spans:** the earlier-starting span wins; later overlaps are skipped in "
        "`replaced_text` but still appear in `detected_spans` with `replaced=false`. "
        "Filter to `replaced=true` to reconstruct exactly what landed.\n\n"
        "**`label_token` requirements:** `SKYFLOW_TOKEN_VAULT_URL`, `SKYFLOW_TOKEN_VAULT_ID`, "
        "and a bearer (`SKYFLOW_TOKEN_BEARER_TOKEN`, falling back to `SKYFLOW_BEARER_TOKEN`). "
        "The vault must be configured per the token-vault setup guide — one table with one "
        "`tok_<label>` column per canonical label, each `DETERMINISTIC_FPT` with regex "
        "`^[A-Za-z0-9]{7}$`. Spans whose canonical label has no vault column fall back to "
        "`[LABEL]` for that span only."
    ),
    response_description="Replaced text, the spans that were rewritten, and per-label counts.",
    responses={
        400: _ERROR_400_EXAMPLE,
        422: _ERROR_422_EXAMPLE,
        502: _ERROR_502_EXAMPLE,
    },
)
async def replace(request: Request, body: ReplaceRequest) -> ReplaceResponse:
    """Find + rewrite the input text under the chosen `mode`.

    Modes:
      - `redact`       -> fixed-length asterisks (`********`)
      - `label`        -> `[EMAIL]` (default)
      - `label_number` -> `[EMAIL_1]`, per-request counter
      - `label_token`  -> `[EMAIL_jRc7QGn]`, deterministic Skyflow vault token
    """
    name, spans = await _run_find(request, body)
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

    # Render each span exactly once in sorted order. Reuse the rendered
    # strings for both the response (out_spans) and the spliced text via
    # splice_pieces — no implicit dependency on the renderer being
    # idempotent across multiple calls.
    ordered = sorted(spans, key=lambda s: (s["start"], s["end"]))
    rendered_pairs: list[tuple[Span, str]] = [(s, render(s)) for s in ordered]

    # Mirror splice_pieces' overlap rule (earlier-starting span wins; later
    # overlaps are skipped) so each ReplacedSpan can carry a faithful
    # `replaced` flag. Clients filtering to `replaced=true` get exactly the
    # set of spans that landed in `replaced_text`.
    cursor = 0
    replaced_flags: list[bool] = []
    for s, _ in rendered_pairs:
        if s["start"] >= cursor:
            replaced_flags.append(True)
            cursor = s["end"]
        else:
            replaced_flags.append(False)

    out_spans = [
        ReplacedSpan(
            label=s["label"],
            raw_label=s["raw_label"],
            start=s["start"],
            end=s["end"],
            text=s["text"],
            replacement=replacement,
            replaced=flag,
        )
        for (s, replacement), flag in zip(rendered_pairs, replaced_flags)
    ]
    replaced_text = splice_pieces(body.text, rendered_pairs)

    by_label = Counter(s.label for s in out_spans)
    return ReplaceResponse(
        detector=name,
        mode=mode,
        text=body.text,
        detected_spans=out_spans,
        replaced_text=replaced_text,
        summary={"span_count": len(out_spans), "by_label": dict(by_label)},
        warning=None,
    )


@router.get(
    "/detectors",
    response_model=DetectorsResponse,
    tags=["Meta"],
    summary="List registered detectors",
    description=(
        "Return every detector registered in this deployment plus the canonical categories "
        "each can produce. `loaded` flips to `true` after first use (or at startup if the "
        "detector is listed in `EAGER_LOAD`). `proxy=true` means the detector calls an "
        "external service."
    ),
    response_description="Registry snapshot.",
)
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


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Meta"],
    summary="Liveness probe",
    description=(
        "Always returns `200` when the process is up. Does **not** probe detector backends — "
        "use `/api/detectors` to inspect which detectors have been loaded."
    ),
    response_description="Liveness status plus which detectors are currently loaded.",
)
async def health(request: Request) -> HealthResponse:
    registry: dict[str, DetectorEntry] = request.app.state.registry
    loaded = sorted(name for name, e in registry.items() if e.loaded)
    return HealthResponse(
        status="ok",
        default_detector=request.app.state.default_detector,
        loaded_detectors=loaded,
    )
