"""Body-size guard + per-request timeout helpers.

Pydantic enforces `text` max length (returns 422). This middleware rejects
oversized **bodies** before Pydantic parses them — useful when an attacker
sends a 100 MB blob to exhaust memory. Returns 413.

Per-request timeout lives in `routes.py` (wraps the detector call so the
event loop can cancel it). Default 30s, override via `REQUEST_TIMEOUT_SECONDS`.
"""

from __future__ import annotations

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


DEFAULT_MAX_BODY_BYTES = 200_000  # ~100 KB text + JSON overhead
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_TEXT_MAX_LENGTH = 100_000


def max_body_bytes() -> int:
    return int(os.environ.get("MAX_BODY_BYTES", DEFAULT_MAX_BODY_BYTES))


def request_timeout_seconds() -> float:
    return float(os.environ.get("REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS))


def text_max_length() -> int:
    return int(os.environ.get("TEXT_MAX_LENGTH", DEFAULT_TEXT_MAX_LENGTH))


class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    """Reject Content-Length > limit with 413. Bodies without Content-Length
    (chunked transfer) fall through to FastAPI's stream handling."""

    def __init__(self, app, *, max_bytes: int) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next):
        cl = request.headers.get("content-length")
        if cl is not None:
            try:
                size = int(cl)
            except ValueError:
                return JSONResponse({"detail": "invalid content-length"}, status_code=400)
            if size > self.max_bytes:
                return JSONResponse(
                    {"detail": f"request body exceeds {self.max_bytes} bytes"},
                    status_code=413,
                )
        return await call_next(request)
