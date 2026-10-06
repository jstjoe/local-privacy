"""A disk cache in front of any `LLMClient`, so reruns cost nothing and give
the same numbers.

    from opf_eval.llm import make_client
    client = make_client("anthropic", cache_dir=ws.root / "llm_cache")
    client.complete_json(system=..., user=..., schema=...)   # calls the API
    client.complete_json(system=..., user=..., schema=...)   # served from disk
    client.hits, client.misses                              # (1, 1)

The key is the sha256 of the provider, model, system prompt, user prompt,
schema and schema name. Prompt versions live inside the prompts themselves
(silver and review prompts embed their PROMPT_VERSION), so a prompt change is
a new key. A backend can add settings that change its answers (Anthropic's
effort level, an OpenAI-compatible server's URL) through `cache_settings()`.

One JSON file per key, laid out as

    <cache_dir>/<provider>/<model>/<key[:2]>/<key>.json

with {"request": {...}, "response": {...}, "created_at": ...}. The request
part keeps only metadata and hashes and never the prompt text. The response
still quotes values found in the text (a silver label is the PII value
itself), so treat a cache directory as being as sensitive as the data.
Errors are never cached: a refusal or a transport failure is retried on the
next run. Lone surrogates in an answer become U+FFFD before it is written
because a JSON file in UTF-8 cannot hold them. When a server-side fallback
model produced the answer, the entry names it under `fallback_model` and a
cache hit reports it through `pop_fallbacks()` just like a fresh call.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .base import FallbackLog, LLMClient, pop_fallbacks, scrub_surrogates

CACHE_FORMAT = 1


def _sha256(text: str) -> str:
    # surrogatepass: a lone surrogate in the text must not stop the key.
    return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()


def cache_key(
    *,
    provider: str,
    model: str,
    system: str,
    user: str,
    schema: dict,
    name: str,
    settings: dict | None = None,
) -> str:
    """The sha256 cache key for one request.

    `settings` is left out of the hashed payload when empty, so a backend
    without extra settings gets exactly the key of the plain request.
    """
    payload: dict[str, Any] = {
        "provider": provider,
        "model": model,
        "system": system,
        "user": user,
        "schema": schema,
        "name": name,
    }
    if settings:
        payload["settings"] = settings
    return _sha256(json.dumps(payload, sort_keys=True))


def _safe_dir_name(value: str) -> str:
    # Model ids contain "/" (openai/gpt-oss-120b-maas) and ":" (Ollama tags).
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "_"


class CachedClient:
    """Wraps an `LLMClient`; same interface plus `.hits`, `.misses`, `.inner`.

    Attributes not defined here (provider, model, remote, operator, region,
    ...) are read from the wrapped client, so `describe()` and remote checks
    see the real backend. Safe to share across threads.
    """

    def __init__(self, inner: LLMClient, cache_dir: str | Path):
        self.inner = inner
        self.cache_dir = Path(cache_dir)
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._fallbacks = FallbackLog()

    # The protocol's attributes are real properties because `isinstance(x,
    # LLMClient)` looks them up statically and does not call __getattr__.
    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def remote(self) -> bool:
        return self.inner.remote

    @property
    def operator(self) -> str:
        return self.inner.operator

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes missing on the wrapper itself.
        if name == "inner":  # guards against recursion before __init__ ran
            raise AttributeError(name)
        return getattr(self.inner, name)

    def __repr__(self) -> str:
        return f"CachedClient({self.inner!r}, cache_dir={str(self.cache_dir)!r})"

    def _settings(self) -> dict:
        fn = getattr(self.inner, "cache_settings", None)
        return dict(fn()) if callable(fn) else {}

    def key_for(self, *, system: str, user: str, schema: dict, name: str = "result") -> str:
        return cache_key(
            provider=self.inner.provider,
            model=self.inner.model,
            system=system,
            user=user,
            schema=schema,
            name=name,
            settings=self._settings(),
        )

    def path_for(self, key: str) -> Path:
        return (
            self.cache_dir
            / _safe_dir_name(self.inner.provider)
            / _safe_dir_name(self.inner.model)
            / key[:2]
            / f"{key}.json"
        )

    def _read(self, path: Path) -> dict | None:
        """The cached entry when it holds a usable response, else None."""
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Missing, half-written or not UTF-8 (ValueError covers both
            # JSONDecodeError and UnicodeDecodeError): a miss, then overwritten.
            return None
        response = entry.get("response") if isinstance(entry, dict) else None
        return entry if isinstance(response, dict) else None

    def pop_fallbacks(self) -> list[str]:
        """Fallback models behind the answers this thread got since the last
        call, from fresh calls and cache hits alike."""
        return self._fallbacks.pop()

    def _write(self, path: Path, entry: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write then rename so a crash or a parallel reader never sees half a file.
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=1)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        name: str = "result",
        max_tokens: int = 8000,
    ) -> dict:
        key = self.key_for(system=system, user=user, schema=schema, name=name)
        path = self.path_for(key)
        cached = self._read(path)
        if cached is not None:
            with self._lock:
                self.hits += 1
            self._fallbacks.note(cached.get("fallback_model"))
            return cached["response"]
        with self._lock:
            self.misses += 1
        pop_fallbacks(self.inner)  # forget calls made around the cache
        # An exception propagates before anything is written: errors are not cached.
        response = scrub_surrogates(self.inner.complete_json(
            system=system, user=user, schema=schema, name=name, max_tokens=max_tokens
        ))
        fallback = (pop_fallbacks(self.inner) or [None])[-1]
        self._fallbacks.note(fallback)
        entry = {
            "format": CACHE_FORMAT,
            "request": {
                "provider": self.inner.provider,
                "model": self.inner.model,
                "name": name,
                "settings": self._settings(),
                "system_sha256": _sha256(system),
                "user_sha256": _sha256(user),
                "user_chars": len(user),
                "schema_sha256": _sha256(json.dumps(schema, sort_keys=True)),
            },
            "response": response,
            **({"fallback_model": fallback} if fallback else {}),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._write(path, entry)
        return response

    def stats(self) -> dict:
        """{"hits", "misses"} for notebook progress lines."""
        with self._lock:
            return {"hits": self.hits, "misses": self.misses}
