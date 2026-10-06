"""LLM clients for silver labels and LLM review (plan 13, Part 3).

One small protocol, `LLMClient.complete_json(system=, user=, schema=)`, in
front of every backend. Each backend says who sees the text:

| provider            | operator                          | remote | default model              |
| ------------------- | --------------------------------- | ------ | -------------------------- |
| `anthropic`         | Anthropic                         | yes    | claude-opus-5-5            |
| `anthropic_vertex`  | your GCP project (Vertex AI)      | yes    | claude-opus-5-5            |
| `openai`            | OpenAI                            | yes    | gpt-6.1-sol                |
| `openai_vertex`     | your GCP project (Vertex AI)      | yes    | openai/gpt-oss-120b-maas   |
| `openai_compatible` | the server at `base_url`          | unless localhost | (required)       |
| `stub`              | local stub                        | no     | stub                       |

Usage:

    from opf_eval import llm
    client = llm.make_client("anthropic", cache_dir=ws.root / "llm_cache")
    print(llm.describe(client))
    # claude-opus-5-5 via Anthropic (hosted: Anthropic sees the text)
    answer = client.complete_json(system="...", user=text, schema=schema)

Hosted providers are gated by `PII_BENCH_ALLOW_REMOTE` (see `remote_allowed`).
`make_client` refuses a hosted provider when it is off, before any
credentials are read. Every hosted request checks the switch again, so a
client built while it was on sends nothing once it is turned off. Amazon Bedrock backends are parked until an AWS account
exists (plan 13, PR 5); the protocol and the cache do not change when they
arrive.

The SDKs are imported only when a client is built, so this package imports
without the `llm` extra installed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import (
    ALLOW_REMOTE_ENV,
    DEFAULT_ALLOW_REMOTE,
    DEFAULT_MODELS,
    OPENAI_MODELS,
    PARKED_PROVIDERS,
    PROVIDERS,
    LLMClient,
    LLMError,
    ProviderInfo,
    compatible_info,
    describe,
    ensure_remote_allowed,
    is_local_url,
    parse_answer,
    parse_json,
    pop_fallbacks,
    remote_allowed,
    remote_refused,
    scrub_surrogates,
)
from .cache import CachedClient, cache_key
from .schema import validate
from .stub import StubClient

__all__ = [
    "ALLOW_REMOTE_ENV",
    "DEFAULT_ALLOW_REMOTE",
    "DEFAULT_MODELS",
    "OPENAI_MODELS",
    "PARKED_PROVIDERS",
    "PROVIDERS",
    "CachedClient",
    "LLMClient",
    "LLMError",
    "ProviderInfo",
    "StubClient",
    "cache_key",
    "describe",
    "ensure_remote_allowed",
    "is_local_url",
    "make_client",
    "parse_answer",
    "parse_json",
    "pop_fallbacks",
    "provider_info",
    "remote_allowed",
    "scrub_surrogates",
    "validate",
]


def provider_info(provider: str, *, base_url: str | None = None) -> ProviderInfo:
    """Who operates `provider` and whether it is remote, without building a client."""
    name = provider.strip().lower()
    if name in PARKED_PROVIDERS:
        raise ValueError(
            f"{name} ({PARKED_PROVIDERS[name]}) is parked until an AWS account exists "
            "(plan 13, PR 5). Use anthropic_vertex or openai_vertex to keep data in your own cloud."
        )
    if name not in PROVIDERS:
        raise ValueError(f"unknown LLM provider {provider!r}; choose one of {sorted(PROVIDERS)}")
    if name == "openai_compatible":
        if not base_url:
            raise ValueError("openai_compatible needs base_url, e.g. http://localhost:1234/v1")
        return compatible_info(base_url)
    return PROVIDERS[name]


def make_client(
    provider: str,
    model: str | None = None,
    *,
    cache_dir: str | Path | None = None,
    allow_remote: bool | None = None,
    **kwargs: Any,
) -> LLMClient:
    """Build the client for `provider`, optionally behind a disk cache.

    `model` defaults to `DEFAULT_MODELS[provider]`. Backend keyword arguments:

    1. anthropic takes `effort` ("low", "medium" (default), "high", "xhigh",
       "max") and `fallbacks` (server-side refusal fallbacks, on when the SDK
       supports them).
    2. anthropic_vertex takes `project_id` (default env GOOGLE_CLOUD_PROJECT),
       `region` (default env CLOUD_ML_REGION, else "us") and `effort`.
    3. openai_vertex takes `project_id` (default env GOOGLE_CLOUD_PROJECT,
       else the ADC project) and `region` (default "us-central1").
    4. openai_compatible takes `base_url` (required) and `api_key` (default
       env OPENAI_COMPATIBLE_API_KEY, else "not-needed").
    5. Every OpenAI backend takes `reasoning_effort` and `mode`.
    6. stub takes `responder(system, user, schema) -> dict`.
    7. Every backend takes `max_retries`, `timeout` (seconds) and
       `sdk_client` (tests).

    `allow_remote=None` reads `PII_BENCH_ALLOW_REMOTE` (default allow, see the
    plan's data-exposure policy). A hosted provider with remote calls
    forbidden raises `LLMError` naming the operator that would see the data.
    The hosted clients keep `allow_remote` and check it again before every
    request, so turning the switch off later also stops a client built earlier.
    """
    name = provider.strip().lower()
    info = provider_info(name, base_url=kwargs.get("base_url"))
    if info.remote and not remote_allowed(allow_remote):
        raise remote_refused(name, info.operator)

    client: Any
    if name == "stub":
        client = StubClient(
            kwargs.pop("responder", None),
            model=model or "stub",
            **{k: kwargs.pop(k) for k in ("provider", "remote", "operator") if k in kwargs},
        )
        if kwargs:
            raise TypeError(f"unexpected arguments for the stub client: {sorted(kwargs)}")
    elif name in ("anthropic", "anthropic_vertex"):
        from .anthropic import AnthropicClient

        client = AnthropicClient(name, model, allow_remote=allow_remote, **kwargs)
    else:
        from .openai import OpenAIClient

        client = OpenAIClient(name, model, allow_remote=allow_remote, **kwargs)

    if cache_dir is not None:
        client = CachedClient(client, cache_dir)
    return client
