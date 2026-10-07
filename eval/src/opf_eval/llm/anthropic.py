"""Claude through the Anthropic SDK, on the Anthropic API or on Vertex AI.

    from opf_eval.llm import make_client
    claude = make_client("anthropic")                      # ANTHROPIC_API_KEY or `ant auth login`
    claude = make_client("anthropic_vertex", region="us")  # GOOGLE_CLOUD_PROJECT + ADC
    claude.complete_json(system=..., user=..., schema=...)

Claude always goes through the Anthropic SDK and never through an
OpenAI-compatible shim. Answers use structured outputs
(`output_config.format` with a JSON schema). Every answer is also validated
here because the API ignores the keywords listed in
`schema.ANTHROPIC_UNSUPPORTED`. A failed validation retries once with the
validator's errors appended to the user message and then raises `LLMError`.

Claude Opus 5.5 always thinks adaptively, so no `thinking` or `temperature`
parameter is sent. `effort` (default "medium", the model's own default) is the
only depth control.

A refusal (`stop_reason == "refusal"`) raises `LLMError("refused: <category>")`
with `.refusal` set to the category, so the caller records an error row
instead of an empty answer. A `max_tokens` stop raises `LLMError` with
`.truncated` set to True. On the
first-party API the client opts into server-side fallbacks
(`fallbacks="default"` under the `server-side-fallback-2026-07-01` beta),
which re-runs a declined request on the model Anthropic recommends for that
refusal category inside the same call. Vertex AI does not offer that
parameter, so a refusal there stays an error. The client notes the model of
every answer a fallback produced, and `pop_fallbacks()` hands them to the
caller so silver labels and reviews can say which records another model
answered.

Any other failure inside the SDK call, such as a request body that cannot be
encoded or a client-side limit on `max_tokens`, also becomes `LLMError`.
Lone surrogates in the prompts are replaced with U+FFFD before sending.

On Vertex AI the GCP project comes from `project_id=`, then
GOOGLE_CLOUD_PROJECT or ANTHROPIC_VERTEX_PROJECT_ID, then the application
default credentials. With none of these the constructor raises `LLMError`.
The SDK loads and refreshes the Google token inside each call, so a failed
refresh during a long run also becomes `LLMError` for that record.
"""

from __future__ import annotations

import inspect
import os
from typing import Any

from .base import (
    DEFAULT_MODELS,
    GCLOUD_HINT,
    PROVIDERS,
    FallbackLog,
    LLMError,
    ensure_remote_allowed,
    google_auth_errors,
    parse_answer,
    retry_note,
    scrub_surrogates,
)
from .schema import for_anthropic

EFFORTS = ("low", "medium", "high", "xhigh", "max")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
INSTALL_HINT = "pip install 'opf-eval[llm]'"
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def _sdk():
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - the test env has the SDK
        raise ImportError(f"The Anthropic SDK is not installed. Run: {INSTALL_HINT}") from exc
    return anthropic


def fallback_model(response: Any) -> str | None:
    """The model that answered through a server-side fallback, or None when
    the requested model answered itself.

    A fallback turn holds a `fallback` content block naming the model it
    switched to, and its usage lists a `fallback_message` iteration with
    that model. The last switch wins when there were several.
    """
    found = None
    for block in getattr(response, "content", None) or []:
        if getattr(block, "type", None) == "fallback":
            found = getattr(getattr(block, "to", None), "model", None) or found
    usage = getattr(response, "usage", None)
    for it in getattr(usage, "iterations", None) or []:
        if getattr(it, "type", None) == "fallback_message":
            found = getattr(it, "model", None) or found
    if found is None:
        return None
    return found if isinstance(found, str) else str(found)


def sdk_supports_fallbacks() -> bool:
    """True when the installed SDK's beta Messages API takes `fallbacks` and `betas`.

    Older SDKs reject unknown keyword arguments, so the client only opts into
    server-side fallbacks when the installed version exposes them.
    """
    try:
        from anthropic.resources.beta.messages import Messages
    except ImportError:
        return False
    params = inspect.signature(Messages.create).parameters
    return "fallbacks" in params and "betas" in params


def _sdk_errors() -> tuple[type[BaseException], ...]:
    try:
        import anthropic
    except ImportError:  # pragma: no cover
        return ()
    return (anthropic.AnthropicError,)


def _adc_project() -> str:
    """The GCP project of the application default credentials, for anthropic_vertex.

    Raises `LLMError` when there are no credentials or they name no project.
    """
    try:
        import google.auth
        import google.auth.exceptions
    except ImportError as exc:  # pragma: no cover - the llm extra installs google-auth
        raise ImportError(f"google-auth is not installed. Run: {INSTALL_HINT}") from exc
    try:
        _, project = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
    except google.auth.exceptions.DefaultCredentialsError as exc:
        raise LLMError(f"no Google application default credentials; {GCLOUD_HINT}") from exc
    if not project:
        raise LLMError(
            "anthropic_vertex needs a GCP project: pass project_id= or set GOOGLE_CLOUD_PROJECT"
        )
    return project


class AnthropicClient:
    """`LLMClient` for providers "anthropic" and "anthropic_vertex".

    `sdk_client` replaces the SDK client object (tests pass a fake). Credentials
    are resolved by the SDK: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or an
    `ant auth login` profile for the first-party API, and Google application
    default credentials for Vertex AI.
    """

    def __init__(
        self,
        provider: str = "anthropic",
        model: str | None = None,
        *,
        project_id: str | None = None,
        region: str | None = None,
        effort: str = "medium",
        fallbacks: bool | None = None,
        max_retries: int = 2,
        timeout: float | None = None,
        sdk_client: Any = None,
        allow_remote: bool | None = None,
    ):
        if provider not in ("anthropic", "anthropic_vertex"):
            raise ValueError(f"AnthropicClient does not serve provider {provider!r}")
        if effort not in EFFORTS:
            raise ValueError(f"effort must be one of {EFFORTS}, got {effort!r}")
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.remote = PROVIDERS[provider].remote
        self.operator = PROVIDERS[provider].operator
        # Checked again before every request (see `ensure_remote_allowed`).
        self.allow_remote = allow_remote
        self.effort = effort
        self._fallbacks = FallbackLog()
        self.region: str | None = None
        self.project_id: str | None = None
        if provider == "anthropic_vertex":
            self.project_id = (
                project_id
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
                or os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID")
            )
            if not self.project_id and sdk_client is None:
                # Without a project the SDK fails on every call with a bare
                # RuntimeError, so resolve it from ADC now and fail early.
                self.project_id = _adc_project()
            # "us" is a multi-region: processing stays in the US without tying
            # the notebook to one region's capacity (plan 13, Part 3).
            self.region = region or os.environ.get("CLOUD_ML_REGION") or "us"
            # Server-side fallbacks are a first-party API feature only.
            self.use_fallbacks = False
        else:
            self.use_fallbacks = (
                sdk_supports_fallbacks() if fallbacks is None else bool(fallbacks)
            )
        self._sdk_client = (
            sdk_client if sdk_client is not None else self._build(max_retries, timeout)
        )

    def __repr__(self) -> str:
        return (
            f"AnthropicClient(provider={self.provider!r}, model={self.model!r}, "
            f"effort={self.effort!r})"
        )

    def _build(self, max_retries: int, timeout: float | None) -> Any:
        anthropic = _sdk()
        options: dict[str, Any] = {"max_retries": max_retries}
        if timeout is not None:
            options["timeout"] = timeout
        try:
            if self.provider == "anthropic":
                return anthropic.Anthropic(**options)
            if not hasattr(anthropic, "AnthropicVertex"):  # pragma: no cover
                raise ImportError(
                    "This Anthropic SDK has no Vertex AI client. "
                    "Run: pip install 'anthropic[vertex]'"
                )
            vertex: dict[str, Any] = {"region": self.region, **options}
            if self.project_id:
                vertex["project_id"] = self.project_id
            return anthropic.AnthropicVertex(**vertex)
        except anthropic.AnthropicError as exc:
            hint = (
                "set ANTHROPIC_API_KEY or run `ant auth login`"
                if self.provider == "anthropic"
                else f"{GCLOUD_HINT} and set GOOGLE_CLOUD_PROJECT"
            )
            raise LLMError(f"cannot create the {self.provider} client ({exc}); {hint}") from exc

    def cache_settings(self) -> dict:
        """Settings that change answers and therefore belong in the cache key."""
        return {"effort": self.effort}

    # ------------------------------------------------------------- calls

    def _create(self, *, system: str, user: str, schema: dict, max_tokens: int) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": user}],
            "output_config": {
                "format": {"type": "json_schema", "schema": schema},
                "effort": self.effort,
            },
        }
        if system:
            kwargs["system"] = system
        try:
            if self.use_fallbacks:
                return self._sdk_client.beta.messages.create(
                    betas=[FALLBACK_BETA], fallbacks="default", **kwargs
                )
            return self._sdk_client.messages.create(**kwargs)
        except _sdk_errors() as exc:
            raise LLMError(
                f"{self.model} via {self.operator}: {type(exc).__name__}: {exc}"
            ) from exc
        except google_auth_errors() as exc:
            # AnthropicVertex loads and refreshes ADC inside the call.
            raise LLMError(
                f"{self.model} via {self.operator}: Google credentials failed "
                f"({type(exc).__name__}: {exc}); {GCLOUD_HINT}"
            ) from exc
        except Exception as exc:  # the protocol promises LLMError
            # For example a UnicodeEncodeError from the request body or the
            # SDK's ValueError for a max_tokens that needs streaming.
            raise LLMError(
                f"{self.model} via {self.operator}: {type(exc).__name__}: {exc}"
            ) from exc

    def pop_fallbacks(self) -> list[str]:
        """Models that answered in place of `self.model` through a server-side
        fallback on this thread since the last call, oldest first."""
        return self._fallbacks.pop()

    def _text(self, response: Any, max_tokens: int) -> str:
        stop = getattr(response, "stop_reason", None)
        if stop == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None) or "unspecified"
            raise LLMError(f"refused: {category}", refusal=str(category))
        if stop == "max_tokens":
            raise LLMError(
                f"{self.model} hit max_tokens={max_tokens} before finishing its answer",
                truncated=True,
            )
        # A non-streaming response omits a declined partial, so after a
        # fallback every text block belongs to the answer.
        return "".join(
            block.text
            for block in (response.content or [])
            if getattr(block, "type", None) == "text"
        )

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        name: str = "result",
        max_tokens: int = 8000,
    ) -> dict:
        ensure_remote_allowed(self, self.allow_remote)
        api_schema = for_anthropic(schema)
        system, user = scrub_surrogates(system), scrub_surrogates(user)
        prompt = user
        errors: list[str] = []
        for _attempt in range(2):
            response = self._create(
                system=system, user=prompt, schema=api_schema, max_tokens=max_tokens
            )
            text = self._text(response, max_tokens)
            answer, errors = parse_answer(text, schema)
            if not errors:
                self._fallbacks.note(fallback_model(response))
                return answer
            prompt = user + retry_note(errors)
        raise LLMError(
            f"{self.model}: answer still invalid after one retry: {'; '.join(errors[:5])}"
        )
