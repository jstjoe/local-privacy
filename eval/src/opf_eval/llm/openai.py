"""OpenAI models through the OpenAI SDK: the OpenAI API, gpt-oss on Vertex AI,
and any OpenAI-compatible server (LM Studio, vLLM, Ollama).

    from opf_eval.llm import make_client
    gpt = make_client("openai")                                   # OPENAI_API_KEY
    oss = make_client("openai_vertex")                            # GOOGLE_CLOUD_PROJECT + ADC
    local = make_client("openai_compatible", "qwen3:8b", base_url="http://localhost:11434/v1")

All three use Chat Completions. The OpenAI API gets a JSON-schema
`response_format`, in strict mode when the schema allows it (every property
required and `additionalProperties: false` on every object). Vertex AI and
local servers do not all support JSON-schema output, so for them the client
steps down once per client on a 400 that mentions the response format:

    json_schema  ->  json_object + the schema pasted into the system prompt  ->  prompt only

Every answer is validated against the schema whatever the mode. A failed
validation retries once with the validator's errors and then raises
`LLMError`, as do refusals, truncated answers, transport errors and any
other failure inside the SDK call. A refusal sets `.refusal` on the error and
a `length` stop sets `.truncated`. Lone surrogates in the prompts are replaced
with U+FFFD before sending because the request body must be valid UTF-8.

Vertex AI authenticates with a short-lived Google access token. The client
passes the SDK a callable that refreshes the token from application default
credentials whenever it has expired, so long runs do not fail after an hour.
"""

from __future__ import annotations

import json
import os
import re
import threading
from typing import Any

from .base import (
    DEFAULT_MODELS,
    GCLOUD_HINT,
    PROVIDERS,
    LLMError,
    compatible_info,
    ensure_remote_allowed,
    google_auth_errors,
    parse_answer,
    retry_note,
    scrub_surrogates,
)
from .schema import strict_problems

INSTALL_HINT = "pip install 'opf-eval[llm]'"
CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
VERTEX_DEFAULT_REGION = "us-central1"
# Environment variable for an OpenAI-compatible server's key. OPENAI_API_KEY
# is deliberately not used so a real OpenAI key is never sent to another host.
COMPATIBLE_KEY_ENV = "OPENAI_COMPATIBLE_API_KEY"

MODES = ("json_schema", "json_object", "prompt")


def _sdk():
    try:
        import openai
    except ImportError as exc:  # pragma: no cover - the test env has the SDK
        raise ImportError(f"The OpenAI SDK is not installed. Run: {INSTALL_HINT}") from exc
    return openai


def vertex_base_url(project_id: str, region: str) -> str:
    """The Vertex AI OpenAI-compatible endpoint for a project and location."""
    if region == "global":
        host = "aiplatform.googleapis.com"
    elif region in ("us", "eu"):
        host = f"aiplatform.{region}.rep.googleapis.com"
    else:
        host = f"{region}-aiplatform.googleapis.com"
    return f"https://{host}/v1/projects/{project_id}/locations/{region}/endpoints/openapi"


class GoogleTokenProvider:
    """A callable that returns a fresh Google access token for Vertex AI.

    Wraps `google.auth.default()` credentials and refreshes them when they are
    missing a token or have expired. Thread-safe so a thread pool shares one
    refresh.
    """

    def __init__(self, credentials: Any = None):
        try:
            import google.auth
            import google.auth.exceptions
            import google.auth.transport.requests
        except ImportError as exc:
            raise ImportError(f"google-auth is not installed. Run: {INSTALL_HINT}") from exc
        self.project_id: str | None = None
        if credentials is None:
            try:
                credentials, self.project_id = google.auth.default(scopes=[CLOUD_PLATFORM_SCOPE])
            except google.auth.exceptions.DefaultCredentialsError as exc:
                raise LLMError(f"no Google application default credentials; {GCLOUD_HINT}") from exc
        self.credentials = credentials
        self._request = google.auth.transport.requests.Request
        self._lock = threading.Lock()

    def __call__(self) -> str:
        with self._lock:
            if not getattr(self.credentials, "valid", False) or not self.credentials.token:
                self.credentials.refresh(self._request())
            return self.credentials.token


def _schema_name(name: str) -> str:
    # OpenAI accepts only [A-Za-z0-9_-] up to 64 characters here.
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name)[:64]
    return cleaned or "result"


def _is_response_format_error(exc: Exception) -> bool:
    text = str(exc).lower()
    body = getattr(exc, "body", None)
    if body is not None:
        text += " " + json.dumps(body, default=str).lower()
    return any(
        word in text
        for word in ("response_format", "json_schema", "json_object", "response format")
    )


class OpenAIClient:
    """`LLMClient` for providers "openai", "openai_vertex" and "openai_compatible".

    `sdk_client` replaces the SDK client object (tests pass a fake).
    `reasoning_effort` is passed through when set, for reasoning models that
    take it. `mode` forces a response-format mode from `MODES` instead of
    stepping down on errors.
    """

    def __init__(
        self,
        provider: str = "openai",
        model: str | None = None,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        project_id: str | None = None,
        region: str | None = None,
        reasoning_effort: str | None = None,
        mode: str | None = None,
        max_retries: int = 2,
        timeout: float | None = None,
        sdk_client: Any = None,
        token_provider: Any = None,
        allow_remote: bool | None = None,
    ):
        if provider not in ("openai", "openai_vertex", "openai_compatible"):
            raise ValueError(f"OpenAIClient does not serve provider {provider!r}")
        if mode is not None and mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.provider = provider
        self.reasoning_effort = reasoning_effort
        self.region: str | None = None
        self.project_id: str | None = None
        self._api_key: Any = api_key
        if provider == "openai_compatible":
            if not base_url:
                raise ValueError("openai_compatible needs base_url, e.g. http://localhost:1234/v1")
            if not model:
                raise ValueError("openai_compatible needs a model id as the server names it")
            info = compatible_info(base_url)
            self.base_url: str | None = base_url
            self._api_key = api_key or os.environ.get(COMPATIBLE_KEY_ENV) or "not-needed"
        elif provider == "openai_vertex":
            info = PROVIDERS[provider]
            self.region = region or VERTEX_DEFAULT_REGION
            if token_provider is None and sdk_client is None:
                token_provider = GoogleTokenProvider()
            self.project_id = (
                project_id
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
                or getattr(token_provider, "project_id", None)
            )
            if not self.project_id and not base_url:
                raise LLMError(
                    "openai_vertex needs a GCP project: "
                    "pass project_id= or set GOOGLE_CLOUD_PROJECT"
                )
            self.base_url = base_url or vertex_base_url(self.project_id, self.region)
            self._api_key = token_provider
        else:
            if base_url:
                # Otherwise the OpenAI key would go to another host labelled as OpenAI.
                raise ValueError(
                    "provider 'openai' talks only to OpenAI; use openai_compatible for base_url"
                )
            info = PROVIDERS[provider]
            self.base_url = None
        self.model = model or DEFAULT_MODELS[provider]
        self.remote = info.remote
        self.operator = info.operator
        # Checked again before every request (see `ensure_remote_allowed`).
        self.allow_remote = allow_remote
        # The OpenAI API supports JSON-schema output; others may not.
        self._mode = mode or "json_schema"
        self._mode_fixed = mode is not None or provider == "openai"
        self._mode_lock = threading.Lock()
        self._sdk_client = (
            sdk_client if sdk_client is not None else self._build(max_retries, timeout)
        )

    def __repr__(self) -> str:
        return (
            f"OpenAIClient(provider={self.provider!r}, model={self.model!r}, "
            f"mode={self._mode!r})"
        )

    @property
    def mode(self) -> str:
        """The response-format mode in use (it can step down after errors)."""
        return self._mode

    def _build(self, max_retries: int, timeout: float | None) -> Any:
        openai = _sdk()
        options: dict[str, Any] = {"max_retries": max_retries}
        if timeout is not None:
            options["timeout"] = timeout
        if self.base_url:
            options["base_url"] = self.base_url
        if self._api_key is not None:
            options["api_key"] = self._api_key
        try:
            return openai.OpenAI(**options)
        except openai.OpenAIError as exc:
            raise LLMError(
                f"cannot create the {self.provider} client ({exc}); set OPENAI_API_KEY"
            ) from exc

    def cache_settings(self) -> dict:
        """Settings that change answers and therefore belong in the cache key."""
        settings: dict[str, Any] = {}
        if self.provider == "openai_compatible":
            settings["base_url"] = self.base_url
        if self.reasoning_effort:
            settings["reasoning_effort"] = self.reasoning_effort
        return settings

    # ------------------------------------------------------------- calls

    def _request(
        self, *, system: str, user: str, schema: dict, name: str, max_tokens: int, mode: str
    ) -> dict:
        if mode == "json_schema":
            strict = not strict_problems(schema)
            response_format: dict | None = {
                "type": "json_schema",
                "json_schema": {"name": _schema_name(name), "schema": schema, "strict": strict},
            }
        else:
            # Without server-side enforcement the model only sees the schema in the prompt.
            system = (
                (system + "\n\n" if system else "")
                + "Reply with one JSON object and nothing else. It must match this JSON Schema:\n"
                + json.dumps(schema, ensure_ascii=False)
            )
            response_format = {"type": "json_object"} if mode == "json_object" else None
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if response_format is not None:
            kwargs["response_format"] = response_format
        # The OpenAI API's reasoning models take max_completion_tokens; other
        # servers widely support only max_tokens.
        kwargs["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = max_tokens
        if self.reasoning_effort:
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

    def _create(self, **request: Any) -> Any:
        openai = _sdk()
        while True:
            mode = self._mode
            kwargs = self._request(mode=mode, **request)
            try:
                return self._sdk_client.chat.completions.create(**kwargs)
            except openai.BadRequestError as exc:
                if self._mode_fixed or not _is_response_format_error(exc) or mode == "prompt":
                    raise LLMError(f"{self.model} via {self.operator}: {exc}") from exc
                with self._mode_lock:
                    if self._mode == mode:  # another thread may have stepped down already
                        self._mode = MODES[MODES.index(mode) + 1]
            except openai.OpenAIError as exc:
                raise LLMError(
                    f"{self.model} via {self.operator}: {type(exc).__name__}: {exc}"
                ) from exc
            except google_auth_errors() as exc:
                # The SDK calls GoogleTokenProvider before each request, so a
                # revoked login or an unreachable token endpoint lands here.
                raise LLMError(
                    f"{self.model} via {self.operator}: Google credentials failed "
                    f"({type(exc).__name__}: {exc}); {GCLOUD_HINT}"
                ) from exc
            except Exception as exc:  # the protocol promises LLMError
                # For example a UnicodeEncodeError from the request body.
                raise LLMError(
                    f"{self.model} via {self.operator}: {type(exc).__name__}: {exc}"
                ) from exc

    def _text(self, response: Any, max_tokens: int) -> str:
        if not getattr(response, "choices", None):
            raise LLMError(f"{self.model} returned no choices")
        choice = response.choices[0]
        message = choice.message
        refusal = getattr(message, "refusal", None)
        if refusal:
            raise LLMError(f"refused: {refusal}", refusal=str(refusal))
        if choice.finish_reason == "content_filter":
            raise LLMError("refused: content_filter", refusal="content_filter")
        if choice.finish_reason == "length":
            raise LLMError(
                f"{self.model} hit max_tokens={max_tokens} before finishing its answer",
                truncated=True,
            )
        return message.content or ""

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
        system, user = scrub_surrogates(system), scrub_surrogates(user)
        prompt = user
        errors: list[str] = []
        for _attempt in range(2):
            response = self._create(
                system=system, user=prompt, schema=schema, name=name, max_tokens=max_tokens
            )
            text = self._text(response, max_tokens)
            answer, errors = parse_answer(text, schema)
            if not errors:
                return answer
            prompt = user + retry_note(errors)
        raise LLMError(
            f"{self.model}: answer still invalid after one retry: {'; '.join(errors[:5])}"
        )
