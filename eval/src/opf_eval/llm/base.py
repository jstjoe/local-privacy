"""Core types shared by every LLM backend: the client protocol, the error type,
the provider registry and the remote-provider guard.

Kept separate from `__init__` so the backend modules can import these names
without a circular import. Everything here is re-exported from `opf_eval.llm`.
"""

from __future__ import annotations

import json
import os
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit


class LLMError(RuntimeError):
    """An LLM call that produced no usable answer.

    Raised for refusals, truncated answers, answers that still fail schema
    validation after one retry, missing credentials, and transport errors that
    survived the SDK's own retries. Callers record it as an error row instead of
    treating it as an empty answer.

    Two attributes say why the call failed so that callers never have to match
    the message text. `truncated` is True when the model ran out of output
    tokens (`max_tokens`) before it finished its answer. `refusal` holds the
    refusal category or the model's refusal text when the model declined to
    answer, and is None otherwise. Both clients set them, and the message
    stays readable for people.
    """

    def __init__(self, *args: object, truncated: bool = False, refusal: str | None = None) -> None:
        super().__init__(*args)
        self.truncated = bool(truncated)
        self.refusal = refusal

    def __reduce__(self):
        # Keep the attributes when the error crosses a process boundary.
        return (_rebuild_llm_error, (self.args, self.truncated, self.refusal))


def _rebuild_llm_error(args: tuple, truncated: bool, refusal: str | None) -> LLMError:
    return LLMError(*args, truncated=truncated, refusal=refusal)


@dataclass(frozen=True)
class ProviderInfo:
    """Static facts about a backend. `operator` names who sees the text."""

    name: str
    operator: str
    remote: bool
    default_model: str | None


DEFAULT_MODELS: dict[str, str | None] = {
    "anthropic": "claude-opus-5-5",
    "anthropic_vertex": "claude-opus-5-5",
    "openai": "gpt-6.1-sol",
    "openai_vertex": "openai/gpt-oss-120b-maas",
    "openai_compatible": None,
}

# Documented OpenAI choices for the notebook's dropdown. Any model id is accepted.
OPENAI_MODELS = ("gpt-6.1-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-luna")

PROVIDERS: dict[str, ProviderInfo] = {
    "anthropic": ProviderInfo("anthropic", "Anthropic", True, DEFAULT_MODELS["anthropic"]),
    "anthropic_vertex": ProviderInfo(
        "anthropic_vertex",
        "your GCP project (Vertex AI)",
        True,
        DEFAULT_MODELS["anthropic_vertex"],
    ),
    "openai": ProviderInfo("openai", "OpenAI", True, DEFAULT_MODELS["openai"]),
    "openai_vertex": ProviderInfo(
        "openai_vertex",
        "your GCP project (Vertex AI)",
        True,
        DEFAULT_MODELS["openai_vertex"],
    ),
    # The real operator and locality depend on base_url; see `compatible_info`.
    "openai_compatible": ProviderInfo(
        "openai_compatible", "the server at <base_url>", True, None
    ),
    "stub": ProviderInfo("stub", "local stub", False, "stub"),
}

# Backends from the plan that wait for an AWS account (plan 13, PR 5).
PARKED_PROVIDERS = {
    "anthropic_bedrock": "Claude on Amazon Bedrock",
    "openai_bedrock": "OpenAI models on Amazon Bedrock",
}

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def is_local_url(base_url: str) -> bool:
    """True when `base_url` points at this machine, so the text never leaves it."""
    host = (urlsplit(base_url if "//" in base_url else f"//{base_url}").hostname or "").lower()
    return host in _LOCAL_HOSTS or host.endswith(".localhost")


def compatible_info(base_url: str) -> ProviderInfo:
    """ProviderInfo for an OpenAI-compatible server at `base_url`."""
    return ProviderInfo(
        "openai_compatible",
        f"the server at {base_url}",
        not is_local_url(base_url),
        None,
    )


# ----------------------------------------------------------- remote guard

ALLOW_REMOTE_ENV = "PII_BENCH_ALLOW_REMOTE"
# Plan 13's data-exposure policy allows hosted providers during internal
# development. Flip this to False before the notebook is shared more widely.
DEFAULT_ALLOW_REMOTE = True
_TRUE = {"1", "true", "yes", "on"}


def remote_allowed(flag: bool | None = None) -> bool:
    """Whether hosted providers may see the data.

    An explicit `flag` wins. Otherwise the env var `PII_BENCH_ALLOW_REMOTE`
    decides ("1"/"true"/"yes"/"on" allow, anything else forbids) and an unset
    variable falls back to `DEFAULT_ALLOW_REMOTE`.
    """
    if flag is not None:
        return bool(flag)
    value = os.environ.get(ALLOW_REMOTE_ENV)
    if value is None or not value.strip():
        return DEFAULT_ALLOW_REMOTE
    return value.strip().lower() in _TRUE


def remote_refused(provider: str, operator: str) -> LLMError:
    """The error raised when a hosted provider is used while the switch is off."""
    return LLMError(
        f"{provider} sends the text to {operator}, and hosted providers are switched off "
        f"(allow_remote=False or {ALLOW_REMOTE_ENV}=0). Use a local provider such as "
        "openai_compatible with a localhost base_url, or allow remote providers."
    )


def ensure_remote_allowed(client: Any, flag: bool | None = None) -> None:
    """Raise `LLMError` when `client` is hosted and hosted providers are off.

    Backends call this at the start of every request and not only when the
    client is built. A client built while the switch was on therefore stops
    sending text as soon as `PII_BENCH_ALLOW_REMOTE` is turned off. An explicit
    `flag` (the `allow_remote` given to `make_client`) still wins over the
    environment variable.
    """
    if getattr(client, "remote", False) and not remote_allowed(flag):
        raise remote_refused(client.provider, client.operator)


# --------------------------------------------------------------- protocol


@runtime_checkable
class LLMClient(Protocol):
    """What labelers and reviewers need from a model.

    `complete_json` returns a dict that validates against `schema` or raises
    `LLMError`. It never returns a partial or empty answer in place of an error.
    """

    provider: str
    model: str
    remote: bool
    operator: str

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        name: str = "result",
        max_tokens: int = 8000,
    ) -> dict: ...


def describe(client: LLMClient) -> str:
    """One line naming the model and who sees the text, for notebook output.

    >>> describe(make_client("anthropic"))  # doctest: +SKIP
    'claude-opus-5-5 via Anthropic (hosted: Anthropic sees the text)'
    """
    where = f"{client.model} via {client.operator}"
    region = getattr(client, "region", None)
    if region:
        where += f", region {region}"
    if client.remote:
        return f"{where} (hosted: {client.operator} sees the text)"
    return f"{where} (local: the text stays on this machine)"


def retry_note(errors: list[str], *, limit: int = 20) -> str:
    """The message appended to the user turn when an answer fails validation."""
    shown = errors[:limit]
    more = len(errors) - len(shown)
    lines = "\n".join(f"- {e}" for e in shown)
    if more > 0:
        lines += f"\n- ... and {more} more"
    return (
        "\n\nYour previous answer did not match the required JSON schema:\n"
        f"{lines}\n"
        "Answer again with a single JSON value that matches the schema exactly."
    )


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def json_candidates(text: str) -> Iterator[Any]:
    """Every JSON value a model's answer might mean, most likely first.

    Yields the whole text when it parses, then each fenced code block that
    parses, then each complete JSON object or array that starts somewhere in
    the text. Prompt-only answers often add prose such as "I found [2]
    entities:" before the real answer, so callers should take the first
    candidate that fits their schema rather than the first one found.
    """
    text = (text or "").strip()
    if not text:
        return
    try:
        yield json.loads(text)
        return
    except json.JSONDecodeError:
        pass
    for block in _FENCE.findall(text):
        try:
            yield json.loads(block.strip())
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text, i)
            except json.JSONDecodeError:
                continue
            yield value


def parse_json(text: str):
    """Parse a model's JSON answer, tolerating code fences and stray prose.

    Structured-output modes return bare JSON. Prompt-only modes (local servers
    without JSON support) often wrap it in ```json fences or add a sentence
    first, so this falls back to the first fenced block and then to the first
    complete JSON object or array in the text. Raises ValueError when nothing
    parses. `parse_answer` picks the first candidate that fits a schema.
    """
    if not (text or "").strip():
        raise ValueError("the answer was empty")
    for value in json_candidates(text):
        return value
    raise ValueError(f"the answer is not JSON: {text.strip()[:80]!r}")


_LONE_SURROGATE = re.compile("[\ud800-\udfff]")


def scrub_surrogates(value: Any) -> Any:
    """`value` with every lone UTF-16 surrogate replaced by U+FFFD.

    A model that copies half of an emoji pair answers with an escaped lone
    surrogate such as `"\\ud83d"`, and text read from a JSON file can hold
    one too. Neither can be encoded as UTF-8, so the request body, the cache
    file and every JSONL writer would fail on it. The replacement keeps the
    string's length, so offsets into the text stay right. Strings inside
    lists and dicts are scrubbed too.
    """
    if isinstance(value, str):
        return _LONE_SURROGATE.sub("\ufffd", value)
    if isinstance(value, list):
        return [scrub_surrogates(v) for v in value]
    if isinstance(value, dict):
        return {scrub_surrogates(k): scrub_surrogates(v) for k, v in value.items()}
    return value


class FallbackLog:
    """Which answers on this thread came from a server-side fallback model.

    A client that can be answered by another model than the one it asked
    for calls `note(model)` after every answer, with None when the requested
    model answered. Callers call `pop_fallbacks(client)` around one record's
    calls to learn whether any of them was answered by a fallback. Each
    thread keeps its own list, so concurrent records do not mix.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    def note(self, model: str | None) -> None:
        if model:
            self._list().append(model)

    def pop(self) -> list[str]:
        out = self._list()[:]
        self._local.models = []
        return out

    def _list(self) -> list[str]:
        if not hasattr(self._local, "models"):
            self._local.models = []
        return self._local.models


def pop_fallbacks(client: Any) -> list[str]:
    """The fallback models that answered `client`'s calls on this thread
    since the last pop, oldest first. Empty for clients without fallbacks."""
    fn = getattr(client, "pop_fallbacks", None)
    return list(fn()) if callable(fn) else []


def parse_answer(text: str, schema: dict) -> tuple[Any, list[str]]:
    """Parse a model's answer and validate it against `schema`.

    Returns `(answer, [])` for the first JSON candidate in `text` that
    validates. When none does it returns the first candidate with its
    validation errors, or `(None, [reason])` when nothing parses, so the
    caller can show the errors to the model on its one retry. Lone
    surrogates in the answer become U+FFFD (see `scrub_surrogates`).
    """
    from .schema import validate

    first: tuple[Any, list[str]] | None = None
    for value in json_candidates(text):
        value = scrub_surrogates(value)
        errors = validate(value, schema)
        if not errors:
            return value, []
        if first is None:
            first = (value, errors)
    if first is not None:
        return first
    stripped = (text or "").strip()
    reason = f"the answer is not JSON: {stripped[:80]!r}" if stripped else "the answer was empty"
    return None, [f"$: {reason}"]


def google_auth_errors() -> tuple[type[BaseException], ...]:
    """google-auth's base exception, or nothing when google-auth is not installed.

    Both Vertex AI backends fetch and refresh Google access tokens inside the
    SDK call, so a revoked login or an unreachable token endpoint surfaces as a
    google-auth exception that is not an SDK error. The clients catch these and
    raise `LLMError` instead.
    """
    try:
        import google.auth.exceptions
    except ImportError:
        return ()
    return (google.auth.exceptions.GoogleAuthError,)


GCLOUD_HINT = "run `gcloud auth application-default login` (Colab: nb.gcloud_auth())"
