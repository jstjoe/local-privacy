"""A local stand-in for an LLM, for tests and the notebook's dry run.

    from opf_eval.llm import make_client
    client = make_client("stub", responder=lambda system, user, schema: {"entities": []})
    client.complete_json(system="...", user="...", schema=schema)
    client.calls        # [("...", "...")]

The responder sees exactly what a real model would and returns the parsed
answer. Its answer is validated against the schema like a real model's, so a
test can check that callers survive an `LLMError`. Without a responder the
stub returns the smallest value the schema allows, which is enough to run the
notebook end to end without keys.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from .base import LLMError
from .schema import minimal_instance, validate

Responder = Callable[[str, str, dict], dict]


def _minimal_answer(system: str, user: str, schema: dict) -> dict:
    """The default responder returns the smallest value the schema allows."""
    return minimal_instance(schema)


class StubClient:
    """Implements `LLMClient` by calling `responder(system, user, schema)`.

    A responder may raise `LLMError` to simulate a refusal or an outage. An
    answer that fails the schema raises `LLMError` without a retry because a
    stub answers the same way every time.
    """

    def __init__(
        self,
        responder: Responder | None = None,
        provider: str = "stub",
        model: str = "stub",
        remote: bool = False,
        operator: str = "local stub",
    ):
        self.responder: Responder = responder or _minimal_answer
        self.provider = provider
        self.model = model
        self.remote = remote
        self.operator = operator
        self.calls: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"StubClient(provider={self.provider!r}, model={self.model!r})"

    def complete_json(
        self,
        *,
        system: str,
        user: str,
        schema: dict,
        name: str = "result",
        max_tokens: int = 8000,
    ) -> dict:
        with self._lock:
            self.calls.append((system, user))
        answer = self.responder(system, user, schema)
        errors = validate(answer, schema)
        if errors:
            raise LLMError(
                f"{self.model}: answer does not match the schema: {'; '.join(errors[:5])}"
            )
        return answer
