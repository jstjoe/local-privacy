"""Tests for `opf_eval.llm`: provider registry and remote guard, the schema
validator, the disk cache, the stub client, and the Anthropic and OpenAI
clients with their SDK objects replaced by fakes.

No test calls a hosted API: every SDK client is a fake, and Google
credentials are monkeypatched.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import anthropic
import httpx2
import openai
import pytest
from opf_eval import llm
from opf_eval.llm import (
    CachedClient,
    LLMClient,
    LLMError,
    StubClient,
    cache_key,
    describe,
    make_client,
    parse_json,
    remote_allowed,
)
from opf_eval.llm.anthropic import FALLBACK_BETA, AnthropicClient
from opf_eval.llm.openai import GoogleTokenProvider, OpenAIClient, vertex_base_url
from opf_eval.llm.schema import (
    for_anthropic,
    is_valid,
    minimal_instance,
    strict_problems,
    validate,
)

# A schema shaped like the silver-label schema (WP-C), strict-mode compatible.
ENTITY_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["entities"],
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["value", "label", "context"],
                "properties": {
                    "value": {"type": "string"},
                    "label": {"type": "string", "enum": ["EMAIL", "PERSON"]},
                    "context": {"type": "string"},
                },
            },
        }
    },
}
GOOD = {"entities": [{"value": "jane@example.com", "label": "EMAIL", "context": "mail jane@example.com now"}]}
BAD = {"entities": [{"value": "jane@example.com", "label": "PHONE", "context": "x"}]}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for key in (
        llm.ALLOW_REMOTE_ENV,
        "GOOGLE_CLOUD_PROJECT",
        "ANTHROPIC_VERTEX_PROJECT_ID",
        "CLOUD_ML_REGION",
        "OPENAI_COMPATIBLE_API_KEY",
    ):
        monkeypatch.delenv(key, raising=False)


# ------------------------------------------------------------------ fakes


class FakeAnthropicSDK:
    """Stands in for `anthropic.Anthropic()`; replays canned responses."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []
        self.messages = SimpleNamespace(create=lambda **kw: self._create("messages", kw))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: self._create("beta", kw)))

    def _create(self, kind, kwargs):
        self.calls.append((kind, kwargs))
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def claude_reply(data=None, *, text=None, stop_reason="end_turn", category=None, blocks=None):
    if text is None and data is not None:
        text = json.dumps(data)
    content = list(blocks or [])
    if text is not None:
        content.append(SimpleNamespace(type="text", text=text))
    details = SimpleNamespace(type="refusal", category=category, explanation=None) if stop_reason == "refusal" else None
    return SimpleNamespace(stop_reason=stop_reason, stop_details=details, content=content)


class FakeOpenAISDK:
    """Stands in for `openai.OpenAI()`; replays canned chat completions."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def gpt_reply(data=None, *, text=None, finish_reason="stop", refusal=None):
    if text is None and data is not None:
        text = json.dumps(data)
    message = SimpleNamespace(content=text, refusal=refusal, role="assistant")
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=message, index=0)])


def openai_400(message: str) -> openai.BadRequestError:
    request = httpx2.Request("POST", "http://localhost:1234/v1/chat/completions")
    response = httpx2.Response(400, request=request)
    return openai.BadRequestError(message, response=response, body={"error": {"message": message}})


# ------------------------------------------------------ registry + guard


def test_provider_registry_and_defaults():
    assert set(llm.PROVIDERS) == {
        "anthropic",
        "anthropic_vertex",
        "openai",
        "openai_vertex",
        "openai_compatible",
        "stub",
    }
    assert llm.DEFAULT_MODELS["anthropic"] == "claude-opus-5-5"
    assert llm.DEFAULT_MODELS["anthropic_vertex"] == "claude-opus-5-5"
    assert llm.DEFAULT_MODELS["openai"] == "gpt-6.1-sol"
    assert llm.DEFAULT_MODELS["openai_vertex"] == "openai/gpt-oss-120b-maas"
    assert llm.DEFAULT_MODELS["openai_compatible"] is None
    assert llm.OPENAI_MODELS == ("gpt-6.1-sol", "gpt-6-luna", "gpt-5.6-sol", "gpt-5.6-luna")
    for name, info in llm.PROVIDERS.items():
        assert info.name == name
        assert info.remote is (name != "stub")
    assert llm.PROVIDERS["anthropic"].operator == "Anthropic"
    assert llm.PROVIDERS["openai_vertex"].operator == "your GCP project (Vertex AI)"


@pytest.mark.parametrize(
    "env, flag, expected",
    [
        (None, None, True),  # internal phase default (plan 13 data-exposure policy)
        ("1", None, True),
        ("true", None, True),
        ("TRUE ", None, True),
        ("0", None, False),
        ("false", None, False),
        ("", None, True),
        ("0", True, True),  # explicit flag wins
        ("1", False, False),
    ],
)
def test_remote_allowed(monkeypatch, env, flag, expected):
    if env is not None:
        monkeypatch.setenv(llm.ALLOW_REMOTE_ENV, env)
    assert remote_allowed(flag) is expected


def test_remote_guard_refuses_before_building_the_sdk_client(monkeypatch):
    def boom(*args, **kwargs):  # the guard must fire before credentials are read
        raise AssertionError("SDK client was constructed")

    monkeypatch.setattr(anthropic, "Anthropic", boom)
    monkeypatch.setattr(openai, "OpenAI", boom)
    with pytest.raises(LLMError, match="Anthropic"):
        make_client("anthropic", allow_remote=False)
    monkeypatch.setenv(llm.ALLOW_REMOTE_ENV, "0")
    with pytest.raises(LLMError, match="OpenAI"):
        make_client("openai")
    with pytest.raises(LLMError, match=r"Vertex AI"):
        make_client("anthropic_vertex")
    with pytest.raises(LLMError, match="the server at https://llm.example.com/v1"):
        make_client("openai_compatible", "m", base_url="https://llm.example.com/v1")


def test_remote_guard_also_stops_clients_built_while_the_switch_was_on(monkeypatch, tmp_path):
    # A notebook keeps clients alive across cells. Turning the switch off must
    # stop them at the next request and not only when a new client is built.
    monkeypatch.setenv(llm.ALLOW_REMOTE_ENV, "1")
    claude_sdk = FakeAnthropicSDK(claude_reply(GOOD), claude_reply(GOOD))
    gpt_sdk = FakeOpenAISDK(gpt_reply(GOOD), gpt_reply(GOOD))
    claude = make_client("anthropic", sdk_client=claude_sdk, cache_dir=tmp_path)
    gpt = make_client("openai", sdk_client=gpt_sdk)
    local = make_client(
        "openai_compatible", "qwen3:8b", base_url="http://localhost:11434/v1", sdk_client=FakeOpenAISDK(gpt_reply(GOOD))
    )
    assert claude.complete_json(system="s", user="first", schema=ENTITY_SCHEMA) == GOOD
    assert gpt.complete_json(system="s", user="first", schema=ENTITY_SCHEMA) == GOOD

    monkeypatch.setenv(llm.ALLOW_REMOTE_ENV, "0")
    with pytest.raises(LLMError, match="Anthropic"):
        claude.complete_json(system="s", user="second", schema=ENTITY_SCHEMA)
    with pytest.raises(LLMError, match="OpenAI"):
        gpt.complete_json(system="s", user="second", schema=ENTITY_SCHEMA)
    assert len(claude_sdk.calls) == 1 and len(gpt_sdk.calls) == 1  # nothing was sent
    # A cached answer sends nothing, so it is still served.
    assert claude.complete_json(system="s", user="first", schema=ENTITY_SCHEMA) == GOOD
    # A local server is not gated.
    assert local.complete_json(system="s", user="second", schema=ENTITY_SCHEMA) == GOOD
    # An explicit allow_remote=True given to make_client still wins over the env var.
    pinned = make_client("anthropic", sdk_client=FakeAnthropicSDK(claude_reply(GOOD)), allow_remote=True)
    assert pinned.complete_json(system="s", user="third", schema=ENTITY_SCHEMA) == GOOD


def test_remote_guard_lets_local_providers_through(monkeypatch):
    monkeypatch.setenv(llm.ALLOW_REMOTE_ENV, "0")
    stub = make_client("stub")
    assert stub.remote is False
    local = make_client(
        "openai_compatible", "qwen3:8b", base_url="http://localhost:11434/v1", sdk_client=FakeOpenAISDK()
    )
    assert local.remote is False
    assert describe(local) == (
        "qwen3:8b via the server at http://localhost:11434/v1 (local: the text stays on this machine)"
    )


@pytest.mark.parametrize(
    "url, local",
    [
        ("http://localhost:1234/v1", True),
        ("http://127.0.0.1:8000/v1", True),
        ("http://[::1]:8000/v1", True),
        ("http://vllm.localhost/v1", True),
        ("localhost:1234/v1", True),
        ("https://api.example.com/v1", False),
        ("http://10.0.0.5:8000/v1", False),  # another machine on the network still sees the text
    ],
)
def test_is_local_url(url, local):
    assert llm.is_local_url(url) is local


def test_parked_and_unknown_providers():
    with pytest.raises(ValueError, match="PR 5"):
        make_client("anthropic_bedrock")
    with pytest.raises(ValueError, match="PR 5"):
        make_client("openai_bedrock")
    with pytest.raises(ValueError, match="unknown LLM provider"):
        make_client("mistral")
    with pytest.raises(ValueError, match="base_url"):
        make_client("openai_compatible", "m")


def test_describe_and_protocol(monkeypatch):
    client = make_client("anthropic", sdk_client=FakeAnthropicSDK())
    assert isinstance(client, LLMClient)
    assert describe(client) == "claude-opus-5-5 via Anthropic (hosted: Anthropic sees the text)"
    vertex = make_client("anthropic_vertex", sdk_client=FakeAnthropicSDK())
    assert describe(vertex) == (
        "claude-opus-5-5 via your GCP project (Vertex AI), region us "
        "(hosted: your GCP project (Vertex AI) sees the text)"
    )
    gpt = make_client("openai", "gpt-6-luna", sdk_client=FakeOpenAISDK())
    assert describe(gpt) == "gpt-6-luna via OpenAI (hosted: OpenAI sees the text)"
    assert isinstance(make_client("stub"), LLMClient)


def test_importing_the_package_does_not_import_the_sdks():
    code = (
        "import sys, opf_eval.llm, opf_eval.llm.anthropic, opf_eval.llm.openai;"
        "print(any(m in sys.modules for m in ('anthropic', 'openai', 'google.auth')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --------------------------------------------------------------- schema


def test_validate_accepts_good_and_reports_paths():
    assert validate(GOOD, ENTITY_SCHEMA) == []
    errors = validate(BAD, ENTITY_SCHEMA)
    assert len(errors) == 1
    assert errors[0].startswith("$.entities[0].label:")
    assert "PHONE" in errors[0]


def test_validate_object_rules():
    errors = validate({"entities": [{"value": "a", "label": "EMAIL", "extra": 1}], "more": True}, ENTITY_SCHEMA)
    assert "$: unexpected property 'more'" in errors
    assert "$.entities[0]: missing required property 'context'" in errors
    assert "$.entities[0]: unexpected property 'extra'" in errors
    open_schema = {"type": "object", "additionalProperties": {"type": "integer"}}
    assert validate({"a": 1}, open_schema) == []
    assert validate({"a": "x"}, open_schema) == ["$.a: expected integer, got \"x\""]
    assert validate({"anything": 1}, {"type": "object"}) == []


@pytest.mark.parametrize(
    "value, kind, ok",
    [
        (1, "integer", True),
        (1.0, "integer", False),  # callers index lists with it, so only an int will do
        (float("nan"), "number", False),
        (float("inf"), "number", False),
        (1.5, "integer", False),
        (True, "integer", False),  # bool is not a number in JSON Schema
        (True, "number", False),
        (2.5, "number", True),
        (None, "null", True),
        ("x", "string", True),
        ([], "array", True),
        ({}, "object", True),
        (False, "boolean", True),
        (0, "boolean", False),
    ],
)
def test_validate_types(value, kind, ok):
    assert is_valid(value, {"type": kind}) is ok


def test_validate_scalar_constraints():
    assert validate(5, {"type": "integer", "minimum": 0, "maximum": 4}) == ["$: 5 is above the maximum 4"]
    assert validate(-1, {"type": "integer", "minimum": 0}) == ["$: -1 is below the minimum 0"]
    assert validate(0, {"type": "number", "exclusiveMinimum": 0}) != []
    assert validate(1, {"type": "number", "exclusiveMaximum": 1}) != []
    assert validate("ab", {"type": "string", "minLength": 3}) != []
    assert validate("abcd", {"type": "string", "maxLength": 3}) != []
    assert validate("B7", {"type": "string", "pattern": r"^[A-Z]+[0-9]+$"}) == []
    assert validate("7B", {"type": "string", "pattern": r"^[A-Z]+[0-9]+$"}) != []
    assert validate("x", {"const": "y"}) != []
    assert validate([1], {"type": "array", "minItems": 2}) != []
    assert validate([1, 2, 3], {"type": "array", "maxItems": 2}) != []


def test_validate_combinators():
    nullable = {"anyOf": [{"type": "string", "enum": ["EMAIL"]}, {"type": "null"}]}
    assert validate(None, nullable) == []
    assert validate("EMAIL", nullable) == []
    assert validate("PHONE", nullable) == ['$: "PHONE" matches none of the allowed shapes']
    assert validate(None, {"type": ["string", "null"]}) == []
    assert validate(3, {"type": ["string", "null"]}) == ["$: expected string or null, got 3"]
    both = {"allOf": [{"type": "integer"}, {"minimum": 2}]}
    assert validate(1, both) == ["$: 1 is below the minimum 2"]
    one = {"oneOf": [{"type": "integer"}, {"type": "number"}]}
    assert validate(1.5, one) == []  # only "number" matches
    assert validate(1, one) != []  # both match


def test_validate_keeps_python_types_apart():
    # An integral float would crash `spans[answer["index"]]` outside the LLMError path.
    index_schema = {"type": "object", "properties": {"index": {"type": "integer"}}}
    assert validate({"index": 2.0}, index_schema) == ["$.index: expected integer, got 2.0"]
    assert validate({"index": 2}, index_schema) == []
    # True == 1 in Python, but not in JSON.
    assert validate(True, {"enum": [1]}) != []
    assert validate(1, {"enum": [True]}) != []
    assert validate(1, {"const": True}) != []
    assert validate([True], {"const": [1]}) != []
    assert validate(1.0, {"enum": [1]}) == []  # numbers compare by value
    assert validate({"a": [1, "x"]}, {"const": {"a": [1, "x"]}}) == []
    # NaN compares False against every bound, so it must fail the type check.
    score = {"type": "number", "minimum": 0, "maximum": 1}
    assert validate(float("nan"), score) != []
    assert validate(float("inf"), {"type": "number"}) != []


def test_validate_rejects_unsupported_schemas():
    with pytest.raises(ValueError, match=r"\$ref"):
        validate({}, {"$ref": "#/$defs/x"})
    with pytest.raises(ValueError, match="unknown JSON Schema type"):
        validate(1, {"type": "int"})


def test_for_anthropic_strips_unsupported_keywords_only():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["pattern", "n", "tags"],
        "properties": {
            # A property *named* like a keyword must survive.
            "pattern": {"type": "string", "pattern": "^a", "maxLength": 3},
            "n": {"anyOf": [{"type": "integer", "minimum": 0, "maximum": 9}, {"type": "null"}]},
            "tags": {"type": "array", "minItems": 2, "maxItems": 5, "items": {"type": "string", "minLength": 1}},
            "one": {"type": "array", "minItems": 1},
        },
    }
    stripped = for_anthropic(schema)
    props = stripped["properties"]
    assert props["pattern"] == {"type": "string"}
    assert props["n"]["anyOf"][0] == {"type": "integer"}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert props["one"] == {"type": "array", "minItems": 1}
    assert stripped["required"] == ["pattern", "n", "tags"]
    # The input is untouched.
    assert schema["properties"]["tags"]["minItems"] == 2


def _objects(schema):
    """Every subschema of `schema` that describes an object."""
    if isinstance(schema, dict):
        if schema.get("type") == "object" or "properties" in schema:
            yield schema
        for value in schema.values():
            yield from _objects(value)
    elif isinstance(schema, list):
        for value in schema:
            yield from _objects(value)


def test_for_anthropic_closes_every_object_and_rewrites_one_of():
    # Shaped like a WP-D judge schema written loosely: a nested object without
    # additionalProperties, a free-form map and a oneOf.
    schema = {
        "type": "object",
        "required": ["verdict", "spans", "meta"],
        "properties": {
            "verdict": {"oneOf": [{"type": "string", "enum": ["keep", "drop"]}, {"type": "null"}]},
            "spans": {
                "type": "array",
                "items": {"type": "object", "properties": {"index": {"type": "integer", "minimum": 0}}},
            },
            "meta": {"type": "object", "additionalProperties": {"type": "string", "maxLength": 9}},
            "both": {"anyOf": [{"type": "string"}], "oneOf": [{"type": "string"}, {"type": "integer"}]},
            "open": {"type": "object", "properties": {"x": {"type": "string"}}, "additionalProperties": True},
        },
    }
    sent = for_anthropic(schema)
    objects = list(_objects(sent))
    assert len(objects) == 4
    assert all(obj["additionalProperties"] is False for obj in objects)
    assert "oneOf" not in json.dumps(sent)
    assert sent["properties"]["verdict"] == {"anyOf": [{"type": "string", "enum": ["keep", "drop"]}, {"type": "null"}]}
    assert sent["properties"]["spans"]["items"]["properties"]["index"] == {"type": "integer"}
    assert sent["properties"]["both"] == {
        "anyOf": [{"type": "string"}],
        "allOf": [{"anyOf": [{"type": "string"}, {"type": "integer"}]}],
    }
    # The input is untouched and still enforces its own rules client-side.
    assert "additionalProperties" not in schema
    assert "oneOf" in schema["properties"]["verdict"]
    assert validate({"verdict": "keep", "spans": [], "meta": {}}, schema) == []


def test_strict_problems():
    assert strict_problems(ENTITY_SCHEMA) == []
    loose = {"type": "object", "properties": {"a": {"type": "string"}, "b": {"type": "string"}}, "required": ["a"]}
    problems = strict_problems(loose)
    assert "$: additionalProperties must be false" in problems
    assert "$: properties not in required: b" in problems
    nested = {
        "type": "object",
        "additionalProperties": False,
        "required": ["xs"],
        "properties": {"xs": {"type": "array", "items": {"type": "object", "properties": {}}}},
    }
    assert strict_problems(nested) == ["$.xs[]: additionalProperties must be false"]


def test_minimal_instance_is_valid():
    schemas = [
        ENTITY_SCHEMA,
        {"type": "object", "required": ["n", "flag", "kind"], "properties": {
            "n": {"type": "integer", "minimum": 1.5},
            "flag": {"type": "boolean"},
            "kind": {"anyOf": [{"type": "null"}, {"type": "string"}]},
        }},
        {"type": "array", "minItems": 2, "items": {"type": "string", "enum": ["a", "b"]}},
        {"type": ["null", "string"]},
    ]
    for schema in schemas:
        assert validate(minimal_instance(schema), schema) == [], schema
    assert minimal_instance(ENTITY_SCHEMA) == {"entities": []}


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('Here you go:\n```\n{"a": [1, 2]}\n```\nDone.', {"a": [1, 2]}),
        ('Sure! {"a": "}"} hope that helps', {"a": "}"}),
        ("[1, 2]", [1, 2]),
    ],
)
def test_parse_json(text, expected):
    assert parse_json(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "no json here", "{broken"])
def test_parse_json_errors(text):
    with pytest.raises(ValueError):
        parse_json(text)


def test_parse_answer_skips_prose_fragments_that_do_not_fit():
    from opf_eval.llm import parse_answer

    text = 'I found [2] entities: {"entities": []}'
    assert parse_json(text) == [2]  # the first fragment, kept for compatibility
    assert parse_answer(text, ENTITY_SCHEMA) == ({"entities": []}, [])
    answer, errors = parse_answer("I found [2] things", ENTITY_SCHEMA)
    assert answer == [2] and errors == ["$: expected object, got [2]"]
    assert parse_answer("", ENTITY_SCHEMA) == (None, ["$: the answer was empty"])
    assert parse_answer("nope", ENTITY_SCHEMA)[1][0].startswith("$: the answer is not JSON")


def test_prompt_mode_answer_after_a_prose_fragment_needs_no_retry():
    text = 'I found [1] entity: ```json\n' + json.dumps(GOOD) + '\n```'
    sdk = FakeOpenAISDK(gpt_reply(text=text))
    client = OpenAIClient("openai_compatible", "m", base_url="http://localhost:1/v1", mode="prompt", sdk_client=sdk)
    assert client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA) == GOOD
    assert len(sdk.calls) == 1
    claude = AnthropicClient(sdk_client=FakeAnthropicSDK(claude_reply(text='[1] {"entities": []}')), fallbacks=False)
    assert claude.complete_json(system="s", user="u", schema=ENTITY_SCHEMA) == {"entities": []}


# ----------------------------------------------------------------- stub


def test_stub_records_calls_and_validates():
    client = StubClient(lambda system, user, schema: GOOD)
    assert client.complete_json(system="sys", user="text", schema=ENTITY_SCHEMA) == GOOD
    assert client.calls == [("sys", "text")]
    bad = StubClient(lambda system, user, schema: BAD)
    with pytest.raises(LLMError, match="does not match the schema"):
        bad.complete_json(system="s", user="u", schema=ENTITY_SCHEMA)


def test_stub_default_responder_and_make_client_options():
    client = make_client("stub")
    assert client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA) == {"entities": []}
    named = make_client("stub", "stub-a", responder=lambda s, u, sc: GOOD)
    assert named.model == "stub-a"
    assert named.complete_json(system="s", user="u", schema=ENTITY_SCHEMA) == GOOD
    with pytest.raises(TypeError, match="unexpected arguments"):
        make_client("stub", base_urll="x")


def test_stub_responder_errors_propagate():
    def refuse(system, user, schema):
        raise LLMError("refused: cyber")

    with pytest.raises(LLMError, match="refused: cyber"):
        make_client("stub", responder=refuse).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)


# ---------------------------------------------------------------- cache


def test_cache_key_matches_the_spec_formula():
    payload = {"provider": "p", "model": "m", "system": "s", "user": "u", "schema": {"type": "object"}, "name": "n"}
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    assert cache_key(**payload) == expected
    assert cache_key(**payload, settings={}) == expected
    assert cache_key(**payload, settings={"effort": "high"}) != expected


def test_cache_hit_and_miss(tmp_path):
    inner = StubClient(lambda system, user, schema: GOOD, model="openai/gpt-oss-120b-maas")
    client = CachedClient(inner, tmp_path)
    secret = "Jane Doe, jane@example.com, SSN 078-05-1120"
    first = client.complete_json(system="sys", user=secret, schema=ENTITY_SCHEMA)
    second = client.complete_json(system="sys", user=secret, schema=ENTITY_SCHEMA)
    assert first == second == GOOD
    assert len(inner.calls) == 1
    assert (client.hits, client.misses) == (1, 1)
    assert client.stats() == {"hits": 1, "misses": 1}

    key = client.key_for(system="sys", user=secret, schema=ENTITY_SCHEMA)
    path = tmp_path / "stub" / "openai_gpt-oss-120b-maas" / key[:2] / f"{key}.json"
    assert client.path_for(key) == path
    entry = json.loads(path.read_text())
    assert entry["response"] == GOOD
    assert entry["request"]["provider"] == "stub"
    assert entry["request"]["user_chars"] == len(secret)
    assert "created_at" in entry
    # The request part keeps hashes, never the prompt text. (The response
    # quotes the values it found, which is why the cache is sensitive.)
    stored = path.read_text()
    assert secret not in stored
    assert "078-05-1120" not in stored
    assert set(entry["request"]) == {
        "provider", "model", "name", "settings",
        "system_sha256", "user_sha256", "user_chars", "schema_sha256",
    }

    # Anything in the key changes the entry.
    client.complete_json(system="sys", user=secret + "!", schema=ENTITY_SCHEMA)
    client.complete_json(system="other", user=secret, schema=ENTITY_SCHEMA)
    client.complete_json(system="sys", user=secret, schema=ENTITY_SCHEMA, name="other")
    assert (client.hits, client.misses) == (1, 4)
    # A new client over the same directory reuses the entries.
    again = CachedClient(StubClient(lambda *a: pytest.fail("inner called on a hit"), model="openai/gpt-oss-120b-maas"), tmp_path)
    assert again.complete_json(system="sys", user=secret, schema=ENTITY_SCHEMA) == GOOD
    assert again.hits == 1


def test_llm_error_attributes_default_and_survive_pickling():
    import pickle

    plain = LLMError("boom")
    assert str(plain) == "boom" and plain.truncated is False and plain.refusal is None
    cut = pickle.loads(pickle.dumps(LLMError("hit max_tokens=5", truncated=True)))
    assert str(cut) == "hit max_tokens=5" and cut.truncated is True and cut.refusal is None
    refused = pickle.loads(pickle.dumps(LLMError("refused: cyber", refusal="cyber")))
    assert refused.refusal == "cyber" and refused.truncated is False


def test_cache_passes_structured_errors_through(tmp_path):
    def cut_short(system, user, schema):
        raise LLMError("stub hit max_tokens=1 before finishing its answer", truncated=True)

    client = CachedClient(StubClient(cut_short), tmp_path)
    with pytest.raises(LLMError) as info:
        client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert info.value.truncated is True


def test_cache_does_not_store_errors(tmp_path):
    answers = [LLMError("refused: cyber"), GOOD]

    def flaky(system, user, schema):
        item = answers.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    inner = StubClient(flaky)
    client = CachedClient(inner, tmp_path)
    with pytest.raises(LLMError):
        client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert not list(tmp_path.rglob("*.json"))
    assert client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA) == GOOD
    assert len(inner.calls) == 2
    assert (client.hits, client.misses) == (0, 2)


def test_cache_treats_a_corrupt_entry_as_a_miss(tmp_path):
    inner = StubClient(lambda *a: GOOD)
    client = CachedClient(inner, tmp_path)
    key = client.key_for(system="s", user="u", schema=ENTITY_SCHEMA)
    path = client.path_for(key)
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA) == GOOD
    assert client.misses == 1
    assert json.loads(path.read_text())["response"] == GOOD


def test_cache_treats_a_non_utf8_entry_as_a_miss(tmp_path):
    inner = StubClient(lambda *a: GOOD)
    client = CachedClient(inner, tmp_path)
    path = client.path_for(client.key_for(system="s", user="u", schema=ENTITY_SCHEMA))
    path.parent.mkdir(parents=True)
    path.write_bytes(b"\xff\xfe garbage")
    assert client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA) == GOOD
    assert client.misses == 1
    assert json.loads(path.read_text(encoding="utf-8"))["response"] == GOOD


def test_cache_includes_backend_settings(tmp_path):
    low = make_client("anthropic", effort="low", sdk_client=FakeAnthropicSDK(claude_reply(GOOD)), cache_dir=tmp_path)
    high = make_client("anthropic", effort="high", sdk_client=FakeAnthropicSDK(claude_reply(GOOD)), cache_dir=tmp_path)
    assert isinstance(low, CachedClient)
    k_low = low.key_for(system="s", user="u", schema=ENTITY_SCHEMA)
    k_high = high.key_for(system="s", user="u", schema=ENTITY_SCHEMA)
    assert k_low != k_high
    low.complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    entry = json.loads(low.path_for(k_low).read_text())
    assert entry["request"]["settings"] == {"effort": "low"}


def test_cached_client_delegates_attributes(tmp_path):
    client = make_client("anthropic_vertex", sdk_client=FakeAnthropicSDK(), cache_dir=tmp_path)
    assert client.provider == "anthropic_vertex"
    assert client.model == "claude-opus-5-5"
    assert client.remote is True
    assert client.region == "us"
    assert describe(client) == describe(client.inner)
    assert isinstance(client, LLMClient)


def test_cache_is_thread_safe(tmp_path):
    lock = threading.Lock()
    seen: list[str] = []

    def responder(system, user, schema):
        with lock:
            seen.append(user)
        return GOOD

    client = CachedClient(StubClient(responder), tmp_path)
    users = [f"text {i % 5}" for i in range(40)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda u: client.complete_json(system="s", user=u, schema=ENTITY_SCHEMA), users))
    assert all(r == GOOD for r in results)
    assert client.hits + client.misses == 40
    assert len(set(seen)) == 5
    assert len(list(tmp_path.rglob("*.json"))) == 5
    assert not list(tmp_path.rglob(".tmp-*"))


# ------------------------------------------------------------ anthropic


def test_anthropic_success_uses_structured_output_and_fallbacks():
    sdk = FakeAnthropicSDK(claude_reply(GOOD))
    client = AnthropicClient("anthropic", sdk_client=sdk, fallbacks=True)
    assert client.complete_json(system="be careful", user="text", schema=ENTITY_SCHEMA, max_tokens=1234) == GOOD
    kind, kwargs = sdk.calls[0]
    assert kind == "beta"
    assert kwargs["betas"] == [FALLBACK_BETA]
    assert kwargs["fallbacks"] == "default"
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["max_tokens"] == 1234
    assert kwargs["system"] == "be careful"
    assert kwargs["messages"] == [{"role": "user", "content": "text"}]
    assert kwargs["output_config"] == {
        "format": {"type": "json_schema", "schema": ENTITY_SCHEMA},
        "effort": "medium",
    }
    # Opus 5.5 always thinks adaptively and rejects sampling parameters.
    assert "thinking" not in kwargs
    assert "temperature" not in kwargs


def test_anthropic_fallbacks_follow_the_installed_sdk():
    from opf_eval.llm.anthropic import sdk_supports_fallbacks

    client = AnthropicClient("anthropic", sdk_client=FakeAnthropicSDK())
    assert client.use_fallbacks is sdk_supports_fallbacks()


def test_anthropic_without_fallbacks_uses_plain_messages():
    sdk = FakeAnthropicSDK(claude_reply(GOOD))
    client = AnthropicClient("anthropic", sdk_client=sdk, fallbacks=False, effort="high")
    client.complete_json(system="", user="text", schema=ENTITY_SCHEMA)
    kind, kwargs = sdk.calls[0]
    assert kind == "messages"
    assert "betas" not in kwargs and "fallbacks" not in kwargs
    assert "system" not in kwargs  # an empty system prompt is left out
    assert kwargs["output_config"]["effort"] == "high"


def test_anthropic_vertex_never_sends_fallbacks(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-123")
    sdk = FakeAnthropicSDK(claude_reply(GOOD))
    client = AnthropicClient("anthropic_vertex", sdk_client=sdk, fallbacks=True)
    assert client.project_id == "proj-123"
    assert client.region == "us"
    client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert sdk.calls[0][0] == "messages"


def test_anthropic_vertex_client_wiring(monkeypatch):
    built = {}

    class FakeVertex:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(anthropic, "AnthropicVertex", FakeVertex)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "proj-123")
    client = make_client("anthropic_vertex", max_retries=5)
    assert built == {"project_id": "proj-123", "region": "us", "max_retries": 5}
    assert isinstance(client._sdk_client, FakeVertex)

    built.clear()
    monkeypatch.setenv("CLOUD_ML_REGION", "us-east5")
    make_client("anthropic_vertex", project_id="other", timeout=30)
    assert built == {"project_id": "other", "region": "us-east5", "max_retries": 2, "timeout": 30}


def test_anthropic_first_party_client_wiring(monkeypatch):
    built = {}

    class FakeAnthropic:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(anthropic, "Anthropic", FakeAnthropic)
    make_client("anthropic")
    assert built == {"max_retries": 2}


def test_anthropic_missing_credentials_is_an_llm_error(monkeypatch):
    def no_creds(**kwargs):
        raise anthropic.AnthropicError("no credentials")

    monkeypatch.setattr(anthropic, "Anthropic", no_creds)
    with pytest.raises(LLMError, match="ant auth login"):
        make_client("anthropic")


def test_anthropic_refusal_is_an_error():
    sdk = FakeAnthropicSDK(claude_reply(text="", stop_reason="refusal", category="cyber"))
    with pytest.raises(LLMError, match="refused: cyber") as info:
        AnthropicClient(sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert len(sdk.calls) == 1  # a refusal is not retried by the client
    assert info.value.refusal == "cyber" and info.value.truncated is False

    sdk = FakeAnthropicSDK(claude_reply(stop_reason="refusal"))
    with pytest.raises(LLMError, match="refused: unspecified") as info:
        AnthropicClient(sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert info.value.refusal == "unspecified"


def test_anthropic_max_tokens_is_an_error():
    sdk = FakeAnthropicSDK(claude_reply(text='{"entities": [', stop_reason="max_tokens"))
    with pytest.raises(LLMError, match="max_tokens=50") as info:
        AnthropicClient(sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA, max_tokens=50)
    assert info.value.truncated is True and info.value.refusal is None


def test_anthropic_answer_after_a_server_side_fallback():
    fallback = SimpleNamespace(type="fallback")
    sdk = FakeAnthropicSDK(claude_reply(GOOD, blocks=[fallback]))
    assert AnthropicClient(sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA) == GOOD


def test_anthropic_invalid_answer_retries_once_with_the_errors():
    sdk = FakeAnthropicSDK(claude_reply(BAD), claude_reply(GOOD))
    client = AnthropicClient(sdk_client=sdk)
    assert client.complete_json(system="s", user="the text", schema=ENTITY_SCHEMA) == GOOD
    assert len(sdk.calls) == 2
    retry_user = sdk.calls[1][1]["messages"][0]["content"]
    assert retry_user.startswith("the text")
    assert "did not match the required JSON schema" in retry_user
    assert "$.entities[0].label" in retry_user


def test_anthropic_non_json_then_valid():
    sdk = FakeAnthropicSDK(claude_reply(text="I cannot format that"), claude_reply(GOOD))
    assert AnthropicClient(sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA) == GOOD


def test_anthropic_gives_up_after_one_retry():
    sdk = FakeAnthropicSDK(claude_reply(BAD), claude_reply(BAD))
    with pytest.raises(LLMError, match="after one retry"):
        AnthropicClient(sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert len(sdk.calls) == 2


def test_anthropic_strips_unsupported_keywords_but_still_enforces_them():
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["score"],
        "properties": {"score": {"type": "integer", "minimum": 0, "maximum": 10}},
    }
    sdk = FakeAnthropicSDK(claude_reply({"score": 42}), claude_reply({"score": 7}))
    assert AnthropicClient(sdk_client=sdk).complete_json(system="s", user="u", schema=schema) == {"score": 7}
    sent = sdk.calls[0][1]["output_config"]["format"]["schema"]
    assert sent["properties"]["score"] == {"type": "integer"}
    assert "above the maximum 10" in sdk.calls[1][1]["messages"][0]["content"]


def test_anthropic_transport_errors_become_llm_errors():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    sdk = FakeAnthropicSDK(anthropic.APIConnectionError(request=request))
    with pytest.raises(LLMError, match="APIConnectionError"):
        AnthropicClient(sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)


@pytest.mark.parametrize("exc", [
    ValueError("Streaming is required for operations that may take longer than 10 minutes."),
    UnicodeEncodeError("utf-8", "Ann \ud83d", 4, 5, "surrogates not allowed"),
])
def test_anthropic_other_sdk_exceptions_become_llm_errors(exc):
    sdk = FakeAnthropicSDK(exc)
    with pytest.raises(LLMError, match=type(exc).__name__):
        AnthropicClient(sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA, max_tokens=32000)


def test_anthropic_reports_the_model_behind_a_server_side_fallback(tmp_path):
    fallback = SimpleNamespace(type="fallback", to=SimpleNamespace(model="claude-sonnet-4-5"))
    sdk = FakeAnthropicSDK(claude_reply(GOOD, blocks=[fallback]), claude_reply(GOOD))
    client = AnthropicClient(sdk_client=sdk, fallbacks=True)
    client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert client.pop_fallbacks() == ["claude-sonnet-4-5"]
    assert client.pop_fallbacks() == []
    client.complete_json(system="s", user="u2", schema=ENTITY_SCHEMA)
    assert client.pop_fallbacks() == []  # the requested model answered
    # The usage iterations name the fallback model too.
    usage = SimpleNamespace(iterations=[SimpleNamespace(type="message"),
                                        SimpleNamespace(type="fallback_message", model="claude-haiku-4-5")])
    reply = claude_reply(GOOD)
    reply.usage = usage
    assert llm.anthropic.fallback_model(reply) == "claude-haiku-4-5"

    # The cache records the fallback and reports it again on a hit.
    sdk = FakeAnthropicSDK(claude_reply(GOOD, blocks=[fallback]))
    cached = CachedClient(AnthropicClient(sdk_client=sdk, fallbacks=True), tmp_path)
    cached.complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert cached.pop_fallbacks() == ["claude-sonnet-4-5"]
    (entry,) = [json.loads(f.read_text()) for f in tmp_path.rglob("*.json")]
    assert entry["fallback_model"] == "claude-sonnet-4-5"
    cached.complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert cached.hits == 1
    assert cached.pop_fallbacks() == ["claude-sonnet-4-5"]
    assert llm.pop_fallbacks(StubClient()) == []


def test_lone_surrogates_are_scrubbed_from_prompts_and_answers(tmp_path):
    lone = '{"entities": [{"value": "Ann \\ud83d", "label": "PERSON", "context": "Ann \\ud83d went"}]}'
    sdk = FakeAnthropicSDK(claude_reply(text=lone))
    client = CachedClient(AnthropicClient(sdk_client=sdk), tmp_path)
    answer = client.complete_json(system="s", user="Ann \ud83d went", schema=ENTITY_SCHEMA)
    assert answer["entities"][0]["value"] == "Ann \ufffd"
    assert sdk.calls[0][1]["messages"][0]["content"] == "Ann \ufffd went"  # same length, valid UTF-8
    # The answer was cached, so a rerun is a hit.
    assert client.complete_json(system="s", user="Ann \ud83d went", schema=ENTITY_SCHEMA) == answer
    assert (client.hits, client.misses) == (1, 1)

    sdk = FakeOpenAISDK(gpt_reply(text=lone))
    oai = make_client("openai", sdk_client=sdk)
    assert oai.complete_json(system="s", user="Ann \ud83d", schema=ENTITY_SCHEMA)["entities"][0]["value"] == "Ann \ufffd"
    assert sdk.calls[0]["messages"][-1]["content"] == "Ann \ufffd"


def test_anthropic_rejects_bad_options():
    with pytest.raises(ValueError, match="effort"):
        AnthropicClient(effort="extreme", sdk_client=FakeAnthropicSDK())
    with pytest.raises(ValueError, match="does not serve"):
        AnthropicClient("openai", sdk_client=FakeAnthropicSDK())


# --------------------------------------------------------------- openai


def test_openai_other_sdk_exceptions_become_llm_errors():
    sdk = FakeOpenAISDK(UnicodeEncodeError("utf-8", "Ann \ud83d", 4, 5, "surrogates not allowed"))
    with pytest.raises(LLMError, match="UnicodeEncodeError"):
        make_client("openai", sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)


def test_openai_success_uses_strict_json_schema():
    sdk = FakeOpenAISDK(gpt_reply(GOOD))
    client = make_client("openai", sdk_client=sdk)
    assert client.model == "gpt-6.1-sol"
    assert client.complete_json(system="sys", user="text", schema=ENTITY_SCHEMA, name="silver labels", max_tokens=900) == GOOD
    kwargs = sdk.calls[0]
    assert kwargs["model"] == "gpt-6.1-sol"
    assert kwargs["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "text"}]
    assert kwargs["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "silver_labels", "schema": ENTITY_SCHEMA, "strict": True},
    }
    assert kwargs["max_completion_tokens"] == 900
    assert "max_tokens" not in kwargs


def test_openai_non_strict_schema_and_reasoning_effort():
    loose = {"type": "object", "properties": {"a": {"type": "string"}}}
    sdk = FakeOpenAISDK(gpt_reply({"a": "x"}))
    client = make_client("openai", "gpt-5.6-luna", sdk_client=sdk, reasoning_effort="low")
    client.complete_json(system="", user="u", schema=loose)
    kwargs = sdk.calls[0]
    assert kwargs["response_format"]["json_schema"]["strict"] is False
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["messages"] == [{"role": "user", "content": "u"}]
    assert client.cache_settings() == {"reasoning_effort": "low"}


def test_openai_refusal_and_truncation_are_errors():
    sdk = FakeOpenAISDK(gpt_reply(text=None, refusal="I can't help with that."))
    with pytest.raises(LLMError, match="refused: I can't help") as info:
        make_client("openai", sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert info.value.refusal == "I can't help with that." and info.value.truncated is False
    sdk = FakeOpenAISDK(gpt_reply(text="", finish_reason="content_filter"))
    with pytest.raises(LLMError, match="refused: content_filter") as info:
        make_client("openai", sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert info.value.refusal == "content_filter"
    sdk = FakeOpenAISDK(gpt_reply(text='{"entities": [', finish_reason="length"))
    with pytest.raises(LLMError, match="max_tokens") as info:
        make_client("openai", sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert info.value.truncated is True and info.value.refusal is None
    sdk = FakeOpenAISDK(SimpleNamespace(choices=[]))
    with pytest.raises(LLMError, match="no choices"):
        make_client("openai", sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)


def test_openai_invalid_answer_retries_once():
    sdk = FakeOpenAISDK(gpt_reply(BAD), gpt_reply(GOOD))
    assert make_client("openai", sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA) == GOOD
    assert "did not match the required JSON schema" in sdk.calls[1]["messages"][-1]["content"]
    sdk = FakeOpenAISDK(gpt_reply(BAD), gpt_reply(text="nope"))
    with pytest.raises(LLMError, match="after one retry"):
        make_client("openai", sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)


def test_openai_api_does_not_step_down_on_a_400():
    sdk = FakeOpenAISDK(openai_400("Invalid schema for response_format 'result'"))
    with pytest.raises(LLMError, match="response_format"):
        make_client("openai", sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert len(sdk.calls) == 1


def test_compatible_falls_back_from_json_schema_to_json_object():
    sdk = FakeOpenAISDK(
        openai_400("response_format type json_schema is not supported"),
        gpt_reply(GOOD),
        gpt_reply(GOOD),
    )
    client = make_client("openai_compatible", "llama3", base_url="http://localhost:8000/v1", sdk_client=sdk)
    assert client.mode == "json_schema"
    assert client.complete_json(system="label PII", user="u", schema=ENTITY_SCHEMA, max_tokens=300) == GOOD
    assert sdk.calls[0]["response_format"]["type"] == "json_schema"
    second = sdk.calls[1]
    assert second["response_format"] == {"type": "json_object"}
    system = second["messages"][0]["content"]
    assert system.startswith("label PII\n\n")
    assert json.dumps(ENTITY_SCHEMA) in system
    assert second["max_tokens"] == 300 and "max_completion_tokens" not in second
    # The step down is remembered for the next call.
    assert client.mode == "json_object"
    client.complete_json(system="label PII", user="u2", schema=ENTITY_SCHEMA)
    assert sdk.calls[2]["response_format"] == {"type": "json_object"}
    assert len(sdk.calls) == 3


def test_compatible_steps_down_to_prompt_only_and_parses_fences():
    sdk = FakeOpenAISDK(
        openai_400("response_format json_schema unsupported"),
        openai_400("'response_format' of type 'json_object' is not supported by this model"),
        gpt_reply(text="Here it is:\n```json\n" + json.dumps(GOOD) + "\n```"),
    )
    client = make_client("openai_compatible", "m", base_url="http://127.0.0.1:1234/v1", sdk_client=sdk)
    assert client.complete_json(system="", user="u", schema=ENTITY_SCHEMA) == GOOD
    assert "response_format" not in sdk.calls[2]
    assert "JSON Schema" in sdk.calls[2]["messages"][0]["content"]
    assert client.mode == "prompt"


def test_compatible_other_400s_are_errors():
    sdk = FakeOpenAISDK(openai_400("model 'm' not found"))
    client = make_client("openai_compatible", "m", base_url="http://localhost:1/v1", sdk_client=sdk)
    with pytest.raises(LLMError, match="not found"):
        client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    assert client.mode == "json_schema"


def test_compatible_forced_mode():
    sdk = FakeOpenAISDK(openai_400("response_format unsupported"))
    client = make_client("openai_compatible", "m", base_url="http://localhost:1/v1", sdk_client=sdk, mode="json_schema")
    with pytest.raises(LLMError):
        client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA)
    with pytest.raises(ValueError, match="mode"):
        make_client("openai_compatible", "m", base_url="http://localhost:1/v1", sdk_client=sdk, mode="xml")


def test_compatible_client_wiring_and_key(monkeypatch):
    built = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-openai-key")
    client = make_client("openai_compatible", "qwen", base_url="https://gpu-box.example.com/v1")
    # A real OpenAI key is never sent to another server.
    assert built == {"max_retries": 2, "base_url": "https://gpu-box.example.com/v1", "api_key": "not-needed"}
    assert client.remote is True
    assert client.operator == "the server at https://gpu-box.example.com/v1"
    assert client.cache_settings() == {"base_url": "https://gpu-box.example.com/v1"}

    monkeypatch.setenv("OPENAI_COMPATIBLE_API_KEY", "local-key")
    make_client("openai_compatible", "qwen", base_url="http://localhost:1/v1")
    assert built["api_key"] == "local-key"
    make_client("openai_compatible", "qwen", base_url="http://localhost:1/v1", api_key="explicit")
    assert built["api_key"] == "explicit"
    with pytest.raises(ValueError, match="model id"):
        make_client("openai_compatible", base_url="http://localhost:1/v1")


def test_openai_first_party_wiring_and_missing_key(monkeypatch):
    built = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    make_client("openai")
    assert built == {"max_retries": 2}  # the SDK reads OPENAI_API_KEY itself

    with pytest.raises(ValueError, match="openai_compatible"):
        make_client("openai", base_url="https://elsewhere.example.com/v1")

    def no_key(**kwargs):
        raise openai.OpenAIError("Missing credentials")

    monkeypatch.setattr(openai, "OpenAI", no_key)
    with pytest.raises(LLMError, match="OPENAI_API_KEY"):
        make_client("openai")


def test_openai_transport_errors_become_llm_errors():
    request = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")
    sdk = FakeOpenAISDK(openai.APIConnectionError(request=request))
    with pytest.raises(LLMError, match="APIConnectionError"):
        make_client("openai", sdk_client=sdk).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)


class FakeCredentials:
    def __init__(self):
        self.token = None
        self.valid = False
        self.refreshes = 0

    def refresh(self, request):
        self.refreshes += 1
        self.token = f"ya29.token-{self.refreshes}"
        self.valid = True


def test_vertex_base_url():
    assert vertex_base_url("p", "us-central1") == (
        "https://us-central1-aiplatform.googleapis.com/v1/projects/p/locations/us-central1/endpoints/openapi"
    )
    assert vertex_base_url("p", "global") == (
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi"
    )
    assert vertex_base_url("p", "us").startswith("https://aiplatform.us.rep.googleapis.com/v1/projects/p/locations/us/")


def test_openai_vertex_wiring_uses_adc_tokens(monkeypatch):
    import google.auth

    creds = FakeCredentials()
    scopes_seen = []

    def fake_default(scopes=None, **kwargs):
        scopes_seen.append(scopes)
        return creds, "adc-project"

    built = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(google.auth, "default", fake_default)
    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)

    client = make_client("openai_vertex")
    assert scopes_seen == [["https://www.googleapis.com/auth/cloud-platform"]]
    assert client.model == "openai/gpt-oss-120b-maas"
    assert client.project_id == "adc-project"
    assert client.region == "us-central1"
    assert built["base_url"] == vertex_base_url("adc-project", "us-central1")
    token = built["api_key"]
    assert isinstance(token, GoogleTokenProvider)
    # The SDK calls the provider before each request; it refreshes only when needed.
    assert token() == "ya29.token-1"
    assert token() == "ya29.token-1"
    creds.valid = False  # expired
    assert token() == "ya29.token-2"
    assert creds.refreshes == 2
    assert describe(client).startswith("openai/gpt-oss-120b-maas via your GCP project (Vertex AI), region us-central1")

    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "env-project")
    make_client("openai_vertex", region="global")
    assert built["base_url"] == vertex_base_url("env-project", "global")
    make_client("openai_vertex", project_id="explicit", region="us-east5")
    assert built["base_url"] == vertex_base_url("explicit", "us-east5")


def test_openai_vertex_without_credentials(monkeypatch):
    import google.auth
    import google.auth.exceptions

    def no_adc(**kwargs):
        raise google.auth.exceptions.DefaultCredentialsError("no ADC")

    monkeypatch.setattr(google.auth, "default", no_adc)
    with pytest.raises(LLMError, match="gcloud auth application-default login"):
        make_client("openai_vertex")


def test_openai_vertex_needs_a_project(monkeypatch):
    with pytest.raises(LLMError, match="GOOGLE_CLOUD_PROJECT"):
        OpenAIClient("openai_vertex", token_provider=SimpleNamespace(project_id=None), sdk_client=FakeOpenAISDK())


def test_anthropic_vertex_takes_the_project_from_adc(monkeypatch):
    import google.auth

    monkeypatch.setattr(google.auth, "default", lambda scopes=None, **kw: (FakeCredentials(), "adc-project"))
    built = {}

    class FakeVertex:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(anthropic, "AnthropicVertex", FakeVertex)
    client = make_client("anthropic_vertex")
    assert client.project_id == "adc-project"
    assert built["project_id"] == "adc-project"


def test_anthropic_vertex_without_a_project_fails_at_construction(monkeypatch):
    import google.auth
    import google.auth.exceptions

    monkeypatch.setattr(anthropic, "AnthropicVertex", lambda **kw: pytest.fail("SDK client was built"))
    monkeypatch.setattr(google.auth, "default", lambda scopes=None, **kw: (FakeCredentials(), None))
    with pytest.raises(LLMError, match="anthropic_vertex needs a GCP project"):
        make_client("anthropic_vertex")

    def no_adc(**kwargs):
        raise google.auth.exceptions.DefaultCredentialsError("no ADC")

    monkeypatch.setattr(google.auth, "default", no_adc)
    with pytest.raises(LLMError, match="gcloud auth application-default login"):
        make_client("anthropic_vertex")


class RevokedCredentials:
    """ADC whose refresh fails, as after `invalid_grant` on a long run."""

    token = None
    valid = False
    expired = True

    def __init__(self, error):
        self.error = error

    def refresh(self, request):
        raise self.error


def _google_errors():
    import google.auth.exceptions as gae

    return [
        gae.RefreshError("invalid_grant: Token has been expired or revoked."),
        gae.TransportError("token endpoint unreachable"),
        gae.DefaultCredentialsError("no ADC"),
    ]


@pytest.mark.parametrize("error", _google_errors(), ids=lambda e: type(e).__name__)
def test_openai_vertex_token_failure_is_an_llm_error(error):
    def handler(request):  # never reached: the token fails first
        raise AssertionError("request sent without a token")

    sdk = openai.OpenAI(
        api_key=GoogleTokenProvider(RevokedCredentials(error)),
        base_url="http://vertex.test/v1",
        http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    client = OpenAIClient("openai_vertex", project_id="p", sdk_client=sdk)
    with pytest.raises(LLMError, match="Google credentials failed.*gcloud auth application-default login"):
        client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA)


@pytest.mark.parametrize("error", _google_errors(), ids=lambda e: type(e).__name__)
def test_anthropic_vertex_token_failure_is_an_llm_error(error):
    def handler(request):
        raise AssertionError("request sent without a token")

    sdk = anthropic.AnthropicVertex(
        region="us",
        project_id="p",
        credentials=RevokedCredentials(error),
        base_url="http://vertex.test",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    client = AnthropicClient("anthropic_vertex", project_id="p", sdk_client=sdk)
    with pytest.raises(LLMError, match="Google credentials failed.*gcloud auth application-default login"):
        client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA)


def test_openai_vertex_falls_back_like_a_compatible_server(monkeypatch):
    sdk = FakeOpenAISDK(openai_400("Unsupported parameter: response_format.json_schema"), gpt_reply(GOOD))
    client = OpenAIClient("openai_vertex", project_id="p", sdk_client=sdk)
    assert client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA) == GOOD
    assert sdk.calls[1]["response_format"] == {"type": "json_object"}
    assert sdk.calls[1]["max_tokens"] == 8000


# ------------------------------------- real SDK objects, in-memory transport
#
# These drive the installed SDKs end to end against an httpx2 MockTransport,
# so a parameter the SDK does not accept or serialises differently fails here.
# Nothing leaves the process.


def test_anthropic_request_body_through_the_real_sdk():
    seen = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append((request, json.loads(request.content)))
        return httpx2.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [{"type": "text", "text": json.dumps(GOOD)}],
                "stop_reason": "end_turn",
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 10},
            },
        )

    sdk = anthropic.Anthropic(
        api_key="test-key",
        base_url="http://anthropic.test",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    client = AnthropicClient(sdk_client=sdk, fallbacks=True, effort="low")
    assert client.complete_json(system="sys", user="text", schema=ENTITY_SCHEMA) == GOOD
    request, body = seen[0]
    assert FALLBACK_BETA in request.headers["anthropic-beta"]
    assert body["fallbacks"] == "default"
    assert body["output_config"] == {"format": {"type": "json_schema", "schema": ENTITY_SCHEMA}, "effort": "low"}
    assert body["system"] == "sys"
    assert "thinking" not in body and "temperature" not in body


def test_anthropic_refusal_through_the_real_sdk():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(
            200,
            json={
                "id": "msg_2",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [],
                "stop_reason": "refusal",
                "stop_details": {"type": "refusal", "category": "bio", "explanation": None},
                "stop_sequence": None,
                "usage": {"input_tokens": 10, "output_tokens": 0},
            },
        )

    sdk = anthropic.Anthropic(
        api_key="test-key",
        base_url="http://anthropic.test",
        http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    with pytest.raises(LLMError, match="refused: bio"):
        AnthropicClient(sdk_client=sdk, fallbacks=False).complete_json(system="s", user="u", schema=ENTITY_SCHEMA)


def test_openai_compatible_fallback_through_the_real_sdk():
    bodies = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if body.get("response_format", {}).get("type") == "json_schema":
            return httpx2.Response(400, json={"error": {"message": "response_format json_schema is not supported"}})
        return httpx2.Response(
            200,
            json={
                "id": "c1",
                "object": "chat.completion",
                "created": 0,
                "model": "llama3",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(GOOD), "refusal": None},
                    }
                ],
            },
        )

    sdk = openai.OpenAI(
        api_key="not-needed",
        base_url="http://localhost:9/v1",
        http_client=openai.DefaultHttpxClient(transport=httpx2.MockTransport(handler)),
        max_retries=0,
    )
    client = OpenAIClient("openai_compatible", "llama3", base_url="http://localhost:9/v1", sdk_client=sdk)
    assert client.complete_json(system="s", user="u", schema=ENTITY_SCHEMA, max_tokens=256) == GOOD
    assert [b.get("response_format", {}).get("type") for b in bodies] == ["json_schema", "json_object"]
    assert bodies[0]["response_format"]["json_schema"]["strict"] is True
    assert bodies[1]["max_tokens"] == 256
