"""A small JSON Schema validator for the subset our prompts use, plus helpers
that adapt a schema to each provider's structured-output rules.

Every LLM answer is checked here even when the provider promises structured
output because some backends (local servers, Vertex gpt-oss) only promise
valid JSON and because Anthropic's structured outputs drop numeric and length
constraints (see `for_anthropic`).

    from opf_eval.llm.schema import validate
    errors = validate({"entities": []}, schema)   # [] when valid

Supported keywords: type (a name or a list of names), properties, required,
additionalProperties (bool or schema), items, enum, const, minimum, maximum,
exclusiveMinimum, exclusiveMaximum, minLength, maxLength, pattern, minItems,
maxItems, anyOf, allOf, oneOf. Annotation keywords (title, description,
default, examples, format) are ignored. `$ref` is not supported and raises
ValueError so a schema that relies on it fails loudly instead of passing.

Two rules are stricter than JSON Schema so that a validated answer has the
Python types its schema promises. "integer" accepts only an int and never a
float such as 2.0. "number" rejects NaN and infinity. enum and const compare
by JSON type, so True does not match 1.
"""

from __future__ import annotations

import copy
import json
import math
import re
from typing import Any

_TYPES = ("object", "array", "string", "integer", "number", "boolean", "null")
_UNSUPPORTED = ("$ref", "$dynamicRef", "not", "if", "then", "else", "dependentSchemas")


def _type_ok(value: Any, name: str) -> bool:
    # bool is a subclass of int in Python, so it is excluded from the numbers.
    # "integer" means a Python int: JSON Schema would accept 2.0, but callers
    # use these values as list indexes and offsets, so a float is sent back to
    # the model on the retry instead. NaN and infinity are never numbers.
    if name == "object":
        return isinstance(value, dict)
    if name == "array":
        return isinstance(value, list)
    if name == "string":
        return isinstance(value, str)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "null":
        return value is None
    if name == "integer":
        if isinstance(value, bool):
            return False
        return isinstance(value, int)
    if name == "number":
        if isinstance(value, bool):
            return False
        return isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
    raise ValueError(f"unknown JSON Schema type {name!r}")


def _same(a: Any, b: Any) -> bool:
    """JSON equality for enum and const: True is not 1, but 1 equals 1.0."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return type(a) is type(b) and a == b


def _short(value: Any, limit: int = 60) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def validate(instance: Any, schema: dict, *, path: str = "$") -> list[str]:
    """Return a list of human-readable errors; an empty list means valid.

    Each error starts with a JSONPath-like location such as
    `$.entities[2].label` so it can be shown to the model on a retry.
    """
    errors: list[str] = []
    _validate(instance, schema, path, errors)
    return errors


def is_valid(instance: Any, schema: dict) -> bool:
    return not validate(instance, schema)


def _validate(value: Any, schema: Any, path: str, errors: list[str]) -> None:
    if schema is True or schema == {}:
        return
    if schema is False:
        errors.append(f"{path}: no value is allowed here")
        return
    if not isinstance(schema, dict):
        raise TypeError(f"{path}: schema must be a dict or bool, got {type(schema).__name__}")
    for key in _UNSUPPORTED:
        if key in schema:
            raise ValueError(f"{path}: JSON Schema keyword {key!r} is not supported")

    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        for name in names:
            if name not in _TYPES:
                raise ValueError(f"{path}: unknown JSON Schema type {name!r}")
        if not any(_type_ok(value, n) for n in names):
            expected = " or ".join(names)
            errors.append(f"{path}: expected {expected}, got {_short(value)}")
            return  # further checks would only repeat the type error

    if "const" in schema and not _same(value, schema["const"]):
        errors.append(f"{path}: must equal {_short(schema['const'])}, got {_short(value)}")
    if "enum" in schema and not any(_same(value, option) for option in schema["enum"]):
        allowed = ", ".join(_short(v, 30) for v in schema["enum"][:12])
        if len(schema["enum"]) > 12:
            allowed += ", ..."
        errors.append(f"{path}: {_short(value)} is not one of [{allowed}]")

    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} is below the minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} is above the maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {value} must be above {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            errors.append(f"{path}: {value} must be below {schema['exclusiveMaximum']}")

    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: shorter than {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: longer than {schema['maxLength']} characters")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{path}: {_short(value)} does not match /{schema['pattern']}/")

    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: needs at least {schema['minItems']} items, got {len(value)}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: allows at most {schema['maxItems']} items, got {len(value)}")
        if "items" in schema:
            for i, item in enumerate(value):
                _validate(item, schema["items"], f"{path}[{i}]", errors)

    if isinstance(value, dict):
        props: dict = schema.get("properties", {})
        for key in schema.get("required", []):
            if key not in value:
                errors.append(f"{path}: missing required property {key!r}")
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in props:
                _validate(item, props[key], f"{path}.{key}", errors)
            elif extra is False:
                errors.append(f"{path}: unexpected property {key!r}")
            elif isinstance(extra, dict):
                _validate(item, extra, f"{path}.{key}", errors)

    if "allOf" in schema:
        for sub in schema["allOf"]:
            _validate(value, sub, path, errors)
    if "anyOf" in schema and not any(not validate(value, sub, path=path) for sub in schema["anyOf"]):
        errors.append(f"{path}: {_short(value)} matches none of the allowed shapes")
    if "oneOf" in schema:
        n = sum(1 for sub in schema["oneOf"] if not validate(value, sub, path=path))
        if n != 1:
            errors.append(f"{path}: must match exactly one allowed shape, matched {n}")


# ------------------------------------------------------ provider adapters

# Keywords Anthropic's structured outputs reject. The SDK's `messages.parse`
# strips these for Pydantic models; with a raw schema we strip them ourselves
# and enforce them client-side through `validate` and the one retry.
ANTHROPIC_UNSUPPORTED = (
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "pattern",
    "maxItems",
    "uniqueItems",
)

_SUBSCHEMA_MAPS = ("properties", "$defs", "definitions")
_SUBSCHEMA_LISTS = ("anyOf", "allOf", "oneOf")


def _is_object_schema(schema: dict) -> bool:
    kind = schema.get("type")
    return kind == "object" or (isinstance(kind, list) and "object" in kind) or "properties" in schema


def _strip(schema: Any, drop: set[str]) -> Any:
    if not isinstance(schema, dict):
        return schema
    out: dict = {}
    for key, value in schema.items():
        if key in drop:
            continue
        if key == "minItems" and isinstance(value, int) and value > 1:
            continue  # only minItems 0 and 1 are accepted
        if key in _SUBSCHEMA_MAPS and isinstance(value, dict):
            out[key] = {k: _strip(v, drop) for k, v in value.items()}
        elif key in _SUBSCHEMA_LISTS and isinstance(value, list):
            out[key] = [_strip(v, drop) for v in value]
        elif key in ("items", "additionalProperties") and isinstance(value, dict):
            out[key] = _strip(value, drop)
        else:
            out[key] = copy.deepcopy(value)
    if "oneOf" in out:
        # Only anyOf and allOf are accepted. anyOf is looser than oneOf, and
        # `validate` still enforces "exactly one" on the original schema.
        one_of = out.pop("oneOf")
        if "anyOf" in out:
            out["allOf"] = [*out.get("allOf", []), {"anyOf": one_of}]
        else:
            out["anyOf"] = one_of
    if _is_object_schema(out):
        # Every object must say additionalProperties: false.
        out["additionalProperties"] = False
    return out


def for_anthropic(schema: dict) -> dict:
    """A copy of `schema` that Anthropic's structured outputs accept.

    It drops the keywords in `ANTHROPIC_UNSUPPORTED` and minItems above 1,
    rewrites oneOf as anyOf and sets `additionalProperties: false` on every
    object, as the Anthropic SDK's own schema adapter does. An object that
    allowed extra keys therefore gets none from Claude, which still passes
    `validate` against the original schema. Property names are never touched,
    so a property called "pattern" survives.
    """
    return _strip(schema, set(ANTHROPIC_UNSUPPORTED))


def strict_problems(schema: Any, *, path: str = "$") -> list[str]:
    """Reasons `schema` cannot be sent with OpenAI's `strict: true`.

    Strict mode needs every object to list all its properties in `required`
    and to set `additionalProperties: false`. An empty list means strict mode
    is safe.
    """
    problems: list[str] = []
    if not isinstance(schema, dict):
        return problems
    if _is_object_schema(schema):
        props = schema.get("properties", {})
        if schema.get("additionalProperties", True) is not False:
            problems.append(f"{path}: additionalProperties must be false")
        missing = [k for k in props if k not in schema.get("required", [])]
        if missing:
            problems.append(f"{path}: properties not in required: {', '.join(missing)}")
        for key, sub in props.items():
            problems += strict_problems(sub, path=f"{path}.{key}")
    if isinstance(schema.get("items"), dict):
        problems += strict_problems(schema["items"], path=f"{path}[]")
    for key in _SUBSCHEMA_LISTS:
        for i, sub in enumerate(schema.get(key, []) or []):
            problems += strict_problems(sub, path=f"{path}.{key}[{i}]")
    return problems


def minimal_instance(schema: Any) -> Any:
    """The smallest value that satisfies `schema`, for dry-run stub clients.

    Objects get their required properties, arrays get `minItems` items, enums
    take their first value and numbers their minimum (or 0).
    """
    if not isinstance(schema, dict) or not schema:
        return None
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    if "enum" in schema:
        return copy.deepcopy(schema["enum"][0])
    for key in ("anyOf", "oneOf"):
        if schema.get(key):
            return minimal_instance(schema[key][0])
    kind = schema.get("type")
    if isinstance(kind, list):
        kind = "null" if "null" in kind else kind[0]
    if kind is None and "properties" in schema:
        kind = "object"
    if kind == "object":
        props = schema.get("properties", {})
        return {k: minimal_instance(props.get(k, {})) for k in schema.get("required", [])}
    if kind == "array":
        item = schema.get("items", {})
        return [minimal_instance(item) for _ in range(schema.get("minItems", 0))]
    if kind == "string":
        return "x" * schema.get("minLength", 0)
    if kind in ("integer", "number"):
        low = schema.get("minimum", 0)
        return math.ceil(low) if kind == "integer" else low
    if kind == "boolean":
        return False
    return None
