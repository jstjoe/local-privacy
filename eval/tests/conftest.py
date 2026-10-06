"""Shared pytest setup for the eval tests.

Tests marked `slow` build small models and fetch files from the Hugging Face
Hub. They are skipped unless `PII_BENCH_SLOW_TESTS=1` is set, so the default
test command never touches the network. Run them with

    PII_BENCH_SLOW_TESTS=1 uv run --with pytest python -m pytest eval/tests -q
"""

from __future__ import annotations

import os

import pytest

SLOW_TESTS_ENV = "PII_BENCH_SLOW_TESTS"


def slow_tests_enabled() -> bool:
    """Whether `PII_BENCH_SLOW_TESTS` asks for the slow tests."""
    return os.environ.get(SLOW_TESTS_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if slow_tests_enabled():
        return
    skip = pytest.mark.skip(reason=f"slow test: set {SLOW_TESTS_ENV}=1 to run it")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
