"""SemEval 2013 9.1 NER scoring via the `nervaluate` package.

Schema names from nervaluate map to SemEval terminology as follows:

    nervaluate    SemEval     What it measures
    ----------    -------     ----------------
    strict        Strict      exact boundary + matching label
    exact         Exact       exact boundary, label ignored
    partial       Partial     any overlap, label ignored
    ent_type      Type        any overlap + matching label

Each schema has 5 error counters (per the SemEval spec):

    correct (COR)     - prediction matches gold under that schema
    incorrect (INC)   - prediction overlaps gold but disagrees (label or boundary)
    partial (PAR)     - boundary partially overlaps (Partial schema only)
    missed (MIS)      - gold span not captured by any prediction
    spurious (SPU)    - prediction with no corresponding gold span
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from nervaluate import Evaluator


SCHEMAS: tuple[str, ...] = ("strict", "exact", "partial", "ent_type")
ERROR_KEYS: tuple[str, ...] = ("correct", "incorrect", "partial", "missed", "spurious")


@dataclass(frozen=True)
class SemEvalResult:
    """Structured per-detector SemEval scoring."""

    detector: str
    n_examples: int
    # by_schema[schema] -> {precision, recall, f1, correct, incorrect, partial, missed, spurious, possible, actual}
    by_schema: dict[str, dict[str, float | int]]
    # by_label[tag][schema] -> same shape as above
    by_label: dict[str, dict[str, dict[str, float | int]]]


def _to_nervaluate_spans(spans: Iterable[dict]) -> list[dict]:
    """Project our Span dicts into nervaluate's expected {label, start, end}."""
    return [{"label": s["label"], "start": int(s["start"]), "end": int(s["end"])} for s in spans]


def score(
    *,
    detector: str,
    pairs: list[tuple[list[dict], list[dict]]],
    tags: list[str],
) -> SemEvalResult:
    """Score one detector against gold via nervaluate.

    Args:
        detector: name to attach to the result (for the report).
        pairs: list of (predicted_spans, gold_spans) per example. Spans are our
            existing Span dicts (with `label`, `start`, `end`, ...) — anything
            else is ignored.
        tags: canonical labels to score (everything outside this list is treated
            as non-entity by nervaluate).

    Returns: a SemEvalResult with overall (`by_schema`) and per-tag (`by_label`)
    metrics for all four SemEval schemas.
    """
    true = [_to_nervaluate_spans(gold) for _, gold in pairs]
    pred = [_to_nervaluate_spans(p) for p, _ in pairs]
    ev = Evaluator(true, pred, tags=tags)
    overall, per_tag, _overall_idx, _per_tag_idx = ev.evaluate()
    return SemEvalResult(
        detector=detector,
        n_examples=len(pairs),
        by_schema={s: dict(overall[s]) for s in SCHEMAS},
        by_label={
            tag: {s: dict(per_tag[tag][s]) for s in SCHEMAS} for tag in per_tag
        },
    )
