"""Estimate precision and recall on data without gold labels (plan 13, Parts 5 and 6).

Two reviewers judge the spans the detectors found in a run directory:

1. `review.llm` asks an LLM for a verdict on every pooled span and for the
   personal data no detector found. It gives estimated precision and pooled
   recall (an upper bound) per detector.
2. `review.classifier` asks a typed decision model (`review.clef` runs
   Cloudflare's Clef-flash locally) yes/no and choice questions about each
   span and about each text segment with the detector's spans masked. It
   gives estimated precision and a residual-PII rate per detector.

Both judge each unique span once (`review.pool.pooled_spans`) and credit
the verdict to every detector that found it. Both write a JSONL file plus a
`.meta.json` sidecar into the run directory, so the notebook can rerun the
summaries without calling a model again.

    from opf_eval import llm, review
    from opf_eval.review import classifier, clef

    path = review.llm.review_run(run_dir, llm.make_client("anthropic"))
    review.llm.summarize(run_dir, path)

    path = classifier.review_run(run_dir, clef.ClefBackend())
    classifier.summarize(run_dir, path)

Importing this package is cheap: torch and transformers load only when a
`ClefBackend` is built, and no LLM SDK is imported here.
"""

from __future__ import annotations

from . import classifier, clef, llm, pool
from .classifier import DecisionBackend, StubBackend
from .pool import pooled_spans

__all__ = [
    "DecisionBackend",
    "StubBackend",
    "classifier",
    "clef",
    "llm",
    "pool",
    "pooled_spans",
]
