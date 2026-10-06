"""BM25 search over sanitized text: which sanitization modes keep search working?

A ~130-doc corpus with controlled PII overlap: a few curated "gold" docs the
queries should find, plus procedurally generated filler whose only job is to
outrank the gold docs under modes that can't tell PII *values* apart. Each
query carries its gold relevance set, so precision / recall per mode is
computed rather than asserted.

    corpus, queries = build_corpus()
    by_mode = sanitize_all(detector, corpus + [q.text for q in queries], modes, vault)
    ...

See notebooks/06_search_on_sanitized_data.ipynb for the walkthrough.
"""

from __future__ import annotations

import random
import re
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..transforms import TokenizerProtocol, render_modes

ALICE_DOCS = [
    "Alice (alice@example.com) drafted the proposal last quarter.",
    "The team lead is alice@example.com per the org chart.",
    "Bob (bob@example.com) emailed alice@example.com about the file.",
    "Forwarded from alice@example.com to ops@example.com on Friday.",
    "Compliance review: alice@example.com signed off on the audit.",
]

CHARLIE_DOCS = [
    "Charlie owns the budget account charlie@example.com.",
    "Don submitted feedback to charlie@example.com about scope.",
    "Quarterly review meeting includes charlie@example.com.",
]

_FILLER_EMAILS = [
    "alex@acme.io", "sam@acme.io", "jordan@acme.io", "taylor@acme.io",
    "morgan@acme.io", "riley@acme.io", "casey@acme.io", "cameron@acme.io",
    "quinn@acme.io", "avery@acme.io", "reese@acme.io", "hayden@acme.io",
    "skyler@acme.io", "rowan@acme.io", "emerson@acme.io", "drew@acme.io",
    "frankie@acme.io", "eli@acme.io", "devon@acme.io", "sage@acme.io",
    "ops@acme.io", "support@acme.io", "sales@acme.io", "legal@acme.io",
    "newsletter@acme.io",
]

_FILLER_PHONES = [f"+1-415-555-{n:04d}" for n in (
    100, 143, 199, 207, 313, 401, 442, 519, 567, 612,
    688, 712, 754, 798, 821, 855, 890, 911, 942, 977,
)]

_FILLER_ADDRESSES = [
    "123 Main St", "456 Pine Ave", "789 Oak Rd", "101 Elm Blvd",
    "202 Maple Dr", "303 Cedar Ln", "404 Birch Way", "505 Spruce Ct",
    "606 Walnut Pl", "707 Aspen Cir", "808 Willow Path", "909 Poplar Ter",
]

_MULTI_EMAIL_TEMPLATES = [
    "Thread among {a}, {b}, and {c} on the integration plan.",
    "Loop in {a} (cc {b}, {c}) on the status update.",
    "Email chain: {a} kicked it off, {b} and {c} replied later.",
    "{a} forwarded the report to {b} and copied {c}.",
    "{a}, {b}, and {c} signed off on the proposal yesterday.",
    "Meeting recap shared with {a}, {b}, and {c} after standup.",
    "Action item: {a} to coordinate with {b} and {c} on the rollout.",
]

_MULTI_PHONE_TEMPLATES = [
    "Voicemails from {p1}, {p2}, and {p3} logged at 11:02 AM.",
    "On-call rotation: {p1} primary, {p2} backup, {p3} escalation.",
    "Contact directory: {p1}, {p2}, and {p3} on file.",
    "Hotline routing: {p1} morning, {p2} afternoon, {p3} after-hours.",
]

_MULTI_ADDRESS_TEMPLATES = [
    "Office locations: {a1}, {a2}, and {a3} all currently active.",
    "Mail forwarding from {a1} to {a2}, then to {a3} next month.",
    "Site visits scheduled: {a1} Monday, {a2} Wednesday, {a3} Friday.",
]

_SINGLE_EMAIL_TEMPLATES = [
    "Welcome email sent to {a}.",
    "Account created for {a} this morning.",
    "Password reset link issued to {a}.",
    "Notification delivered to {a} successfully.",
    "Survey link sent to {a} yesterday.",
    "Profile updated for {a} after onboarding.",
    "Login alert dispatched to {a}.",
    "Reminder scheduled for {a} next Tuesday.",
]

_PII_FREE = [
    "Project status: design phase nearly complete.",
    "Quarterly OKRs aligned to the new product strategy.",
]


@dataclass(frozen=True)
class Query:
    text: str
    relevant: frozenset[int]  # corpus indices that should come back


def build_corpus(seed: int = 42) -> tuple[list[str], list[Query]]:
    """The demo corpus (gold docs first, so their indices are stable) and
    the two email queries with their gold relevance sets."""
    rng = random.Random(seed)
    multi_email = [
        rng.choice(_MULTI_EMAIL_TEMPLATES).format(a=a, b=b, c=c)
        for a, b, c in (rng.sample(_FILLER_EMAILS, 3) for _ in range(40))
    ]
    multi_phone = [
        rng.choice(_MULTI_PHONE_TEMPLATES).format(p1=p1, p2=p2, p3=p3)
        for p1, p2, p3 in (rng.sample(_FILLER_PHONES, 3) for _ in range(20))
    ]
    multi_address = [
        rng.choice(_MULTI_ADDRESS_TEMPLATES).format(a1=a1, a2=a2, a3=a3)
        for a1, a2, a3 in (rng.sample(_FILLER_ADDRESSES, 3) for _ in range(10))
    ]
    # Dilutes the [EMAIL_1] pool so label_number can't lean on length
    # normalisation to pull short alice/charlie docs into the top of the rank.
    single_email = [
        rng.choice(_SINGLE_EMAIL_TEMPLATES).format(a=rng.choice(_FILLER_EMAILS))
        for _ in range(50)
    ]
    corpus = (
        ALICE_DOCS            # 0..4
        + CHARLIE_DOCS        # 5..7
        + multi_email         # 8..47
        + multi_phone         # 48..67
        + multi_address       # 68..77
        + single_email        # 78..127
        + _PII_FREE           # 128..129
    )
    n_alice = len(ALICE_DOCS)
    queries = [
        Query("alice@example.com", frozenset(range(0, n_alice))),
        Query("charlie@example.com", frozenset(range(n_alice, n_alice + len(CHARLIE_DOCS)))),
    ]
    return corpus, queries


def sanitize_all(
    detector,
    texts: Sequence[str],
    modes: Iterable[str],
    token_vault: TokenizerProtocol | None = None,
) -> dict[str, list[str]]:
    """Detect once per text, render every mode. Adds a `plain` (unsanitized)
    entry as the no-privacy baseline. Documents and queries must go through
    the same detector and vault or tokens won't line up."""
    modes = [m for m in modes if m != "plain"]
    out: dict[str, list[str]] = {"plain": list(texts), **{m: [] for m in modes}}
    for text in texts:
        spans = detector.detect(text).get("spans") or []
        rendered = render_modes(text, spans, modes=modes, token_vault_client=token_vault)
        for m in modes:
            out[m].append(rendered[m])
    return out


def import_bm25s():
    """Import bm25s (installing it if needed) without letting it initialise CUDA.

    bm25s imports `numba` at load time as an optional speed-up. On Colab numba
    ships with a CUDA backend whose first import reserves GBs of VRAM — enough
    to OOM a detector loaded afterwards. Blocking numba makes bm25s take its
    pure-numpy path (still fast at this size). Side effect: `import numba`
    fails in this kernel afterwards unless you `del sys.modules["numba"]`.
    """
    if "bm25s" in sys.modules:
        return sys.modules["bm25s"]
    import importlib.util

    if importlib.util.find_spec("bm25s") is None:
        from ..nb import pip_install

        pip_install("bm25s")
    sys.modules.setdefault("numba", None)  # type: ignore[assignment]
    import bm25s

    return bm25s


# Keep bracketed sanitization tokens like [EMAIL_u8UBDWQ] / [EMAIL_1] whole,
# then fall back to alphanumeric words. Plain [EMAIL] (label mode) doesn't
# match the first branch and tokenizes to "email" — identically for docs and
# queries, so retrieval still lines up.
_BRACKET_OR_WORD = re.compile(r"\[[A-Z_]+_[A-Za-z0-9]+\]|[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [m.group().lower() for m in _BRACKET_OR_WORD.finditer(text)]


def retrieve(
    corpus_by_mode: dict[str, list[str]],
    queries_by_mode: dict[str, list[str]],
    *,
    k: int = 5,
) -> dict[tuple[int, str], list[int]]:
    """Top-k corpus indices per (query index, mode), one BM25 index per mode.

    Ranked by raw BM25 score with no positive-score filter: on a tiny corpus
    a term in more than half the docs gets a negative IDF, and filtering
    would drop the on-topic hits.
    """
    bm25s = import_bm25s()
    out: dict[tuple[int, str], list[int]] = {}
    for mode, docs in corpus_by_mode.items():
        retriever = bm25s.BM25()
        retriever.index([tokenize(d) for d in docs], show_progress=False)
        kk = min(k, len(docs))
        for qi, q in enumerate(queries_by_mode[mode]):
            ids, _scores = retriever.retrieve([tokenize(q)], k=kk, show_progress=False)
            out[(qi, mode)] = [int(i) for i in ids[0]]
    return out


def score(
    queries: Sequence[Query],
    queries_by_mode: dict[str, list[str]],
    retrievals: dict[tuple[int, str], list[int]],
) -> list[dict]:
    """One row per (query, mode): sanitized query, hits, precision, recall."""
    rows = []
    for qi, q in enumerate(queries):
        for mode in queries_by_mode:
            got = retrievals[(qi, mode)]
            tp = len(set(got) & q.relevant)
            rows.append({
                "query": qi,
                "mode": mode,
                "sanitized_query": queries_by_mode[mode][qi],
                "retrieved": got,
                "precision": tp / len(set(got)) if got else 0.0,
                "recall": tp / len(q.relevant) if q.relevant else 0.0,
            })
    return rows


def verdict(rows: Sequence[dict]) -> list[dict]:
    """Average precision / recall per mode across queries."""
    modes: dict[str, list[dict]] = {}
    for r in rows:
        modes.setdefault(r["mode"], []).append(r)
    return [
        {
            "mode": m,
            "avg_precision": sum(r["precision"] for r in rs) / len(rs),
            "avg_recall": sum(r["recall"] for r in rs) / len(rs),
        }
        for m, rs in modes.items()
    ]
