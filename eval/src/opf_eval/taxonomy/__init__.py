"""Canonical label taxonomy: one vocabulary every detector and dataset maps into.

Three pieces:

- **Labels** (`labels.py`) — the canonical hierarchy. 15 coarse categories
  (the historical scoring set) plus fine sub-types under some of them.
- **Vocabularies** (`vocabs.py` + `register_vocab`) — per-source maps from a
  raw label (what a detector emits / a dataset annotates) to a canonical
  label. Built-ins cover every shipped detector and dataset; custom sources
  register their own at runtime without editing this package.
- **Detector -> vocabulary bindings** — which vocabulary a detector name
  speaks, used to work out what each detector can be fairly scored on.

Everything that existed before the hierarchy (`CANONICAL_MAP`,
`opf_to_canonical`, `dataset_canonicals`, `fair_labels`, ...) keeps its old
signature and returns **coarse** labels, so existing callers and published
results are unaffected. New code should prefer `to_canonical` and pass
`level=` explicitly.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from . import vocabs as _builtin_vocabs
from .labels import (
    ALL_LABELS,
    COARSE_LABELS,
    FINE_LABELS,
    LABELS,
    LEVELS,
    LabelInfo,
    check_level,
    children,
    is_known,
    parent,
)

_BIO_PREFIXES = frozenset("BIESLU")


def _strip_bio(label: str) -> str:
    """`B-EMAIL` -> `EMAIL`. Only single-letter BIO/BIOES/BILOU prefixes."""
    if len(label) > 2 and label[1] == "-" and label[0].upper() in _BIO_PREFIXES:
        return label[2:]
    return label


@dataclass
class Vocab:
    """A raw-label -> canonical-label mapping for one source."""

    key: str
    mapping: dict[str, str]
    kind: str = "dataset"  # "detector" | "dataset" | "generic"
    description: str = ""
    _folded: dict[str, str] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        bad = sorted({v for v in self.mapping.values() if not is_known(v)})
        if bad:
            raise ValueError(
                f"vocab {self.key!r} maps to unknown canonical label(s) {bad}; "
                f"valid labels: {', '.join(ALL_LABELS)}"
            )
        self._folded = {k.casefold(): v for k, v in self.mapping.items()}

    def lookup(self, raw: str) -> str | None:
        """Exact match, then case-insensitive, then with a BIO prefix stripped."""
        if not raw:
            return None
        hit = self.mapping.get(raw) or self._folded.get(raw.casefold())
        if hit:
            return hit
        bare = _strip_bio(raw)
        if bare != raw:
            return self.mapping.get(bare) or self._folded.get(bare.casefold())
        return None

    def labels(self, level: str = "fine") -> set[str]:
        check_level(level)
        if level == "coarse":
            return {parent(v) for v in self.mapping.values()}
        return set(self.mapping.values())


_VOCABS: dict[str, Vocab] = {}


def register_vocab(
    key: str,
    mapping: dict[str, str],
    *,
    kind: str = "dataset",
    description: str = "",
    overwrite: bool = False,
) -> Vocab:
    """Register a raw-label -> canonical-label vocabulary under `key`.

    Targets must be canonical labels (coarse or fine; see `ALL_LABELS`).
    Raw labels are matched exactly, then case-insensitively, then with a
    `B-`/`I-`-style prefix stripped — so one entry covers `EMAIL`, `email`
    and `B-EMAIL`. Raw labels you leave out are dropped at load/detect time.
    """
    if key in _VOCABS and not overwrite:
        raise ValueError(f"vocab {key!r} already registered (pass overwrite=True to replace)")
    vocab = Vocab(key, dict(mapping), kind=kind, description=description)
    _VOCABS[key] = vocab
    return vocab


def get_vocab(key: str) -> Vocab:
    if key not in _VOCABS:
        raise KeyError(f"unknown vocab {key!r}; registered: {sorted(_VOCABS)}")
    return _VOCABS[key]


def vocab_keys() -> list[str]:
    return sorted(_VOCABS)


for _key, (_mapping, _kind, _desc) in _builtin_vocabs.BUILTIN.items():
    register_vocab(_key, _mapping, kind=_kind, description=_desc)

# Identity vocabulary for data that is already labelled with canonical names
# (custom datasets written against this taxonomy).
CANONICAL_VOCAB = "canonical"
register_vocab(
    CANONICAL_VOCAB,
    {label: label for label in ALL_LABELS},
    kind="generic",
    description="Identity: raw labels are already canonical label names",
)


def to_canonical(vocab_key: str, raw: str, *, level: str = "fine") -> str | None:
    """Map one raw label to a canonical label at `level` (None if unmapped)."""
    hit = get_vocab(vocab_key).lookup(raw)
    if hit is None:
        return None
    return parent(hit) if check_level(level) == "coarse" else hit


def vocab_labels(vocab_key: str, level: str = "fine") -> set[str]:
    """Every canonical label a vocabulary can produce, at `level`."""
    return get_vocab(vocab_key).labels(level)


def project(label: str, level: str) -> str:
    """Project a canonical label onto `level` (fine labels are left alone
    at the fine level; everything rolls up to its parent at coarse)."""
    return parent(label) if check_level(level) == "coarse" else label


# ------------------------------------------------- detector <-> vocab bindings

_DETECTOR_VOCAB: dict[str, str] = {}


def register_detector_vocab(detector: str, vocab_key: str) -> None:
    """Bind a detector name to the vocabulary its raw labels come from."""
    get_vocab(vocab_key)  # fail fast on typos
    _DETECTOR_VOCAB[detector] = vocab_key


def detector_vocab(detector: str) -> str | None:
    """Vocabulary a detector speaks. Built-in detectors are bound by the
    detector registry; an unbound name that is itself a vocab key (e.g. a
    custom detector registered under its own vocab name) resolves to it."""
    if detector not in _DETECTOR_VOCAB:
        # Importing the registry binds every built-in detector. Local import:
        # the registry imports this module.
        from ..detectors import registry  # noqa: F401
    if detector in _DETECTOR_VOCAB:
        return _DETECTOR_VOCAB[detector]
    return detector if detector in _VOCABS else None


def _is_ensemble(detector: str) -> bool:
    return detector.startswith("ensemble_")


def detector_supported_canonicals(detector: str, level: str = "coarse") -> set[str]:
    """Canonical labels this detector can produce.

    Ensembles (any name starting with `ensemble_`) report every label: by
    construction they can emit any label a constituent emits.
    """
    check_level(level)
    if _is_ensemble(detector):
        return set(COARSE_LABELS) if level == "coarse" else set(ALL_LABELS)
    vk = detector_vocab(detector)
    return vocab_labels(vk, level) if vk else set()


def dataset_canonicals(vocab_key: str, level: str = "coarse") -> set[str]:
    """Canonical labels a dataset vocabulary annotates."""
    return vocab_labels(vocab_key, level)


def fair_labels(detector: str, vocab_key: str, level: str = "coarse") -> set[str]:
    """Fair-view label set: (dataset annotates) ∩ (detector supports)."""
    if level == "fine":
        det = detector_supported_canonicals(detector, "fine")
        ds = dataset_canonicals(vocab_key, "fine")
        proj = shared_projector(det, ds)
        det_p = {proj(x) for x in det}
        allow = det_p & {proj(x) for x in ds}
        # Keep the coarse parent of every fine-scored category the detector
        # can emit (mirrors `scoring.scope_for`).
        return allow | ({parent(x) for x in allow if parent(x) != x} & det_p)
    return detector_supported_canonicals(detector, level) & dataset_canonicals(vocab_key, level)


def shared_projector(
    detector_labels: Iterable[str], dataset_labels: Iterable[str]
) -> Callable[[str], str]:
    """Label projection for fine-level scoring of one detector on one dataset.

    A coarse category is scored at fine granularity only when *both* sides
    have fine labels in it; otherwise it's rolled up to the coarse parent.
    Without this, a detector that only emits coarse `PERSON` would be scored
    against `GIVEN_NAME`/`FAMILY_NAME` gold and look like it never finds a
    name — penalising granularity rather than detection.
    """
    det_fine = {parent(x) for x in detector_labels if parent(x) != x}
    ds_fine = {parent(x) for x in dataset_labels if parent(x) != x}
    shared = det_fine & ds_fine

    def proj(label: str) -> str:
        p = parent(label)
        return label if p in shared else p

    return proj


# ------------------------------------------------------ back-compat surface
# Pre-hierarchy API. Every function here returns *coarse* labels.

CANONICAL_LABELS: tuple[str, ...] = COARSE_LABELS


def _coarse_view() -> dict[str, dict[str, tuple[str, ...]]]:
    out: dict[str, dict[str, tuple[str, ...]]] = {c: {} for c in COARSE_LABELS}
    for key, (mapping, _kind, _desc) in _builtin_vocabs.BUILTIN.items():
        for coarse in COARSE_LABELS:
            out[coarse][key] = tuple(r for r, c in mapping.items() if parent(c) == coarse)
    return out


# canonical (coarse) -> source -> raw labels. Read-only view of the built-in
# vocabularies, kept for callers written against the old flat table.
CANONICAL_MAP: dict[str, dict[str, tuple[str, ...]]] = _coarse_view()

# The 8 categories OPF supports natively.
OPF_CANONICAL_LABELS: tuple[str, ...] = tuple(
    c for c in COARSE_LABELS if c in vocab_labels("opf", "coarse")
)


def _coarse(vocab_key: str) -> Callable[[str], str | None]:
    def fn(label: str) -> str | None:
        return to_canonical(vocab_key, label, level="coarse")

    fn.__name__ = f"{vocab_key}_to_canonical"
    fn.__doc__ = f"Map a raw `{vocab_key}` label to its coarse canonical label."
    return fn


opf_to_canonical = _coarse("opf")
skyflow_to_canonical = _coarse("skyflow")
presidio_to_canonical = _coarse("presidio")
gliner_to_canonical = _coarse("gliner")
gretel_to_canonical = _coarse("gretel")
openmed_to_canonical = _coarse("openmed")
pii300k_to_canonical = _coarse("pii300k")


def dataset_to_canonical(vocab_key: str, label: str) -> str | None:
    """Coarse lookup in a dataset vocabulary. Strips a BIO prefix if present."""
    return to_canonical(vocab_key, label, level="coarse")


def prompts_for(vocab_key: str, canonicals: Iterable[str] | None = None) -> list[str]:
    """Raw labels of a vocabulary, in order, optionally restricted to those
    whose coarse category is covered by `canonicals` (coarse or fine).
    Used as the prompt list for prompt-driven detectors (GLiNER)."""
    targets = None if canonicals is None else {parent(c) for c in canonicals}
    return [
        raw for raw, canon in get_vocab(vocab_key).mapping.items()
        if targets is None or parent(canon) in targets
    ]


def gliner_prompts(canonicals: Iterable[str] | None = None) -> list[str]:
    """GLiNER prompt strings, optionally restricted to prompts whose coarse
    category is covered by `canonicals` (coarse or fine labels)."""
    return prompts_for("gliner", canonicals)


def gretel_prompts(canonicals: Iterable[str] | None = None) -> list[str]:
    """Gretel snake_case labels, optionally restricted like `gliner_prompts`."""
    return prompts_for("gretel", canonicals)


def canonical_to_skyflow_request_types(canonicals: Iterable[str]) -> list[str]:
    """Skyflow request-side entity_type strings (lowercase) covering the given
    canonical labels' coarse categories, in the order given.

    The Detect API request enum is the lowercase form of the response
    entity_type values: response `EMAIL_ADDRESS` -> request `email_address`.
    """
    order: list[str] = []
    for c in canonicals:
        p = parent(c)
        if p not in order:
            order.append(p)
    mapping = get_vocab("skyflow").mapping
    out: list[str] = []
    for coarse in order:
        for raw, canon in mapping.items():
            if parent(canon) == coarse and raw.lower() not in out:
                out.append(raw.lower())
    return out


__all__ = [
    "ALL_LABELS",
    "CANONICAL_LABELS",
    "CANONICAL_MAP",
    "CANONICAL_VOCAB",
    "COARSE_LABELS",
    "FINE_LABELS",
    "LABELS",
    "LEVELS",
    "LabelInfo",
    "OPF_CANONICAL_LABELS",
    "Vocab",
    "canonical_to_skyflow_request_types",
    "check_level",
    "children",
    "dataset_canonicals",
    "dataset_to_canonical",
    "detector_supported_canonicals",
    "detector_vocab",
    "fair_labels",
    "get_vocab",
    "gliner_prompts",
    "gliner_to_canonical",
    "gretel_prompts",
    "gretel_to_canonical",
    "is_known",
    "openmed_to_canonical",
    "opf_to_canonical",
    "parent",
    "pii300k_to_canonical",
    "presidio_to_canonical",
    "project",
    "prompts_for",
    "register_detector_vocab",
    "register_vocab",
    "shared_projector",
    "skyflow_to_canonical",
    "to_canonical",
    "vocab_keys",
    "vocab_labels",
]
