"""Detector registry: one declarative entry per detector name.

The runner, the notebooks and the REST API all build detectors through
`build(name, ...)` — there is no per-name if-chain anywhere else. Heavy
libraries (torch, transformers, gliner, presidio, the OPF runtime) are imported
inside each factory, so importing this module is cheap and listing detectors
never loads a model.

Register your own detector:

    from opf_eval.detectors import registry
    from opf_eval.taxonomy import register_vocab

    register_vocab("mymodel", {"NAME": "PERSON", "SSN": "GOV_ID"}, kind="detector")

    @registry.detector("my_model", vocab="mymodel", requires=("transformers",))
    def _build(ctx):
        return MyDetector(device=ctx.device)

Or use one of the family helpers for common model shapes:
`register_gliner_model(...)` and `register_hf_token_classifier(...)`.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .. import taxonomy
from .base import Detector


@dataclass(frozen=True)
class BuildContext:
    """Everything a factory may use to configure its detector.

    dataset_labels: canonical labels the target dataset annotates. Detectors
        that take a per-call label set (GLiNER prompts, Skyflow entity types)
        restrict themselves to it. `None` means "no dataset" — build the
        detector with its full vocabulary (the API and ad-hoc notebook use).
    options: per-detector overrides, e.g. `{"threshold": 0.5}` for GLiNER.
    """

    dataset_labels: frozenset[str] | None = None
    device: str = "cpu"
    skyflow_entity_types: tuple[str, ...] | None = None
    opf_calibration_path: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)


Factory = Callable[[BuildContext], Detector]


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    vocab: str                       # taxonomy vocabulary its raw labels come from
    factory: Factory
    description: str = ""
    requires: tuple[str, ...] = ()   # importable modules the factory needs
    remote: bool = False             # network-backed: runner throttles / threads it
    in_api: bool = True              # expose in the REST API's default registry
    license: str = ""
    tags: tuple[str, ...] = ()

    def available(self) -> bool:
        """True if every module in `requires` is importable (no import done)."""
        return all(_importable(m) for m in self.requires)

    def missing(self) -> list[str]:
        return [m for m in self.requires if not _importable(m)]


def _importable(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


_REGISTRY: dict[str, DetectorSpec] = {}


def register_detector(spec: DetectorSpec, *, overwrite: bool = False) -> DetectorSpec:
    if spec.name in _REGISTRY and not overwrite:
        raise ValueError(f"detector {spec.name!r} already registered (pass overwrite=True)")
    taxonomy.register_detector_vocab(spec.name, spec.vocab)
    _REGISTRY[spec.name] = spec
    return spec


def detector(name: str, *, vocab: str, overwrite: bool = False, **spec_kwargs: Any):
    """Decorator form of `register_detector` for a factory function."""

    def deco(factory: Factory) -> Factory:
        register_detector(
            DetectorSpec(name=name, vocab=vocab, factory=factory, **spec_kwargs),
            overwrite=overwrite,
        )
        return factory

    return deco


def get(name: str) -> DetectorSpec:
    if name not in _REGISTRY:
        raise ValueError(f"unknown detector: {name!r}; registered: {names()}")
    return _REGISTRY[name]


def names() -> list[str]:
    return sorted(_REGISTRY)


def specs() -> list[DetectorSpec]:
    return [_REGISTRY[n] for n in names()]


def available_names() -> list[str]:
    return [s.name for s in specs() if s.available()]


def build(
    name: str,
    *,
    dataset_labels: Iterable[str] | None = None,
    device: str = "cpu",
    skyflow_entity_types: Iterable[str] | None = None,
    opf_calibration_path: str | None = None,
    options: Mapping[str, Any] | None = None,
) -> Detector:
    """Instantiate the named detector. See `BuildContext` for the arguments."""
    spec = get(name)
    missing = spec.missing()
    if missing:
        raise ImportError(
            f"detector {name!r} needs {missing} — install the detector's "
            f"dependencies (see docs/guides/detectors.md)"
        )
    ctx = BuildContext(
        dataset_labels=frozenset(dataset_labels) if dataset_labels is not None else None,
        device=device,
        skyflow_entity_types=tuple(skyflow_entity_types) if skyflow_entity_types else None,
        opf_calibration_path=opf_calibration_path,
        options=dict(options or {}),
    )
    det = spec.factory(ctx)
    if getattr(det, "name", None) != name:
        try:
            det.name = name  # type: ignore[misc]
        except AttributeError:
            pass
    return det


def describe() -> list[dict[str, Any]]:
    """One row per registered detector — handy for a notebook table."""
    return [
        {
            "name": s.name,
            "vocab": s.vocab,
            "available": s.available(),
            "remote": s.remote,
            "license": s.license,
            "description": s.description,
        }
        for s in specs()
    ]


# ------------------------------------------------------------ family helpers

# Shared GLiNER threshold — every gliner_* variant runs at it for cross-variant
# comparability. Gretel's recommended setting; trades some recall (vs Nvidia's
# card at 0.3 and GLiNER's library default of 0.5) for fewer false positives.
GLINER_THRESHOLD = 0.7


def register_gliner_model(
    name: str,
    model_id: str,
    *,
    vocab: str = "gliner",
    threshold: float = GLINER_THRESHOLD,
    dataset_aware: bool = True,
    description: str = "",
    license: str = "",
    overwrite: bool = False,
) -> DetectorSpec:
    """Register a GLiNER-library checkpoint as a detector.

    `vocab` is both the prompt list fed to the model and the map back to
    canonical labels. With `dataset_aware=True` prompts are restricted to
    the categories the dataset annotates (fewer spurious labels).
    """

    def factory(ctx: BuildContext) -> Detector:
        from .gliner import GLiNERDetector

        restrict = ctx.dataset_labels if dataset_aware else None
        return GLiNERDetector(
            model_name=ctx.options.get("model_name", model_id),
            threshold=float(ctx.options.get("threshold", threshold)),
            prompts=taxonomy.prompts_for(vocab, restrict),
            vocab=vocab,
            name=name,
            device=ctx.device,
        )

    return register_detector(
        DetectorSpec(
            name=name, vocab=vocab, factory=factory, description=description,
            requires=("gliner",), license=license, tags=("gliner", "local"),
        ),
        overwrite=overwrite,
    )


def register_hf_token_classifier(
    name: str,
    model_id: str,
    *,
    vocab: str,
    aggregation_strategy: str = "simple",
    description: str = "",
    license: str = "",
    overwrite: bool = False,
) -> DetectorSpec:
    """Register a Hugging Face `token-classification` checkpoint as a detector.

    Covers most fine-tuned BERT/DeBERTa/ModernBERT/XLM-R PII models: register
    a vocabulary for the model's label names, then this one call.
    """

    def factory(ctx: BuildContext) -> Detector:
        from .hf_token import HFTokenClassifierDetector

        return HFTokenClassifierDetector(
            model_name=ctx.options.get("model_name", model_id),
            vocab=vocab,
            name=name,
            aggregation_strategy=ctx.options.get("aggregation_strategy", aggregation_strategy),
            device=ctx.device,
        )

    return register_detector(
        DetectorSpec(
            name=name, vocab=vocab, factory=factory, description=description,
            requires=("transformers",), license=license, tags=("hf", "local"),
        ),
        overwrite=overwrite,
    )


from . import builtin as _builtin  # noqa: E402,F401  — registers the shipped detectors
