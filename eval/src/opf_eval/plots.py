"""Matplotlib charts for notebooks. Each function returns the Figure."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .scoring import Run, latencies


def per_label_bars(
    series: Mapping[str, Mapping[str, float]],
    labels: Sequence[str],
    *,
    title: str = "Per-category F1",
    ylabel: str = "SemEval Type F1 (any overlap + matching label)",
):
    """Grouped bars: one group per label, one bar per series (detector /
    composite). `series[name][label] -> F1`."""
    import matplotlib.pyplot as plt
    import numpy as np

    labels = list(labels)
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    fig, ax = plt.subplots(figsize=(max(12, 0.9 * len(labels)), 5))
    for i, (name, by_label) in enumerate(series.items()):
        ax.bar(x + i * width, [by_label.get(lbl, 0.0) for lbl in labels], width, label=name)
    ax.set_xticks(x + width * (len(series) - 1) / 2)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", ncol=2 if len(series) > 4 else 1, fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def headline_bars(rows: Sequence[Mapping], *, schema: str = "strict", title: str | None = None):
    """Horizontal bars of one schema's F1 per detector, best at the top.
    `rows` is `scoring.headline(...)` output."""
    import matplotlib.pyplot as plt

    rows = sorted((r for r in rows if r.get(schema) is not None), key=lambda r: r[schema])
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(rows) + 1.2))
    ax.barh([r["detector"] for r in rows], [r[schema] for r in rows])
    for i, r in enumerate(rows):
        ax.text(r[schema] + 0.01, i, f"{r[schema]:.3f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(f"{schema} F1")
    ax.set_title(title or f"Headline F1 ({schema})")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def latency_bars(run: Run, detectors: Sequence[str] | None = None, *, log: bool = True):
    """p50 / p95 / p99 per-call latency per detector."""
    import matplotlib.pyplot as plt
    import numpy as np

    names = list(detectors or run.detectors)
    stats = [latencies(run, d) for d in names]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(names)), 5))
    for offset, key in ((-0.25, "p50"), (0.0, "p95"), (0.25, "p99")):
        ax.bar(x + offset, [s[key] for s in stats], 0.25, label=key)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("latency per call (ms)")
    ax.set_title("Per-detector latency")
    if log:
        ax.set_yscale("log")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5, which="both")
    fig.tight_layout()
    return fig
