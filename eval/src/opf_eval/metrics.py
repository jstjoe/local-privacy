"""Span scoring: exact-match + partial-overlap (>=0.5 IoU), per canonical label."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Iterable

from .detectors.base import Span


@dataclass
class PRF:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class ScoreReport:
    overall_exact: PRF = field(default_factory=PRF)
    overall_partial: PRF = field(default_factory=PRF)
    per_label_exact: dict[str, PRF] = field(default_factory=lambda: defaultdict(PRF))
    per_label_partial: dict[str, PRF] = field(default_factory=lambda: defaultdict(PRF))


def _iou(a: Span | dict, b: Span | dict) -> float:
    inter_lo = max(a["start"], b["start"])
    inter_hi = min(a["end"], b["end"])
    inter = max(0, inter_hi - inter_lo)
    union = (a["end"] - a["start"]) + (b["end"] - b["start"]) - inter
    return inter / union if union > 0 else 0.0


def _match(
    pred: list[dict],
    gold: list[dict],
    *,
    iou_threshold: float,
) -> tuple[int, int, int, dict[tuple[str, bool], int]]:
    """Greedy 1:1 match by IoU within same label. Returns (tp, fp, fn, per_label_counts)."""
    used_gold: set[int] = set()
    tp = 0
    per_label: dict[tuple[str, bool], int] = defaultdict(int)  # (label, kind=tp?) -> count, kind: True=tp,False=fp
    for p in pred:
        best_idx = -1
        best_iou = 0.0
        for j, g in enumerate(gold):
            if j in used_gold or g["label"] != p["label"]:
                continue
            score = 1.0 if iou_threshold >= 1.0 else _iou(p, g)
            if iou_threshold >= 1.0:
                ok = p["start"] == g["start"] and p["end"] == g["end"]
                if ok and score > best_iou:
                    best_iou, best_idx = 1.0, j
            elif score >= iou_threshold and score > best_iou:
                best_iou, best_idx = score, j
        if best_idx >= 0:
            used_gold.add(best_idx)
            tp += 1
            per_label[(p["label"], True)] += 1
        else:
            per_label[(p["label"], False)] += 1
    fp = len(pred) - tp
    fn = len(gold) - len(used_gold)
    for j, g in enumerate(gold):
        if j not in used_gold:
            per_label[(g["label"], False)] += 0  # ensure label appears
    return tp, fp, fn, per_label


def score(
    pairs: Iterable[tuple[list[dict], list[dict]]],
) -> ScoreReport:
    """pairs: iterable of (predicted_spans, gold_spans) per example."""
    rep = ScoreReport()
    for pred, gold in pairs:
        # exact
        tp, fp, fn, _ = _match(pred, gold, iou_threshold=1.0)
        rep.overall_exact.tp += tp
        rep.overall_exact.fp += fp
        rep.overall_exact.fn += fn
        # partial
        tp_p, fp_p, fn_p, _ = _match(pred, gold, iou_threshold=0.5)
        rep.overall_partial.tp += tp_p
        rep.overall_partial.fp += fp_p
        rep.overall_partial.fn += fn_p
        # per-label (partial)
        _per_label_update(rep.per_label_exact, pred, gold, iou_threshold=1.0)
        _per_label_update(rep.per_label_partial, pred, gold, iou_threshold=0.5)
    return rep


def _per_label_update(
    table: dict[str, PRF],
    pred: list[dict],
    gold: list[dict],
    *,
    iou_threshold: float,
) -> None:
    by_label_pred: dict[str, list[dict]] = defaultdict(list)
    by_label_gold: dict[str, list[dict]] = defaultdict(list)
    for p in pred:
        by_label_pred[p["label"]].append(p)
    for g in gold:
        by_label_gold[g["label"]].append(g)
    labels = set(by_label_pred) | set(by_label_gold)
    for lbl in labels:
        tp, fp, fn, _ = _match(by_label_pred.get(lbl, []), by_label_gold.get(lbl, []), iou_threshold=iou_threshold)
        bucket = table[lbl]
        bucket.tp += tp
        bucket.fp += fp
        bucket.fn += fn


def latency_summary(latencies: list[float]) -> dict[str, float]:
    if not latencies:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "n": 0}
    s = sorted(latencies)
    def pct(q: float) -> float:
        idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
        return s[idx]
    return {
        "p50": median(s),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "n": len(s),
    }
