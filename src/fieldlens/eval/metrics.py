"""Metrics implemented directly in NumPy.

Deliberately dependency-free so training and evaluation do not require
scikit-learn. Macro-averaged metrics are the headline numbers here because the
rare classes (``Potato___healthy`` has 152 images) would otherwise be hidden by
plain accuracy.

Every function is a pure function of arrays, which is what keeps the evaluation
layer incapable of retraining or quietly adjusting the test set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

__all__ = [
    "confusion_matrix",
    "accuracy",
    "per_class_scores",
    "macro_scores",
    "top_k_accuracy",
    "expected_calibration_error",
    "brier_score",
    "reliability_bins",
    "CalibrationResult",
    "selective_curve",
]


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int
) -> np.ndarray:
    """Counts matrix: rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {y_true.shape} "
            f"and {y_pred.shape}"
        )
    if y_true.size and (y_true.min() < 0 or y_true.max() >= num_classes):
        raise ValueError(
            f"labels out of range for {num_classes} classes: "
            f"[{y_true.min()}, {y_true.max()}]"
        )
    flat_index = y_true * num_classes + y_pred
    counts = np.bincount(flat_index, minlength=num_classes * num_classes)
    return counts.reshape(num_classes, num_classes)


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Fraction of exactly correct predictions."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return float("nan")
    return float((y_true == y_pred).mean())


def per_class_scores(matrix: np.ndarray) -> list[dict[str, float]]:
    """Precision, recall, F1 and support per class, from a confusion matrix.

    Zero-division cases (a class with no predictions, or no support) report 0.0
    rather than NaN, so macro averages stay meaningful.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    num_classes = matrix.shape[0]
    true_positive = np.diag(matrix)
    predicted_total = matrix.sum(axis=0)
    actual_total = matrix.sum(axis=1)

    scores: list[dict[str, float]] = []
    for index in range(num_classes):
        tp = float(true_positive[index])
        precision = tp / float(predicted_total[index]) if predicted_total[index] else 0.0
        recall = tp / float(actual_total[index]) if actual_total[index] else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        scores.append(
            {
                "index": float(index),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": float(actual_total[index]),
            }
        )
    return scores


def macro_scores(
    matrix: np.ndarray, *, only_supported: bool = True
) -> dict[str, float]:
    """Macro-averaged precision, recall and F1.

    ``only_supported`` averages over classes that actually appear in the data, so
    a class absent from a small test split cannot drag the macro average to zero.
    """
    scores = per_class_scores(matrix)
    if only_supported:
        scores = [s for s in scores if s["support"] > 0]
    if not scores:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    return {
        "precision": float(np.mean([s["precision"] for s in scores])),
        "recall": float(np.mean([s["recall"] for s in scores])),
        "f1": float(np.mean([s["f1"] for s in scores])),
    }


def top_k_accuracy(
    probabilities: np.ndarray, y_true: Sequence[int], k: int
) -> float:
    """Fraction of samples whose true label is within the top ``k`` scores."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    if probabilities.size == 0:
        return float("nan")
    k = min(k, probabilities.shape[1])
    top_k = np.argsort(-probabilities, axis=1)[:, :k]
    return float((top_k == y_true[:, None]).any(axis=1).mean())


def brier_score(probabilities: np.ndarray, y_true: Sequence[int]) -> float:
    """Multi-class Brier score (mean squared error of the probability vector)."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    if probabilities.size == 0:
        return float("nan")
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(y_true.size), y_true] = 1.0
    return float(((probabilities - one_hot) ** 2).sum(axis=1).mean())


def reliability_bins(
    probabilities: np.ndarray, y_true: Sequence[int], n_bins: int = 15
) -> dict[str, list[float]]:
    """Per-bin confidence/accuracy/count data for a reliability diagram."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    if probabilities.size == 0:
        return {"bin_lower": [], "bin_upper": [], "confidence": [], "accuracy": [], "count": []}

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: dict[str, list[float]] = {
        "bin_lower": [],
        "bin_upper": [],
        "confidence": [],
        "accuracy": [],
        "count": [],
    }
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        # First bin is closed on the left so a confidence of exactly 0 is counted.
        mask = (confidences >= lower) & (
            confidences <= upper if index == n_bins - 1 else confidences < upper
        )
        count = int(mask.sum())
        out["bin_lower"].append(float(lower))
        out["bin_upper"].append(float(upper))
        out["count"].append(float(count))
        out["confidence"].append(float(confidences[mask].mean()) if count else 0.0)
        out["accuracy"].append(float(correct[mask].mean()) if count else 0.0)
    return out


@dataclass(frozen=True)
class CalibrationResult:
    """Expected and maximum calibration error, plus the raw bins."""

    ece: float
    mce: float
    bins: dict[str, list[float]]
    num_samples: int

    def to_json(self) -> dict:
        return {
            "ece": self.ece,
            "mce": self.mce,
            "num_samples": self.num_samples,
            "bins": self.bins,
        }


def expected_calibration_error(
    probabilities: np.ndarray, y_true: Sequence[int], n_bins: int = 15
) -> CalibrationResult:
    """Expected and maximum calibration error over equal-width confidence bins.

    ECE is the count-weighted mean gap between confidence and accuracy. It is the
    headline calibration number for this project: it measures whether a stated
    confidence of 0.8 actually corresponds to being right about 80% of the time.
    """
    bins = reliability_bins(probabilities, y_true, n_bins=n_bins)
    counts = np.asarray(bins["count"], dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return CalibrationResult(float("nan"), float("nan"), bins, 0)

    confidence = np.asarray(bins["confidence"], dtype=np.float64)
    accuracy = np.asarray(bins["accuracy"], dtype=np.float64)
    gaps = np.abs(confidence - accuracy)
    weights = counts / total

    ece = float((weights * gaps).sum())
    mce = float(gaps[counts > 0].max()) if (counts > 0).any() else float("nan")
    return CalibrationResult(ece, mce, bins, int(total))


def selective_curve(
    probabilities: np.ndarray,
    y_true: Sequence[int],
    *,
    coverages: Sequence[float] | None = None,
    steps: int = 50,
) -> list[dict[str, float]]:
    """Risk-coverage data for selective prediction.

    At each coverage level, keep only the most confident fraction of samples and
    report the accuracy achieved on them. This is the honest way to present a
    confidence threshold: accuracy *at a stated coverage*, not a single number.
    """
    probabilities = np.asarray(probabilities, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=np.int64)
    if probabilities.size == 0:
        return []

    confidences = probabilities.max(axis=1)
    correct = (probabilities.argmax(axis=1) == y_true).astype(np.float64)
    order = np.argsort(-confidences)

    if coverages is None:
        coverages = np.linspace(1.0 / steps, 1.0, steps)

    curve: list[dict[str, float]] = []
    for coverage in coverages:
        keep = max(1, int(round(coverage * len(order))))
        selected = order[:keep]
        curve.append(
            {
                "coverage": float(keep / len(order)),
                "accuracy": float(correct[selected].mean()),
                "mean_confidence": float(confidences[selected].mean()),
                "threshold": float(confidences[selected].min()),
            }
        )
    return curve


def threshold_for_coverage(
    probabilities: np.ndarray, target_coverage: float
) -> float:
    """Confidence threshold that retains roughly ``target_coverage`` of samples."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.size == 0:
        return 0.0
    confidences = np.sort(probabilities.max(axis=1))[::-1]
    index = min(len(confidences) - 1, max(0, int(round(target_coverage * len(confidences))) - 1))
    return float(confidences[index])
