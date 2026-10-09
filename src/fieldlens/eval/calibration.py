"""Temperature scaling.

A single scalar ``T`` is fitted on the **validation** split by minimising negative
log-likelihood, then applied unchanged to test. Fitting on test would make every
reported calibration number meaningless, so the split used is recorded in the
report and the function refuses to run without an explicit label.

Temperature scaling can only soften or sharpen the existing distribution; it
cannot repair a model that is systematically wrong. If ECE does not improve, that
is a legitimate finding to publish rather than something to tune around.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["TemperatureFit", "fit_temperature", "apply_temperature", "probabilities"]

#: T is clamped away from zero; below this the logits overflow and the fit is
#: meaningless anyway.
MIN_TEMPERATURE = 0.05
MAX_TEMPERATURE = 100.0


@dataclass(frozen=True)
class TemperatureFit:
    """Result of fitting a temperature, with the objective before and after."""

    temperature: float
    nll_before: float
    nll_after: float
    num_samples: int
    split: str

    @property
    def improved(self) -> bool:
        return self.nll_after <= self.nll_before

    def to_json(self) -> dict:
        return {
            "temperature": self.temperature,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
            "nll_improvement": self.nll_before - self.nll_after,
            "improved": self.improved,
            "num_samples": self.num_samples,
            "fitted_on": self.split,
        }


def _to_tensors(logits: np.ndarray, labels: np.ndarray):
    import torch

    return (
        torch.tensor(np.asarray(logits), dtype=torch.float32),
        torch.tensor(np.asarray(labels), dtype=torch.long),
    )


def _nll(logits, labels, temperature: float) -> float:
    import torch

    scaled = logits / max(temperature, MIN_TEMPERATURE)
    return float(torch.nn.functional.cross_entropy(scaled, labels))


def fit_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    split: str,
    max_iter: int = 100,
) -> TemperatureFit:
    """Fit a single temperature by minimising NLL on the given split.

    ``split`` is required and recorded so a caller cannot silently fit on test.
    """
    if split not in {"val", "validation", "train"}:
        raise ValueError(
            f"temperature must be fitted on a non-test split, got {split!r}"
        )
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    if logits.size == 0:
        raise ValueError("cannot fit a temperature on an empty set")
    if logits.shape[0] != labels.shape[0]:
        raise ValueError(
            f"logits and labels disagree: {logits.shape[0]} vs {labels.shape[0]}"
        )

    import torch

    logits_t, labels_t = _to_tensors(logits, labels)
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=max_iter)

    def closure():
        optimizer.zero_grad()
        scaled = logits_t / temperature.clamp(min=MIN_TEMPERATURE, max=MAX_TEMPERATURE)
        loss = torch.nn.functional.cross_entropy(scaled, labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    fitted = float(
        temperature.detach().clamp(min=MIN_TEMPERATURE, max=MAX_TEMPERATURE).item()
    )

    return TemperatureFit(
        temperature=fitted,
        nll_before=_nll(logits_t, labels_t, 1.0),
        nll_after=_nll(logits_t, labels_t, fitted),
        num_samples=int(labels.shape[0]),
        split=split,
    )


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Scale logits by ``1 / temperature`` (no softmax applied)."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return np.asarray(logits, dtype=np.float64) / float(temperature)


def probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Softmax of scaled logits, computed in a numerically stable way."""
    scaled = apply_temperature(logits, temperature)
    shifted = scaled - scaled.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)
