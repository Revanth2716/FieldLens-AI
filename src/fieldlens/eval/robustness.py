"""Reliability experiments: sensitivity to controlled image degradations.

Each sweep applies **one named transformation at a measured severity** to the
held-out test images, then reports macro F1, accuracy and mean confidence at each
level.

**How these numbers may be described.** They measure *sensitivity to that
specific synthetic perturbation*. They are **not** real-world accuracy under
"poor lighting" or "low-quality photos", and no result here may be restated that
way. Every sweep carries its transformation name and parameter values into the
report so the claim can never be separated from its definition.

Frozen weights only: nothing is retrained, so a sweep costs one forward pass per
level and is cheap enough to re-run.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from fieldlens.eval.metrics import accuracy, confusion_matrix, macro_scores

__all__ = [
    "DEGRADATIONS",
    "levels_for",
    "apply_brightness",
    "apply_resolution",
    "apply_jpeg",
    "apply_blur",
    "run_sweeps",
    "sweep_metadata",
]

#: Every sweep is a named transformation; the label is what gets reported.
DEGRADATIONS: dict[str, str] = {
    "brightness": "multiply pixel values by a gain factor",
    "resolution": "downscale to N px, then upscale back to the model input size",
    "jpeg": "re-encode as JPEG at a given quality factor",
    "blur": "Gaussian blur with standard deviation sigma",
}


def apply_brightness(image, gain: float):
    """Scale intensity by ``gain`` (1.0 is the unmodified baseline)."""
    from PIL import Image

    if gain == 1.0:
        return image
    array = np.asarray(image, dtype=np.float32) * float(gain)
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))


def apply_resolution(image, size: int):
    """Downscale to ``size`` px on the short edge, then upscale back.

    Simulates a low-resolution capture. The re-upscale keeps the tensor shape
    identical so only information loss varies between levels.
    """
    from PIL import Image

    width, height = image.size
    scale = size / min(width, height)
    if scale >= 1.0:
        return image
    small = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))), Image.BILINEAR
    )
    return small.resize((width, height), Image.BILINEAR)


def apply_jpeg(image, quality: int):
    """Re-encode through JPEG at ``quality`` (lower is more lossy)."""
    from PIL import Image

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    with Image.open(buffer) as handle:
        return handle.convert("RGB")


def apply_blur(image, sigma: float):
    """Gaussian blur; ``sigma`` 0 returns the input unchanged."""
    from PIL import ImageFilter

    if sigma <= 0:
        return image
    return image.filter(ImageFilter.GaussianBlur(radius=float(sigma)))


_APPLIERS: dict[str, Callable[[Any, float], Any]] = {
    "brightness": apply_brightness,
    "resolution": apply_resolution,
    "jpeg": apply_jpeg,
    "blur": apply_blur,
}


def sweep_metadata() -> dict[str, str]:
    """One-line description per sweep, for the report and the UI."""
    return dict(DEGRADATIONS)


@dataclass(frozen=True)
class _LevelResult:
    level: float
    accuracy: float
    macro_f1: float
    mean_confidence: float
    num_samples: int


#: Keys accepted inside a named degradation block, e.g. ``brightness: {gains: [...]}``.
_LEVEL_KEYS = ("gains", "sizes", "qualities", "sigmas", "levels", "values")


def levels_for(name: str, spec: Any) -> list[float]:
    """Extract the severity levels for one degradation.

    Accepts either a bare list (``brightness: [0.3, 1.0]``) or a named mapping
    (``brightness: {gains: [0.3, 1.0]}``), so the config can stay self-describing
    without the sweep having to guess.
    """
    if isinstance(spec, dict):
        for key in _LEVEL_KEYS:
            if key in spec:
                return [float(v) for v in spec[key]]
        if len(spec) == 1:
            return [float(v) for v in next(iter(spec.values()))]
        raise ValueError(
            f"degradation {name!r} must hold a list or a single named list, "
            f"got keys {sorted(spec)}"
        )
    if isinstance(spec, (list, tuple)):
        return [float(v) for v in spec]
    raise ValueError(
        f"degradation {name!r} must be a list or mapping of levels, "
        f"got {type(spec).__name__}"
    )


def _load_images(entries: Sequence, image_root) -> list:
    """Load PIL images once; sweeps reuse them across levels."""
    from PIL import Image

    images = []
    for entry in entries:
        path = image_root / entry.path
        if not path.exists():
            continue
        with Image.open(path) as handle:
            images.append((entry, handle.convert("RGB")))
    return images


def _score(model, transform, images: list, device: str, temperature: float, batch_size: int):
    """Score a list of (entry, PIL image) pairs, returning probabilities."""
    import torch

    probabilities = []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            chunk = images[start : start + batch_size]
            batch = torch.stack([transform(image) for _entry, image in chunk])
            logits = model(batch.to(device))
            probs = torch.softmax(logits / temperature, dim=-1)
            probabilities.append(probs.float().cpu().numpy())
    return np.concatenate(probabilities, axis=0) if probabilities else np.zeros((0, 1))


def run_sweeps(
    model,
    entries: Sequence,
    class_index: dict[str, int],
    transform,
    image_root,
    *,
    device: str,
    degradation: dict[str, Any],
    temperature: float = 1.0,
    num_classes: int,
    batch_size: int = 32,
    max_images: int | None = None,
) -> dict[str, Any]:
    """Run every configured sweep on frozen weights.

    Returns a report dict whose every entry states the transformation it used.
    """
    images = _load_images(entries, image_root)
    if max_images and len(images) > max_images:
        images = images[:max_images]
    if not images:
        return {"sweeps": {}, "note": "no images available to sweep"}

    labels = np.asarray([class_index[entry.class_name] for entry, _ in images])

    # The reference point for every "drop from baseline" figure is the model's
    # score on the *unperturbed* images, measured here once. Using this rather
    # than each sweep's mildest level keeps the baseline identical across sweeps
    # and well-defined even when no sweep level equals the identity (blur has no
    # sigma 0 point, jpeg no quality 100).
    baseline_probs = _score(model, transform, images, device, temperature, batch_size)
    baseline_predictions = baseline_probs.argmax(axis=1)
    baseline_macro = macro_scores(
        confusion_matrix(labels, baseline_predictions, num_classes), only_supported=True
    )
    baseline = {
        "accuracy": accuracy(labels, baseline_predictions),
        "macro_f1": baseline_macro["f1"],
        "mean_confidence": float(baseline_probs.max(axis=1).mean()),
        "num_samples": int(len(labels)),
    }

    results: dict[str, Any] = {
        "note": (
            "Each figure measures sensitivity to the named synthetic "
            "transformation at the listed severity. These are not field-condition "
            "accuracies and must not be restated as such."
        ),
        "num_images": len(images),
        "transformation_legend": sweep_metadata(),
        "baseline": baseline,
        "sweeps": {},
    }

    for name, spec in degradation.items():
        applier = _APPLIERS.get(name)
        if applier is None:
            results["sweeps"][name] = {"error": f"unknown degradation {name!r}"}
            continue
        try:
            levels = levels_for(name, spec)
        except ValueError as exc:
            results["sweeps"][name] = {"error": str(exc)}
            continue
        if not levels:
            results["sweeps"][name] = {"error": "no severity levels configured"}
            continue

        level_results: list[dict[str, Any]] = []
        for level in levels:
            transformed = [(entry, applier(image, level)) for entry, image in images]
            probs = _score(model, transform, transformed, device, temperature, batch_size)
            predictions = probs.argmax(axis=1)
            matrix = confusion_matrix(labels, predictions, num_classes)
            macro = macro_scores(matrix, only_supported=True)
            level_results.append(
                {
                    "level": float(level),
                    "accuracy": accuracy(labels, predictions),
                    "macro_f1": macro["f1"],
                    "macro_recall": macro["recall"],
                    "mean_confidence": float(probs.max(axis=1).mean()),
                    "num_samples": int(len(labels)),
                }
            )
            print(
                f"    {name}={level:<6} acc {level_results[-1]['accuracy']:.4f}  "
                f"macroF1 {level_results[-1]['macro_f1']:.4f}  "
                f"mean conf {level_results[-1]['mean_confidence']:.4f}",
                flush=True,
            )

        worst = min(level_results, key=lambda r: r["macro_f1"])
        results["sweeps"][name] = {
            "transformation": DEGRADATIONS[name],
            "levels": level_results,
            "baseline_macro_f1": baseline["macro_f1"],
            "worst_level": worst["level"],
            "worst_macro_f1": worst["macro_f1"],
            "macro_f1_drop_from_baseline": baseline["macro_f1"] - worst["macro_f1"],
        }

    return results


def figure_sweeps(report: dict[str, Any], path) -> bool:
    """Plot macro F1 (and mean confidence) against severity for each sweep."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - figures are optional
        return False

    sweeps = report.get("sweeps", {})
    if not sweeps:
        return False

    figure, axes = plt.subplots(1, len(sweeps), figsize=(4.2 * len(sweeps), 3.6))
    if len(sweeps) == 1:
        axes = [axes]

    for axis, (name, payload) in zip(axes, sweeps.items(), strict=True):
        levels = payload.get("levels", [])
        if not levels:
            continue
        x = [item["level"] for item in levels]
        axis.plot(x, [item["macro_f1"] for item in levels], marker="o", label="macro F1")
        axis.plot(
            x,
            [item["mean_confidence"] for item in levels],
            marker="s",
            linestyle="--",
            label="mean confidence",
        )
        axis.set_title(name)
        axis.set_xlabel("severity parameter")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("score")
    axes[0].legend(loc="lower left", fontsize=8)
    figure.suptitle("Sensitivity to synthetic degradations (held-out test images)")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return True
