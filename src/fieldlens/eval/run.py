"""Full evaluation pipeline.

Produces the genuine, measured numbers this project is built around:

* accuracy, macro precision/recall/F1, top-3 accuracy, per-class and per-crop tables
* confusion matrix (counts and row-normalised)
* temperature scaling fitted on **validation only**, with ECE and Brier score
  reported before and after, plus a reliability diagram
* risk-coverage curve with thresholds selected on validation at target coverage

Everything is written to ``artifacts/reports/evaluation.json`` and, when
matplotlib is available, to figures. The number of images actually evaluated is
recorded, because a capped acquisition is not the whole test split and pretending
otherwise would overstate the result.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from fieldlens.config import DatasetConfig, EvalConfig
from fieldlens.data.leakage import audit_manifest
from fieldlens.data.split import Manifest, default_manifest_path
from fieldlens.data.transforms import build_eval_transform
from fieldlens.eval.calibration import fit_temperature, probabilities
from fieldlens.eval.metrics import (
    accuracy,
    brier_score,
    confusion_matrix,
    expected_calibration_error,
    macro_scores,
    per_class_scores,
    selective_curve,
    threshold_for_coverage,
    top_k_accuracy,
)
from fieldlens.paths import (
    FIGURE_DIR,
    IMAGE_DIR,
    PROMOTED_CHECKPOINT,
    REPORT_DIR,
    ensure_dirs,
)
from fieldlens.training.dataset import LeafImageDataset

__all__ = ["run_evaluation", "collect_logits"]


def collect_logits(
    model,
    entries,
    class_index: dict[str, int],
    transform,
    image_root: Path,
    *,
    device: str,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Run the model over entries, returning logits, labels and the count used.

    Entries whose image is absent are skipped and reported, so a partially
    acquired dataset cannot silently shrink the evaluation without saying so.
    """
    import torch

    present = [e for e in entries if (image_root / e.path).exists()]
    skipped = len(entries) - len(present)
    if not present:
        return np.zeros((0, len(class_index))), np.zeros((0,), dtype=int), skipped

    dataset = LeafImageDataset(present, class_index, transform, image_root)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    all_logits, all_labels = [], []
    model.eval()
    with torch.inference_mode():
        for inputs, labels in loader:
            all_logits.append(model(inputs.to(device)).detach().float().cpu())
            all_labels.append(labels)
    return (
        torch.cat(all_logits).numpy(),
        torch.cat(all_labels).numpy(),
        skipped,
    )


def _per_crop_table(
    labels: np.ndarray, predictions: np.ndarray, class_names: tuple[str, ...]
) -> dict[str, dict[str, float]]:
    """Macro metrics aggregated by crop, which is the axis reviewers question."""
    crops: dict[str, list[int]] = {}
    for index, name in enumerate(class_names):
        crops.setdefault(name.split("___", 1)[0], []).append(index)

    overall = confusion_matrix(labels, predictions, len(class_names))
    table: dict[str, dict[str, float]] = {}
    for crop, indices in sorted(crops.items()):
        sub = overall[np.ix_(indices, indices)]
        macro = macro_scores(sub, only_supported=True)
        support = int(sub.sum())
        correct = float(np.trace(sub))
        table[crop] = {
            "accuracy": (correct / support) if support else float("nan"),
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "macro_f1": macro["f1"],
            "support": support,
            "num_classes": len(indices),
        }
    return table


def _figure_confusion(matrix: np.ndarray, class_names: tuple[str, ...], path: Path) -> bool:
    """Row-normalised confusion matrix figure. Returns False when unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001 - figures are optional
        return False

    normalised = matrix.astype(float)
    row_sums = normalised.sum(axis=1, keepdims=True)
    normalised = np.divide(normalised, row_sums, out=np.zeros_like(normalised), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    short = [n.replace("___", "\n") for n in class_names]
    ax.set_xticklabels(short, rotation=90, fontsize=6)
    ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (row-normalised)")
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _figure_reliability(bins: dict, ece: float, path: Path) -> bool:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False

    confidence = np.asarray(bins["confidence"], dtype=float)
    acc = np.asarray(bins["accuracy"], dtype=float)
    counts = np.asarray(bins["count"], dtype=float)
    mask = counts > 0

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="perfect calibration")
    ax.plot(confidence[mask], acc[mask], marker="o", label="model")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("observed accuracy")
    ax.set_title(f"Reliability diagram (ECE {ece:.3f})")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _store_temperature(checkpoint_path: Path, temperature: float) -> bool:
    """Write the fitted temperature into a checkpoint so serving matches."""
    import torch

    if not checkpoint_path.exists():
        return False
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata")
    if not metadata:
        return False
    metadata["temperature"] = float(temperature)
    metadata.setdefault("calibration", {})["source"] = "validation split"
    torch.save(payload, checkpoint_path)
    return True


def run_evaluation(
    *,
    checkpoint: Path | None = None,
    image_root: Path | None = None,
    batch_size: int = 32,
    skip_sweeps: bool = False,
) -> int:
    """Evaluate the promoted (or given) checkpoint. Returns an exit code."""
    import torch

    ensure_dirs()
    dataset_config: DatasetConfig = DatasetConfig.load()
    eval_config: EvalConfig = EvalConfig.load()

    manifest_path = default_manifest_path(dataset_config)
    if not manifest_path.exists():
        print(f"manifest missing: {manifest_path}; run `fieldlens audit` first.")
        return 2
    manifest = Manifest.load(manifest_path)

    leakage = audit_manifest(manifest)
    if not leakage.passed:
        print("refusing to evaluate: the leakage audit does not pass.")
        print(leakage.summary())
        return 2

    checkpoint = Path(checkpoint) if checkpoint else PROMOTED_CHECKPOINT
    if not checkpoint.exists():
        print(
            f"checkpoint missing: {checkpoint}\n"
            "Train one first with `fieldlens train`."
        )
        return 2

    from fieldlens.models.checkpoint import load_checkpoint

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_checkpoint(
        checkpoint, dataset_classes=dataset_config.classes, device=device
    )

    image_root = Path(image_root) if image_root else IMAGE_DIR
    class_index = dataset_config.class_index()
    class_names = dataset_config.classes
    transform = build_eval_transform(_spec_from(metadata))

    num_classes = len(class_names)
    started = time.time()

    val_entries = manifest.entries_for("val")
    test_entries = manifest.entries_for("test")

    print("FieldLens AI - evaluation")
    print(f"  checkpoint : {checkpoint}")
    print(f"  device     : {device}")

    val_logits, val_labels, val_skipped = collect_logits(
        model, val_entries, class_index, transform, image_root,
        device=device, batch_size=batch_size,
    )
    if val_logits.shape[0] == 0:
        print(
            f"no validation images found under {image_root}; cannot calibrate.\n"
            "Run `fieldlens extract` first."
        )
        return 2

    # --- calibration: fitted on validation ONLY ---------------------------
    fit = fit_temperature(val_logits, val_labels, split="val")
    stored = _store_temperature(checkpoint, fit.temperature)

    print(f"  val        : {val_logits.shape[0]} images ({val_skipped} skipped, missing)")
    print(
        f"  temperature: {fit.temperature:.4f} "
        f"(val NLL {fit.nll_before:.4f} -> {fit.nll_after:.4f}, "
        f"{'improved' if fit.improved else 'no improvement'})"
    )

    test_logits, test_labels, test_skipped = collect_logits(
        model, test_entries, class_index, transform, image_root,
        device=device, batch_size=batch_size,
    )
    if test_logits.shape[0] == 0:
        print(f"no test images found under {image_root}; nothing to evaluate.")
        return 2

    evaluated = int(test_logits.shape[0])
    print(
        f"  test       : {evaluated} of {len(test_entries)} manifest entries "
        f"({test_skipped} not on disk)"
    )

    # --- metrics before and after calibration -----------------------------
    results: dict[str, object] = {}
    for label, temperature in (("uncalibrated", 1.0), ("calibrated", fit.temperature)):
        probs = probabilities(test_logits, temperature)
        predictions = probs.argmax(axis=1)
        matrix = confusion_matrix(test_labels, predictions, num_classes)
        macro = macro_scores(matrix, only_supported=True)
        calibration = expected_calibration_error(probs, test_labels)
        results[label] = {
            "temperature": temperature,
            "accuracy": accuracy(test_labels, predictions),
            "macro_precision": macro["precision"],
            "macro_recall": macro["recall"],
            "macro_f1": macro["f1"],
            "top3_accuracy": top_k_accuracy(probs, test_labels, 3),
            "brier": brier_score(probs, test_labels),
            "ece": calibration.ece,
            "mce": calibration.mce,
            "confusion_matrix": matrix.tolist(),
            "per_class": [
                {**scores, "class_name": class_names[int(scores["index"])]}
                for scores in per_class_scores(matrix)
            ],
            "per_crop": _per_crop_table(test_labels, predictions, class_names),
            "reliability_bins": calibration.bins,
        }

    calibrated_probs = probabilities(test_logits, fit.temperature)

    # --- selective prediction: thresholds chosen on validation ------------
    val_probs = probabilities(val_logits, fit.temperature)
    thresholds = {
        str(coverage): {
            "threshold": threshold_for_coverage(val_probs, coverage),
            "selected_on": "val",
        }
        for coverage in eval_config.target_coverages
    }
    curve = selective_curve(calibrated_probs, test_labels, coverages=None, steps=40)

    # Thresholds come from validation; the accuracy they achieve is measured on
    # test, which is the only honest way to present a coverage claim.
    val_confidences = val_probs.max(axis=1)
    test_confidences = calibrated_probs.max(axis=1)
    test_correct = (calibrated_probs.argmax(axis=1) == test_labels).astype(float)
    for entry in thresholds.values():
        keep = test_confidences >= entry["threshold"]
        entry["test_coverage"] = float(keep.mean()) if keep.size else 0.0
        entry["test_accuracy_at_threshold"] = (
            float(test_correct[keep].mean()) if keep.any() else float("nan")
        )
        entry["test_images_kept"] = int(keep.sum())
    _ = val_confidences

    results["selective_prediction"] = {
        "thresholds": thresholds,
        "risk_coverage_curve": curve,
    }

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checkpoint": str(checkpoint),
        "device": device,
        "temperature_stored_in_checkpoint": stored,
        "calibration_fit": fit.to_json(),
        "split_policy": manifest.split_policy,
        "leakage_audit_passed": leakage.passed,
        "dataset": {
            "repo_id": dataset_config.repo_id,
            "license": dataset_config.license,
            "classes": list(class_names),
            "manifest_counts": manifest.counts(),
            "test_entries_in_manifest": len(test_entries),
            "test_images_evaluated": evaluated,
            "test_images_skipped_missing": test_skipped,
            "val_images_evaluated": int(val_logits.shape[0]),
            "images_evaluated_fraction_of_test": (
                evaluated / len(test_entries) if test_entries else 0.0
            ),
        },
        "model": {
            "backbone": metadata.backbone,
            "image_size": metadata.image_size,
            "parameters": sum(p.numel() for p in model.parameters()),
        },
        "results": results,
        "elapsed_seconds": round(time.time() - started, 2),
        "caveats": [
            "Metrics are on a controlled benchmark of single detached leaves with "
            "uniform backgrounds, not field conditions.",
            "Temperature scaling is fitted on validation and does not transfer to "
            "other cameras, lighting, or backgrounds.",
            "If the acquisition was capped, only a subset of the official test "
            "split was evaluated; see dataset.test_images_evaluated.",
        ],
    }

    report_path = REPORT_DIR / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    figures = {
        "confusion_matrix": _figure_confusion(
            np.asarray(results["calibrated"]["confusion_matrix"]),
            class_names,
            FIGURE_DIR / "confusion_matrix.png",
        ),
        "reliability_diagram": _figure_reliability(
            results["calibrated"]["reliability_bins"],
            results["calibrated"]["ece"],
            FIGURE_DIR / "reliability_diagram.png",
        ),
    }
    report["figures"] = figures

    # --- console summary --------------------------------------------------
    print()
    print(f"  {'metric':<18}{'uncalibrated':>14}{'calibrated':>14}")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1",
                "top3_accuracy", "ece", "brier"):
        print(
            f"  {key:<18}{results['uncalibrated'][key]:>14.4f}"
            f"{results['calibrated'][key]:>14.4f}"
        )
    print()
    weakest = sorted(
        results["calibrated"]["per_class"], key=lambda r: r["f1"]
    )[:3]
    print("  weakest classes by F1:")
    for row in weakest:
        print(
            f"    {row['class_name']:<52} F1 {row['f1']:.3f} "
            f"(recall {row['recall']:.3f}, n={int(row['support'])})"
        )
    print()
    print("  selective prediction (threshold from validation, accuracy on test):")
    for coverage, entry in sorted(thresholds.items(), key=lambda kv: float(kv[0])):
        print(
            f"    target {float(coverage):.0%} -> threshold "
            f"{entry['threshold']:.3f}, test coverage "
            f"{entry['test_coverage']:.1%}, accuracy "
            f"{entry['test_accuracy_at_threshold']:.4f}"
        )
    print()
    print(f"  report : {report_path}")
    print(f"  figures: {figures}")
    return 0


def _spec_from(metadata):
    """Rebuild the preprocessing spec from checkpoint metadata."""
    from fieldlens.data.transforms import spec_from_checkpoint

    return spec_from_checkpoint(metadata)
