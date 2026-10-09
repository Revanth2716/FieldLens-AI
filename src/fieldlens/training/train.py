"""Staged training loop.

Stage 1 freezes the backbone and trains only the classifier head (fast, produces
a usable checkpoint within minutes). Stage 2 unfreezes the last feature block and
fine-tunes at a low learning rate.

Early stopping is on **validation macro F1**, not accuracy: ``Potato___healthy``
holds only ~152 images in this dataset, and overall accuracy would happily hide a
model that fails the rarest class.

The loop is deliberately small and explicit. Everything it measures is written to
a run manifest so published numbers are traceable to a command.
"""

from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from fieldlens.config import DatasetConfig, ModelConfig, TrainConfig
from fieldlens.data.leakage import assert_no_leakage, audit_manifest
from fieldlens.data.split import Manifest, default_manifest_path
from fieldlens.data.transforms import build_eval_transform, build_train_transform
from fieldlens.env import format_bytes
from fieldlens.eval.metrics import accuracy, confusion_matrix, macro_scores
from fieldlens.models.checkpoint import CheckpointMetadata, save_checkpoint
from fieldlens.models.factory import (
    build_model,
    count_parameters,
    freeze_backbone,
    preprocessing_spec,
    unfreeze_last_block,
)
from fieldlens.paths import (
    IMAGE_DIR,
    PROMOTED_CHECKPOINT,
    REPORT_DIR,
    ensure_dirs,
)
from fieldlens.training.dataset import LeafImageDataset

__all__ = ["train_from_config", "RunManifest"]


@dataclass
class RunManifest:
    """Everything needed to reproduce or audit a training run."""

    run_name: str
    started_at: str
    finished_at: str | None = None
    seed: int = 0
    device: str = "cpu"
    device_name: str | None = None
    amp_enabled: bool = False
    dataset: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    train: dict = field(default_factory=dict)
    split_counts: dict = field(default_factory=dict)
    class_counts: dict = field(default_factory=dict)
    leakage: dict = field(default_factory=dict)
    parameters: dict = field(default_factory=dict)
    epochs: list[dict] = field(default_factory=list)
    peak_vram_bytes: int | None = None
    epoch_seconds: list[float] = field(default_factory=list)
    history: dict = field(default_factory=dict)
    git_sha: str | None = None
    python: str = ""
    platform: str = ""
    torch: str | None = None
    torchvision: str | None = None
    checkpoint: str | None = None


def _git_sha() -> str | None:
    """Current commit, or None when the repo has no commits yet."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return None


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device() -> tuple[str, str | None]:
    import torch

    if torch.cuda.is_available():
        return "cuda", torch.cuda.get_device_name(0)
    return "cpu", None


def evaluate_model(model, loader, device: str, num_classes: int) -> dict:
    """Loss-free evaluation returning accuracy, macro scores and a matrix."""
    import torch

    model.eval()
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            all_logits.append(logits.detach().float().cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    predictions = logits.argmax(dim=1).numpy()
    matrix = confusion_matrix(labels, predictions, num_classes)
    macro = macro_scores(matrix, only_supported=True)
    return {
        "accuracy": accuracy(labels, predictions),
        "macro_precision": macro["precision"],
        "macro_recall": macro["recall"],
        "macro_f1": macro["f1"],
        "confusion_matrix": matrix.tolist(),
        "num_samples": int(len(labels)),
    }


def _run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device: str,
    *,
    scaler=None,
    train: bool = True,
) -> float:
    import torch

    model.train(train)
    total_loss = 0.0
    total_samples = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(inputs)
                loss = criterion(logits, labels)
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        else:
            logits = model(inputs)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                optimizer.step()

        total_loss += float(loss.detach()) * labels.size(0)
        total_samples += labels.size(0)
    return total_loss / max(1, total_samples)


def _train_stage(
    model,
    train_loader,
    val_loader,
    *,
    epochs: int,
    learning_rate: float,
    train_config: TrainConfig,
    device: str,
    num_classes: int,
    manifest: RunManifest,
    patience: int,
    best_state: dict | None = None,
    best_f1: float = -1.0,
) -> tuple[dict | None, float, int]:
    """Train for up to ``epochs``, returning the best state and score."""
    import torch

    if epochs <= 0:
        return best_state, best_f1, 0

    criterion = torch.nn.CrossEntropyLoss(
        label_smoothing=train_config.label_smoothing
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = train_config.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    stale = 0
    ran = 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss = _run_epoch(
            model, train_loader, criterion, optimizer, device, scaler=scaler, train=True
        )
        elapsed = time.perf_counter() - started
        scheduler.step()

        val_metrics = evaluate_model(model, val_loader, device, num_classes)
        f1 = val_metrics["macro_f1"]
        ran += 1

        manifest.epochs.append(
            {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val_accuracy": round(val_metrics["accuracy"], 6),
                "val_macro_f1": round(f1, 6),
                "seconds": round(elapsed, 2),
            }
        )
        manifest.epoch_seconds.append(round(elapsed, 2))
        if device == "cuda":
            manifest.peak_vram_bytes = int(torch.cuda.max_memory_allocated())

        print(
            f"    epoch {epoch:>2}/{epochs}  loss {train_loss:.4f}  "
            f"val_acc {val_metrics['accuracy']:.4f}  val_macroF1 {f1:.4f}  "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"    early stop: no macro-F1 improvement for {patience} epochs")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return best_state, best_f1, ran


def train_from_config(
    *,
    epochs_override: int | None = None,
    batch_size_override: int | None = None,
    image_root: Path | None = None,
    run_name: str | None = None,
    promote: bool = True,
    only_present: bool = False,
) -> int:
    """Train and save a checkpoint. Returns a process exit code."""
    import torch

    ensure_dirs()
    dataset_config: DatasetConfig = DatasetConfig.load()
    model_config: ModelConfig = ModelConfig.load()
    train_config: TrainConfig = TrainConfig.load()

    if batch_size_override:
        train_config = _replace(train_config, batch_size=batch_size_override)
    if epochs_override:
        # Split the override across the two stages, keeping the head brief.
        head = max(0, min(train_config.head_epochs, epochs_override // 4))
        train_config = _replace(
            train_config, head_epochs=head, finetune_epochs=epochs_override - head
        )

    manifest_path = default_manifest_path(dataset_config)
    if not manifest_path.exists():
        print(
            f"manifest missing: {manifest_path}\n"
            "Run `fieldlens audit` to build it from the dataset metadata.",
            file=sys.stderr,
        )
        return 2

    manifest = Manifest.load(manifest_path)
    leakage = audit_manifest(manifest)
    assert_no_leakage(leakage, context="training")

    image_root = Path(image_root) if image_root else IMAGE_DIR
    run_name = run_name or f"run_{time.strftime('%Y%m%d_%H%M%S')}"

    _seed_everything(train_config.seed)
    device, device_name = _resolve_device()

    spec = preprocessing_spec(
        model_config.backbone, model_config.weights, model_config.image_size
    )
    class_index = dataset_config.class_index()

    train_entries = manifest.entries_for("train")
    val_entries = manifest.entries_for("val")

    if only_present:
        # Used by the MVP path and for resumable acquisition runs: train on the
        # images that actually exist, and report exactly how many were dropped
        # rather than silently skipping them.
        def _present(entries):
            kept = [e for e in entries if (image_root / e.path).exists()]
            return kept

        before = (len(train_entries), len(val_entries))
        # image_root is resolved just above; re-resolve for clarity
        image_root = Path(image_root)
        train_entries, val_entries = _present(train_entries), _present(val_entries)
        print(
            f"  only-present   : kept {len(train_entries)}/{before[0]} train and "
            f"{len(val_entries)}/{before[1]} val images that exist under {image_root}"
        )

    if not train_entries or not val_entries:
        print(
            f"manifest has {len(train_entries)} train and {len(val_entries)} val "
            "entries; both must be non-empty",
            file=sys.stderr,
        )
        return 2

    train_ds = LeafImageDataset(
        train_entries, class_index, build_train_transform(spec, train_config.augment), image_root
    )
    val_ds = LeafImageDataset(
        val_entries, class_index, build_eval_transform(spec), image_root
    )

    missing = train_ds.missing_files()[:5]
    if missing:
        print(
            f"{len(train_ds.missing_files())} training images are missing under "
            f"{image_root}, e.g. {missing[0]}\n"
            "Acquire the images first (`fieldlens extract`).",
            file=sys.stderr,
        )
        return 2

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        persistent_workers=train_config.num_workers > 0,
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=train_config.batch_size,
        shuffle=False,
        num_workers=train_config.num_workers,
        persistent_workers=train_config.num_workers > 0,
    )

    model = build_model(model_config, dataset_config.num_classes, pretrained=True)
    model.to(device)

    run_manifest = RunManifest(
        run_name=run_name,
        started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        seed=train_config.seed,
        device=device,
        device_name=device_name,
        amp_enabled=bool(train_config.amp and device == "cuda"),
        dataset={"repo_id": dataset_config.repo_id, "config": dataset_config.config,
                 "license": dataset_config.license, "split_policy": manifest.split_policy,
                 "image_root": str(image_root)},
        model=asdict(model_config),
        train=asdict(train_config),
        split_counts=manifest.counts(),
        class_counts={"train": train_ds.counts_by_class(), "val": val_ds.counts_by_class()},
        leakage=leakage.to_json(),
        git_sha=_git_sha(),
        python=sys.version.split()[0],
        platform=platform.platform(),
        torch=torch.__version__,
        torchvision=getattr(__import__("torchvision"), "__version__", None),
    )

    print(f"FieldLens AI - training run '{run_name}'")
    print(f"  device        : {device} {device_name or ''}")
    print(f"  amp           : {run_manifest.amp_enabled}")
    print(f"  train / val   : {len(train_ds)} / {len(val_ds)} images")
    print(f"  image root    : {image_root}")
    print(f"  split policy  : {manifest.split_policy}")

    # --- stage 1: head only ---
    if train_config.head_epochs > 0:
        freeze_backbone(model)
        unfrozen = count_parameters(model)["trainable"]
        print(f"\n  stage 1: head only ({unfrozen:,} trainable params)")
        best_state, best_f1, _ = _train_stage(
            model, train_loader, val_loader,
            epochs=train_config.head_epochs,
            learning_rate=train_config.head_lr,
            train_config=train_config, device=device,
            num_classes=dataset_config.num_classes, manifest=run_manifest,
            patience=train_config.early_stopping_patience,
        )
    else:
        best_state, best_f1 = None, -1.0

    # --- stage 2: fine-tune the last block ---
    if train_config.finetune_epochs > 0:
        trainable = unfreeze_last_block(model)
        print(f"\n  stage 2: fine-tune last block ({trainable:,} trainable params)")
        best_state, best_f1, _ = _train_stage(
            model, train_loader, val_loader,
            epochs=train_config.finetune_epochs,
            learning_rate=train_config.finetune_lr,
            train_config=train_config, device=device,
            num_classes=dataset_config.num_classes, manifest=run_manifest,
            patience=train_config.early_stopping_patience,
            best_state=best_state, best_f1=best_f1,
        )

    run_manifest.parameters = count_parameters(model)
    run_manifest.finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_manifest.history = {
        "best_val_macro_f1": None if best_f1 < 0 else round(best_f1, 6),
        "epochs_run": len(run_manifest.epochs),
        "mean_epoch_seconds": (
            round(float(np.mean(run_manifest.epoch_seconds)), 2)
            if run_manifest.epoch_seconds
            else None
        ),
        "total_seconds": round(sum(run_manifest.epoch_seconds), 2),
    }

    metadata = CheckpointMetadata(
        class_names=dataset_config.classes,
        preprocessing=spec.as_dict(),
        backbone=model_config.backbone,
        image_size=model_config.image_size,
        weights=model_config.weights,
        temperature=1.0,  # replaced by the calibration step
        metrics={"val_macro_f1": best_f1 if best_f1 >= 0 else None},
        git_sha=run_manifest.git_sha,
        trained_at=run_manifest.finished_at,
        seed=train_config.seed,
        run_name=run_name,
    )

    run_dir = PROMOTED_CHECKPOINT.parent / run_name
    checkpoint_path = run_dir / "best.pt"
    save_checkpoint(checkpoint_path, model, metadata)
    run_manifest.checkpoint = str(checkpoint_path)

    if promote:
        save_checkpoint(PROMOTED_CHECKPOINT, model, metadata)
        print(f"\n  promoted checkpoint -> {PROMOTED_CHECKPOINT}")

    run_report = REPORT_DIR / "runs" / f"{run_name}.json"
    run_report.parent.mkdir(parents=True, exist_ok=True)
    run_report.write_text(json.dumps(asdict(run_manifest), indent=2), encoding="utf-8")

    print(f"\n  checkpoint    : {checkpoint_path}")
    print(f"  run manifest  : {run_report}")
    if run_manifest.history["mean_epoch_seconds"]:
        print(f"  mean epoch    : {run_manifest.history['mean_epoch_seconds']}s")
    if run_manifest.peak_vram_bytes:
        print(f"  peak VRAM     : {format_bytes(run_manifest.peak_vram_bytes)}")
    if best_f1 >= 0:
        print(f"  best val macro-F1: {best_f1:.4f}")
    return 0


def _replace(config: TrainConfig, **overrides) -> TrainConfig:
    """Return a copy of a frozen config with fields replaced."""
    return TrainConfig(**{**asdict(config), **overrides})
