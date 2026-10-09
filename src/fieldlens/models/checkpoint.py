"""Checkpoint persistence with strict schema validation.

A checkpoint carries more than weights: the ordered class list, the preprocessing
spec, the calibrated temperature and provenance. Loading validates all of it, so
a stale or mismatched checkpoint fails loudly instead of silently predicting the
wrong class names — the worst possible failure mode for this project.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "CheckpointError",
    "CheckpointMetadata",
    "save_checkpoint",
    "load_checkpoint",
    "load_metadata",
]

SCHEMA_VERSION = 1


class CheckpointError(RuntimeError):
    """A checkpoint is missing, corrupt, or inconsistent with its metadata."""


@dataclass
class CheckpointMetadata:
    """Everything needed to reproduce a prediction exactly."""

    class_names: tuple[str, ...]
    preprocessing: dict[str, Any]
    backbone: str
    image_size: int
    weights: str = "DEFAULT"
    temperature: float = 1.0
    metrics: dict[str, Any] = field(default_factory=dict)
    git_sha: str | None = None
    trained_at: str | None = None
    seed: int | None = None
    schema_version: int = SCHEMA_VERSION
    run_name: str | None = None

    def __post_init__(self) -> None:
        if not self.class_names:
            raise CheckpointError("checkpoint metadata has an empty class list")
        if len(set(self.class_names)) != len(self.class_names):
            raise CheckpointError("checkpoint metadata has duplicate class names")
        if self.image_size <= 0:
            raise CheckpointError(
                f"checkpoint metadata has invalid image_size {self.image_size}"
            )
        if self.temperature <= 0:
            raise CheckpointError(
                f"checkpoint temperature must be positive, got {self.temperature}"
            )

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    def class_index(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.class_names)}

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "class_names": list(self.class_names),
            "preprocessing": self.preprocessing,
            "backbone": self.backbone,
            "image_size": self.image_size,
            "weights": self.weights,
            "temperature": self.temperature,
            "metrics": self.metrics,
            "git_sha": self.git_sha,
            "trained_at": self.trained_at,
            "seed": self.seed,
            "run_name": self.run_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointMetadata:
        version = int(data.get("schema_version", 0))
        if version != SCHEMA_VERSION:
            raise CheckpointError(
                f"checkpoint schema version {version} is not supported "
                f"(expected {SCHEMA_VERSION}); retrain to regenerate it"
            )
        return cls(
            class_names=tuple(data["class_names"]),
            preprocessing=dict(data.get("preprocessing", {})),
            backbone=data.get("backbone", "unknown"),
            image_size=int(data.get("image_size", data["preprocessing"]["image_size"])),
            weights=data.get("weights", "DEFAULT"),
            temperature=float(data.get("temperature", 1.0)),
            metrics=dict(data.get("metrics", {})),
            git_sha=data.get("git_sha"),
            trained_at=data.get("trained_at"),
            seed=data.get("seed"),
            run_name=data.get("run_name"),
        )

    def validate_against(self, dataset_classes: tuple[str, ...]) -> None:
        """Fail when a checkpoint's classes do not match the configured classes.

        Order matters: the head's output index maps to ``class_names``, so a
        reordered class list would silently mislabel every prediction.
        """
        if tuple(self.class_names) != tuple(dataset_classes):
            raise CheckpointError(
                "checkpoint class list does not match the configured classes.\n"
                f"  checkpoint ({len(self.class_names)}): {list(self.class_names)}\n"
                f"  configured ({len(dataset_classes)}): {list(dataset_classes)}\n"
                "Retrain, or point --checkpoint at the matching run."
            )


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def save_checkpoint(
    path: Path,
    model,
    metadata: CheckpointMetadata,
    *,
    optimizer_state: dict | None = None,
    extra: dict | None = None,
) -> Path:
    """Persist model weights and metadata to ``path``."""
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "metadata": metadata.as_dict(),
    }
    if optimizer_state is not None:
        payload["optimizer_state"] = optimizer_state
    if extra:
        payload["extra"] = extra
    torch.save(payload, path)
    return path


def load_metadata(path: Path) -> CheckpointMetadata:
    """Read only the metadata (cheap; does not build the model)."""
    import torch

    if not path.exists():
        raise CheckpointError(
            f"checkpoint not found: {path}\n"
            "Train one with `fieldlens train`, or pass --checkpoint."
        )
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - surface any load failure clearly
        raise CheckpointError(f"cannot read checkpoint {path}: {exc}") from exc
    if "metadata" not in payload:
        raise CheckpointError(
            f"checkpoint {path} has no metadata block; it predates "
            f"schema version {SCHEMA_VERSION}"
        )
    return CheckpointMetadata.from_dict(payload["metadata"])


def load_checkpoint(
    path: Path,
    *,
    dataset_classes: tuple[str, ...] | None = None,
    device: str = "cpu",
    build: bool = True,
):
    """Load a checkpoint, validating its class list when given.

    Returns ``(model, metadata)``. The model is in eval mode.
    """
    import torch

    metadata = load_metadata(path)
    if dataset_classes is not None:
        metadata.validate_against(dataset_classes)

    payload = torch.load(path, map_location=device, weights_only=False)

    if not build:
        return None, metadata

    from fieldlens.config import ModelConfig
    from fieldlens.models.factory import build_model

    model_config = ModelConfig(
        backbone=metadata.backbone,
        weights="NONE",  # architecture only; weights come from the state dict
        image_size=metadata.image_size,
        dropout=0.0,
    )
    model = build_model(model_config, metadata.num_classes, pretrained=False)
    try:
        model.load_state_dict(payload["state_dict"])
    except RuntimeError as exc:  # pragma: no cover - corrupt or mismatched state
        raise CheckpointError(
            f"state dict in {path} does not fit the {metadata.backbone} "
            f"architecture with {metadata.num_classes} classes: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    return model, metadata


def write_metadata_sidecar(path: Path, metadata: CheckpointMetadata) -> Path:
    """Write a human-readable ``*.json`` beside the checkpoint for reviewers."""
    sidecar = path.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata.as_dict(), indent=2), encoding="utf-8")
    return sidecar
