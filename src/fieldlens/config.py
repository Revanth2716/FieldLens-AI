"""Configuration loading with validation.

Configs are YAML so the class list and hyperparameters can change without
touching code. Every loader validates its input and raises
:class:`ConfigError` with a message naming the offending key, so a malformed
config fails loudly instead of producing subtly wrong training runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from fieldlens.paths import CONFIG_DIR

__all__ = [
    "ConfigError",
    "DatasetConfig",
    "ModelConfig",
    "TrainConfig",
    "EvalConfig",
    "load_config",
]


class ConfigError(ValueError):
    """A configuration file is missing a key or holds an invalid value."""


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ConfigError(f"config not found: {path}")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:  # pragma: no cover - malformed file
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"config {path} must contain a mapping at the top level")
    return data


def _require(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing required key '{key}' in {context}")
    return mapping[key]


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset source, metadata locations, splits, classes and image sizes."""

    repo_id: str
    config: str
    license: str
    citation: str
    homepage: str
    archive_name: str
    archive_mb: float
    metadata_base_url: str
    metadata_files: dict[str, float]
    metadata_dir: Path
    val_fraction: float
    seed: int
    classes: tuple[str, ...]
    image_size: int
    cache_size: int
    split_policy: str = "leak_free"

    def __post_init__(self) -> None:
        if self.split_policy not in ("leak_free", "official"):
            raise ConfigError(
                f"dataset config: split_policy must be 'leak_free' or 'official', "
                f"got {self.split_policy!r}"
            )
        if not self.classes:
            raise ConfigError("dataset config: 'classes' must not be empty")
        if len(set(self.classes)) != len(self.classes):
            raise ConfigError("dataset config: 'classes' contains duplicates")
        if not 0.0 < self.val_fraction < 1.0:
            raise ConfigError(
                f"dataset config: val_fraction_of_train_leaves must be in (0, 1), "
                f"got {self.val_fraction}"
            )
        for name, value in (("image.size", self.image_size), ("image.cache_size", self.cache_size)):
            if value <= 0:
                raise ConfigError(f"dataset config: {name} must be positive, got {value}")
        if self.cache_size < self.image_size:
            raise ConfigError(
                f"dataset config: image.cache_size ({self.cache_size}) must be >= "
                f"image.size ({self.image_size})"
            )

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def class_index(self) -> dict[str, int]:
        """Deterministic class name -> index mapping, ordered as configured.

        This ordering is written into every checkpoint; changing the config
        order invalidates existing checkpoints, which the loader detects.
        """
        return {name: i for i, name in enumerate(self.classes)}

    def class_to_crop(self) -> dict[str, str]:
        """Map each class to its crop, from the ``Crop___State`` convention."""
        return {name: name.split("___", 1)[0] for name in self.classes}

    @classmethod
    def load(cls, path: Path | None = None) -> DatasetConfig:
        path = path or CONFIG_DIR / "dataset.yaml"
        raw = _read_yaml(path)
        source = _require(raw, "source", "dataset.yaml")
        metadata = _require(raw, "metadata", "dataset.yaml")
        splits = _require(raw, "splits", "dataset.yaml")
        image = _require(raw, "image", "dataset.yaml")
        return cls(
            repo_id=_require(source, "repo_id", "source"),
            config=_require(source, "config", "source"),
            license=source.get("license", "unknown"),
            citation=(source.get("citation") or "").strip(),
            homepage=source.get("homepage", ""),
            archive_name=source.get("archive_name", "data.zip"),
            archive_mb=float(source.get("archive_mb", 0.0)),
            metadata_base_url=_require(metadata, "base_url", "metadata"),
            metadata_files=dict(metadata.get("files", {})),
            metadata_dir=Path(_require(metadata, "local_dir", "metadata")),
            val_fraction=float(
                _require(splits, "val_fraction_of_train_leaves", "splits")
            ),
            seed=int(_require(splits, "seed", "splits")),
            classes=tuple(_require(raw, "classes", "dataset.yaml")),
            image_size=int(_require(image, "size", "image")),
            cache_size=int(_require(image, "cache_size", "image")),
            split_policy=splits.get("split_policy", "leak_free"),
        )


@dataclass(frozen=True)
class ModelConfig:
    """Backbone selection and inference input specification."""

    backbone: str
    weights: str
    image_size: int
    dropout: float

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ConfigError(f"model config: image_size must be positive, got {self.image_size}")
        if not 0.0 <= self.dropout < 1.0:
            raise ConfigError(f"model config: dropout must be in [0, 1), got {self.dropout}")

    @classmethod
    def load(cls, path: Path | None = None) -> ModelConfig:
        raw = _read_yaml(path or CONFIG_DIR / "model.yaml")
        model = _require(raw, "model", "model.yaml")
        return cls(
            backbone=_require(model, "backbone", "model"),
            weights=model.get("weights", "DEFAULT"),
            image_size=int(_require(model, "image_size", "model")),
            dropout=float(model.get("dropout", 0.2)),
        )


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation and augmentation settings for the staged training loop."""

    seed: int
    batch_size: int
    num_workers: int
    head_epochs: int
    finetune_epochs: int
    head_lr: float
    finetune_lr: float
    weight_decay: float
    label_smoothing: float
    amp: bool
    early_stopping_patience: int
    augment: dict[str, Any]
    balanced_sampling: bool

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ConfigError(f"train config: batch_size must be positive, got {self.batch_size}")
        if self.num_workers < 0:
            raise ConfigError(f"train config: num_workers must be >= 0, got {self.num_workers}")
        if self.head_epochs < 0 or self.finetune_epochs < 0:
            raise ConfigError("train config: epoch counts must be non-negative")
        if self.head_epochs + self.finetune_epochs == 0:
            raise ConfigError("train config: at least one epoch must be configured")
        if self.early_stopping_patience < 1:
            raise ConfigError("train config: early_stopping_patience must be >= 1")

    @property
    def total_epochs(self) -> int:
        return self.head_epochs + self.finetune_epochs

    @classmethod
    def load(cls, path: Path | None = None) -> TrainConfig:
        raw = _read_yaml(path or CONFIG_DIR / "train.yaml")
        train = _require(raw, "train", "train.yaml")
        return cls(
            seed=int(train.get("seed", 42)),
            batch_size=int(_require(train, "batch_size", "train")),
            num_workers=int(train.get("num_workers", 2)),
            head_epochs=int(train.get("head_epochs", 0)),
            finetune_epochs=int(train.get("finetune_epochs", 0)),
            head_lr=float(train.get("head_lr", 1e-3)),
            finetune_lr=float(train.get("finetune_lr", 1e-4)),
            weight_decay=float(train.get("weight_decay", 1e-4)),
            label_smoothing=float(train.get("label_smoothing", 0.0)),
            amp=bool(train.get("amp", True)),
            early_stopping_patience=int(train.get("early_stopping_patience", 3)),
            augment=dict(train.get("augment", {})),
            balanced_sampling=bool(train.get("balanced_sampling", False)),
        )


@dataclass(frozen=True)
class EvalConfig:
    """Confidence policy and reliability-experiment definitions."""

    target_coverages: tuple[float, ...]
    warning_bands: dict[str, float]
    degradation: dict[str, Any]

    def __post_init__(self) -> None:
        for coverage in self.target_coverages:
            if not 0.0 < coverage <= 1.0:
                raise ConfigError(
                    f"eval config: target coverages must be in (0, 1], got {coverage}"
                )
        if sorted(self.target_coverages) != list(self.target_coverages):
            raise ConfigError("eval config: target_coverages must be sorted ascending")

    @classmethod
    def load(cls, path: Path | None = None) -> EvalConfig:
        raw = _read_yaml(path or CONFIG_DIR / "eval.yaml")
        evaluation = _require(raw, "eval", "eval.yaml")
        coverages = tuple(
            float(c) for c in _require(evaluation, "target_coverages", "eval")
        )
        return cls(
            target_coverages=coverages,
            warning_bands=dict(evaluation.get("warning_bands", {})),
            degradation=dict(evaluation.get("degradation", {})),
        )


_LOADERS = {
    "dataset": DatasetConfig.load,
    "model": ModelConfig.load,
    "train": TrainConfig.load,
    "eval": EvalConfig.load,
}


def load_config(name: str, path: Path | None = None):
    """Load a config by short name (``"dataset"``, ``"model"``, ...)."""
    try:
        loader = _LOADERS[name]
    except KeyError:
        raise ConfigError(
            f"unknown config '{name}'; expected one of {sorted(_LOADERS)}"
        ) from None
    return loader(path)
