"""Model construction.

Normalisation constants are read from torchvision's own weight metadata rather
than copied by hand, so preprocessing can never silently drift from the weights
that were actually loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fieldlens.config import ModelConfig

__all__ = [
    "ModelBuildError",
    "PreprocessingSpec",
    "SUPPORTED_BACKBONES",
    "build_model",
    "preprocessing_spec",
    "count_parameters",
]

#: Backbones we support. MobileNetV3-Small is the plan's choice: ~2.5M params and
#: small enough to train comfortably in 4 GB of VRAM.
SUPPORTED_BACKBONES = {
    "mobilenet_v3_small": "MobileNet_V3_Small_Weights",
    "mobilenet_v3_large": "MobileNet_V3_Large_Weights",
}


class ModelBuildError(ValueError):
    """The requested backbone or weight set is not available."""


@dataclass(frozen=True)
class PreprocessingSpec:
    """Input normalisation and geometry, recorded inside every checkpoint."""

    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    resize_size: int
    center_crop: int
    version: str = "imagenet-v1"

    def as_dict(self) -> dict[str, Any]:
        return {
            "image_size": self.image_size,
            "mean": list(self.mean),
            "std": list(self.std),
            "resize_size": self.resize_size,
            "center_crop": self.center_crop,
            "version": self.version,
        }


def _weights_enum(backbone: str, weights: str):
    import torchvision.models as tvm

    if backbone not in SUPPORTED_BACKBONES:
        raise ModelBuildError(
            f"unsupported backbone {backbone!r}; expected one of "
            f"{sorted(SUPPORTED_BACKBONES)}"
        )
    enum_name = SUPPORTED_BACKBONES[backbone]
    try:
        enum = getattr(tvm, enum_name)
    except AttributeError as exc:  # pragma: no cover - torchvision API change
        raise ModelBuildError(
            f"this torchvision build has no {enum_name}"
        ) from exc
    try:
        return enum[weights]
    except KeyError as exc:
        available = [w.name for w in enum]
        raise ModelBuildError(
            f"unknown weights {weights!r} for {backbone}; available: {available}"
        ) from exc


def preprocessing_spec(backbone: str, weights: str, image_size: int) -> PreprocessingSpec:
    """Preprocessing metadata for a backbone/weight combination.

    When pretrained weights are used, the mean/std and resize geometry come from
    ``weights.meta``. For randomly initialised models we fall back to the
    ImageNet convention and label the version accordingly, so a checkpoint always
    states which normalisation it expects.
    """
    if weights.upper() == "NONE":
        return PreprocessingSpec(
            image_size=image_size,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            resize_size=int(image_size / 0.875),
            center_crop=image_size,
            version="imagenet-convention-untrained",
        )

    enum = _weights_enum(backbone, weights)
    meta = enum.meta
    mean = tuple(float(v) for v in meta.get("mean", (0.485, 0.456, 0.406)))
    std = tuple(float(v) for v in meta.get("std", (0.229, 0.224, 0.225)))
    return PreprocessingSpec(
        image_size=image_size,
        mean=mean,  # type: ignore[arg-type]
        std=std,  # type: ignore[arg-type]
        resize_size=int(image_size / 0.875),
        center_crop=image_size,
        version=f"{backbone}-{weights.lower()}",
    )


def build_model(config: ModelConfig, num_classes: int, *, pretrained: bool = True):
    """Build the classifier with a fresh head for ``num_classes`` classes."""
    import torch.nn as nn
    import torchvision.models as tvm

    if num_classes < 2:
        raise ModelBuildError(f"num_classes must be >= 2, got {num_classes}")
    if config.backbone not in SUPPORTED_BACKBONES:
        raise ModelBuildError(
            f"unsupported backbone {config.backbone!r}; expected one of "
            f"{sorted(SUPPORTED_BACKBONES)}"
        )

    factory = getattr(tvm, config.backbone)
    weights = None
    if pretrained and config.weights.upper() != "NONE":
        weights = _weights_enum(config.backbone, config.weights)

    model = factory(weights=weights)

    # Replace the classifier head. MobileNetV3 exposes ``classifier``; handle the
    # generic ``fc`` layout too so a backbone swap stays a config change.
    if hasattr(model, "classifier"):
        last_linear_idx = None
        for idx, layer in enumerate(model.classifier):
            if isinstance(layer, nn.Linear):
                last_linear_idx = idx
        if last_linear_idx is None:  # pragma: no cover - unexpected architecture
            raise ModelBuildError(f"{config.backbone} classifier has no Linear layer")
        in_features = model.classifier[last_linear_idx].in_features
        # Replace the head *in place*. Inserting an extra Dropout would shift the
        # positional indices of the remaining layers (classifier.3 vs
        # classifier.4), which makes checkpoints saved with one dropout setting
        # fail to load under another. MobileNetV3 already has a Dropout in the
        # head, so its probability is adjusted rather than a new layer added.
        layers: list[nn.Module] = []
        for idx, layer in enumerate(model.classifier):
            if idx == last_linear_idx:
                layers.append(nn.Linear(in_features, num_classes))
            elif isinstance(layer, nn.Dropout) and config.dropout > 0:
                layers.append(nn.Dropout(p=config.dropout))
            else:
                layers.append(layer)
        model.classifier = nn.Sequential(*layers)
    elif hasattr(model, "fc"):  # pragma: no cover - alternative architectures
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:  # pragma: no cover
        raise ModelBuildError(f"cannot locate the classifier head of {config.backbone}")

    return model


def count_parameters(model) -> dict[str, int]:
    """Parameter counts, recorded in run manifests and the model card."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def freeze_backbone(model) -> None:
    """Freeze everything except the classifier head (training stage 1)."""
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith("classifier")


def unfreeze_last_block(model, *, block: str = "features.12") -> int:
    """Unfreeze feature blocks from ``block`` onward plus the head (stage 2).

    Returns the number of parameters left trainable, so the run manifest can
    record exactly what was optimised.
    """
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(("classifier", block))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
