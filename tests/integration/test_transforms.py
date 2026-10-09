"""Integration tests for training and evaluation transforms."""

from __future__ import annotations

import pytest

from fieldlens.data.transforms import (
    build_eval_transform,
    build_train_transform,
    spec_from_checkpoint,
    spec_to_dict,
)
from fieldlens.models.factory import preprocessing_spec


@pytest.fixture
def spec():
    return preprocessing_spec("mobilenet_v3_small", "NONE", 64)


def test_eval_transform_output_shape(spec, rgb_image):
    tensor = build_eval_transform(spec)(rgb_image)
    assert tuple(tensor.shape) == (3, spec.center_crop, spec.center_crop)


def test_train_transform_output_shape(spec, rgb_image):
    tensor = build_train_transform(spec, {})(rgb_image)
    assert tuple(tensor.shape) == (3, spec.image_size, spec.image_size)


def test_eval_transform_is_deterministic(spec, rgb_image):
    import torch

    transform = build_eval_transform(spec)
    assert torch.equal(transform(rgb_image), transform(rgb_image))


def test_train_transform_is_reproducible_under_a_seed(spec, rgb_image):
    import torch

    torch.manual_seed(123)
    first = build_train_transform(spec, {})(rgb_image)
    torch.manual_seed(123)
    second = build_train_transform(spec, {})(rgb_image)
    assert torch.equal(first, second)


def test_train_transform_augments(spec, rgb_image):
    """With augmentation enabled, two draws should generally differ."""
    import torch

    transform = build_train_transform(spec, {})
    torch.manual_seed(1)
    first = transform(rgb_image)
    torch.manual_seed(999)
    second = transform(rgb_image)
    assert not torch.equal(first, second)


def test_normalisation_uses_the_spec_constants(rgb_image):
    """A white image must normalise to (1 - mean) / std exactly."""

    spec = preprocessing_spec("mobilenet_v3_small", "NONE", 64)
    from PIL import Image

    white = Image.new("RGB", (64, 64), (255, 255, 255))
    tensor = build_eval_transform(spec)(white)
    for channel in range(3):
        expected = (1.0 - spec.mean[channel]) / spec.std[channel]
        assert tensor[channel].mean().item() == pytest.approx(expected, abs=1e-4)


def test_spec_round_trips_through_a_checkpoint_metadata(tiny_checkpoint):
    from fieldlens.models.checkpoint import load_metadata

    path, _ = tiny_checkpoint
    metadata = load_metadata(path)
    recovered = spec_from_checkpoint(metadata)
    assert recovered.image_size == metadata.image_size
    assert len(recovered.mean) == 3
    assert len(recovered.std) == 3


def test_spec_to_dict_is_serialisable():
    import json

    spec = preprocessing_spec("mobilenet_v3_small", "NONE", 32)
    payload = spec_to_dict(spec)
    json.dumps(payload)
    assert payload["image_size"] == 32
    assert len(payload["mean"]) == 3


def test_train_transform_disables_augmentation_when_asked(spec, rgb_image):
    """With every augmentation switched off the transform is deterministic."""
    import torch

    options = {
        # Pin the crop so it becomes a full-image resize rather than a random one.
        "random_resized_crop_scale": [1.0, 1.0],
        "random_resized_crop_ratio": [1.0, 1.0],
        "horizontal_flip": False,
        "vertical_flip": False,
        "rotation_degrees": 0,
        "color_jitter": 0,
    }
    transform = build_train_transform(spec, options)
    assert torch.equal(transform(rgb_image), transform(rgb_image))
