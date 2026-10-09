"""Integration fixtures.

These build a *real* torch model and a real checkpoint file, but use
``pretrained=False`` so the suite never needs the network. The image size is
reduced to keep the tests fast; nothing here downloads the dataset.
"""

from __future__ import annotations

from pathlib import Path

import pytest

TEST_CLASSES = (
    "Corn_(maize)___healthy",
    "Grape___healthy",
    "Potato___Early_blight",
)
TEST_IMAGE_SIZE = 64


@pytest.fixture(scope="session")
def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
    except Exception:  # pragma: no cover - torch is a core dependency here
        pytest.skip("torch is not installed")
    return True


@pytest.fixture
def tiny_model(_torch_available):
    """An untrained MobileNetV3-Small sized for the test class list."""
    import torch

    from fieldlens.config import ModelConfig
    from fieldlens.models.factory import build_model

    torch.manual_seed(0)
    config = ModelConfig(
        backbone="mobilenet_v3_small",
        weights="NONE",
        image_size=TEST_IMAGE_SIZE,
        dropout=0.0,
    )
    return build_model(config, len(TEST_CLASSES), pretrained=False)


@pytest.fixture
def tiny_checkpoint(tmp_path: Path, _torch_available, tiny_model):
    """A saved checkpoint on disk, plus its class list."""
    import torch

    from fieldlens.models.checkpoint import CheckpointMetadata, save_checkpoint
    from fieldlens.models.factory import preprocessing_spec

    torch.manual_seed(0)
    spec = preprocessing_spec("mobilenet_v3_small", "NONE", TEST_IMAGE_SIZE)
    metadata = CheckpointMetadata(
        class_names=TEST_CLASSES,
        preprocessing=spec.as_dict(),
        backbone="mobilenet_v3_small",
        image_size=TEST_IMAGE_SIZE,
        weights="NONE",
        temperature=1.0,
        run_name="test",
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, tiny_model, metadata)
    return path, TEST_CLASSES


@pytest.fixture
def saved_png(tmp_path: Path, rgb_image) -> Path:
    """A synthetic image on disk (generated, not dataset content)."""
    path = tmp_path / "leaf.png"
    rgb_image.save(path)
    return path
