"""Shared fixtures.

Tests never download the dataset and never require a trained checkpoint: split
metadata is generated synthetically, and model-dependent tests skip cleanly when
no checkpoint is present. This keeps the suite fast and offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

CLASSES = (
    "Corn_(maize)___healthy",
    "Grape___healthy",
    "Potato___Early_blight",
)


def _entry(class_name: str, uuid: str, leaf_tail: str) -> str:
    return f"raw/color/{class_name}/{uuid}___{leaf_tail}.JPG"


@pytest.fixture
def synthetic_metadata(tmp_path: Path) -> Path:
    """Build split files with a *deliberate* cross-split leaf overlap.

    Layout (leaf ids are the lowercased text after ``___``):

    train: leaf 1 (x2 images), leaf 4
    test : leaf 2, leaf 3, leaf 4   <- leaf 4 leaks across the boundary
    """
    metadata = tmp_path / "metadata"
    (metadata / "splits").mkdir(parents=True)
    (metadata / "leaf_grouping").mkdir(parents=True)

    corn, grape, potato = CLASSES
    train_lines = [
        _entry(corn, "aaaaaaaa-0000-0000-0000-000000000001", "leaf 1"),
        _entry(corn, "aaaaaaaa-0000-0000-0000-000000000002", "leaf 1"),
        _entry(corn, "aaaaaaaa-0000-0000-0000-000000000004", "leaf 4"),
        _entry(grape, "bbbbbbbb-0000-0000-0000-000000000005", "leaf 5"),
        _entry(grape, "bbbbbbbb-0000-0000-0000-000000000006", "leaf 6"),
        _entry(potato, "cccccccc-0000-0000-0000-000000000007", "leaf 7"),
        _entry(potato, "cccccccc-0000-0000-0000-000000000008", "leaf 8"),
        _entry(potato, "cccccccc-0000-0000-0000-000000000009", "leaf 9"),
        _entry(potato, "cccccccc-0000-0000-0000-000000000010", "leaf 10"),
    ]
    test_lines = [
        _entry(corn, "dddddddd-0000-0000-0000-000000000002", "leaf 2"),
        _entry(corn, "dddddddd-0000-0000-0000-000000000003", "leaf 3"),
        _entry(corn, "dddddddd-0000-0000-0000-000000000004", "leaf 4"),
        _entry(grape, "eeeeeeee-0000-0000-0000-000000000011", "leaf 11"),
        _entry(potato, "ffffffff-0000-0000-0000-000000000012", "leaf 12"),
    ]

    (metadata / "splits" / "color_train.txt").write_text(
        "\n".join(train_lines) + "\n", encoding="utf-8"
    )
    (metadata / "splits" / "color_test.txt").write_text(
        "\n".join(test_lines) + "\n", encoding="utf-8"
    )

    leaf_map = {
        "leaf 1": ["Corn_(maize)___healthy:::1.0"],
        "leaf 4": ["Corn_(maize)___healthy:::4.0"],
        "leaf 5": ["Grape___healthy:::5.0"],
    }
    (metadata / "leaf_grouping" / "leaf-map.json").write_text(
        json.dumps(leaf_map), encoding="utf-8"
    )
    return metadata


@pytest.fixture
def dataset_config(synthetic_metadata: Path):
    """A DatasetConfig bound to the synthetic metadata."""
    from fieldlens.config import DatasetConfig

    return DatasetConfig(
        repo_id="synthetic/test",
        config="color",
        license="CC-BY-SA-3.0",
        citation="synthetic",
        homepage="https://example.invalid",
        archive_name="data.zip",
        archive_mb=1.0,
        metadata_base_url="https://example.invalid",
        metadata_files={},
        metadata_dir=synthetic_metadata,
        val_fraction=0.25,
        seed=42,
        classes=CLASSES,
        image_size=64,
        cache_size=96,
        split_policy="official",
    )


@pytest.fixture
def synthetic_manifest(dataset_config):
    """Manifest built with the ``official`` policy, so the leak is still present."""
    from fieldlens.data.split import build_manifest

    return build_manifest(dataset_config, policy="official")


@pytest.fixture
def rgb_image():
    """A deterministic synthetic RGB image (no dataset content)."""
    from PIL import Image

    image = Image.new("RGB", (128, 128))
    pixels = image.load()
    for x in range(128):
        for y in range(128):
            pixels[x, y] = ((x * 2) % 256, (y * 2) % 256, ((x + y) * 3) % 256)
    return image
