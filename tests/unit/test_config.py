"""Unit tests for configuration loading, including the real shipped configs."""

from __future__ import annotations

import pytest
import yaml

from fieldlens.config import (
    ConfigError,
    DatasetConfig,
    EvalConfig,
    ModelConfig,
    TrainConfig,
    load_config,
)
from fieldlens.paths import CONFIG_DIR


def _base_dataset(**overrides):
    data = {
        "repo_id": "x/y",
        "config": "color",
        "license": "CC-BY-SA-3.0",
        "citation": "c",
        "homepage": "h",
        "archive_name": "data.zip",
        "archive_mb": 1.0,
        "metadata_base_url": "https://example.invalid",
        "metadata_files": {},
        "metadata_dir": "data/metadata",
        "val_fraction": 0.1,
        "seed": 1,
        "classes": ("A___healthy", "B___healthy"),
        "image_size": 32,
        "cache_size": 64,
    }
    data.update(overrides)
    return DatasetConfig(**data)


def test_dataset_config_accepts_valid_values():
    config = _base_dataset()
    assert config.num_classes == 2
    assert config.split_policy == "leak_free"


def test_dataset_config_rejects_empty_classes():
    with pytest.raises(ConfigError, match="must not be empty"):
        _base_dataset(classes=())


def test_dataset_config_rejects_duplicate_classes():
    with pytest.raises(ConfigError, match="duplicates"):
        _base_dataset(classes=("A___healthy", "A___healthy"))


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.2, 1.5])
def test_dataset_config_rejects_bad_val_fraction(bad):
    with pytest.raises(ConfigError, match="val_fraction"):
        _base_dataset(val_fraction=bad)


def test_dataset_config_rejects_unknown_split_policy():
    with pytest.raises(ConfigError, match="split_policy"):
        _base_dataset(split_policy="whatever")


def test_dataset_config_requires_cache_at_least_image_size():
    with pytest.raises(ConfigError, match="cache_size"):
        _base_dataset(image_size=224, cache_size=128)


def test_class_index_is_ordered_as_configured():
    config = _base_dataset(classes=("Z___healthy", "A___healthy"))
    assert config.class_index() == {"Z___healthy": 0, "A___healthy": 1}


def test_class_to_crop_derives_crop_names():
    config = _base_dataset(classes=("Tomato___healthy", "Grape___Black_rot"))
    assert config.class_to_crop()["Tomato___healthy"] == "Tomato"
    assert config.class_to_crop()["Grape___Black_rot"] == "Grape"


def test_model_config_validation():
    assert ModelConfig("mobilenet_v3_small", "DEFAULT", 224, 0.2).image_size == 224
    with pytest.raises(ConfigError, match="image_size"):
        ModelConfig("m", "DEFAULT", 0, 0.2)
    with pytest.raises(ConfigError, match="dropout"):
        ModelConfig("m", "DEFAULT", 224, 1.0)


def test_train_config_requires_at_least_one_epoch():
    with pytest.raises(ConfigError, match="at least one epoch"):
        TrainConfig(
            seed=1,
            batch_size=4,
            num_workers=0,
            head_epochs=0,
            finetune_epochs=0,
            head_lr=1e-3,
            finetune_lr=1e-4,
            weight_decay=0.0,
            label_smoothing=0.0,
            amp=True,
            early_stopping_patience=2,
            augment={},
            balanced_sampling=False,
        )


def test_train_config_total_epochs():
    config = TrainConfig(
        seed=1,
        batch_size=4,
        num_workers=0,
        head_epochs=2,
        finetune_epochs=3,
        head_lr=1e-3,
        finetune_lr=1e-4,
        weight_decay=0.0,
        label_smoothing=0.0,
        amp=True,
        early_stopping_patience=2,
        augment={},
        balanced_sampling=False,
    )
    assert config.total_epochs == 5


def test_eval_config_rejects_unsorted_coverages():
    with pytest.raises(ConfigError, match="ascending"):
        EvalConfig(target_coverages=(0.9, 0.5), warning_bands={}, degradation={})


@pytest.mark.parametrize("bad", [0.0, 1.5, -0.1])
def test_eval_config_rejects_out_of_range_coverage(bad):
    with pytest.raises(ConfigError, match="coverages"):
        EvalConfig(target_coverages=(bad,), warning_bands={}, degradation={})


def test_unknown_config_name():
    with pytest.raises(ConfigError, match="unknown config"):
        load_config("nonsense")


def test_missing_config_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        DatasetConfig.load(tmp_path / "absent.yaml")


def test_malformed_yaml(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("just a string", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        DatasetConfig.load(path)


def test_missing_required_key_reports_the_key(tmp_path):
    path = tmp_path / "d.yaml"
    payload = {
        "source": {},
        "metadata": {},
        "splits": {},
        "image": {"size": 32, "cache_size": 64},
        "classes": ["A___healthy", "B___healthy"],
    }
    path.write_text(yaml.safe_dump(payload), "utf-8")
    with pytest.raises(ConfigError, match="repo_id"):
        DatasetConfig.load(path)


# --- The configs actually shipped with the project ---------------------------
def test_shipped_dataset_config_loads_and_has_12_classes():
    config = DatasetConfig.load(CONFIG_DIR / "dataset.yaml")
    assert config.num_classes == 12
    assert config.repo_id == "mohanty/PlantVillage"
    assert config.license == "CC-BY-SA-3.0"
    assert config.split_policy == "leak_free"


def test_shipped_dataset_config_has_4_crops_three_states_each():
    config = DatasetConfig.load(CONFIG_DIR / "dataset.yaml")
    crops = {}
    for name in config.classes:
        crop = name.split("___", 1)[0]
        crops[crop] = crops.get(crop, 0) + 1
    assert len(crops) == 4
    assert set(crops.values()) == {3}, f"expected 3 classes per crop, got {crops}"


def test_shipped_dataset_config_includes_the_confusable_blight_pairs():
    config = DatasetConfig.load(CONFIG_DIR / "dataset.yaml")
    assert "Potato___Early_blight" in config.classes
    assert "Potato___Late_blight" in config.classes
    assert "Tomato___Early_blight" in config.classes
    assert "Tomato___Late_blight" in config.classes


def test_shipped_model_and_train_and_eval_configs_load():
    model = ModelConfig.load(CONFIG_DIR / "model.yaml")
    train = TrainConfig.load(CONFIG_DIR / "train.yaml")
    evaluation = EvalConfig.load(CONFIG_DIR / "eval.yaml")
    assert model.backbone == "mobilenet_v3_small"
    assert train.total_epochs > 0
    assert evaluation.target_coverages == tuple(sorted(evaluation.target_coverages))
