"""Integration tests for the model factory and checkpoint persistence."""

from __future__ import annotations

import pytest
from tests.integration.conftest import TEST_CLASSES, TEST_IMAGE_SIZE

from fieldlens.config import ModelConfig
from fieldlens.models.checkpoint import (
    CheckpointError,
    CheckpointMetadata,
    load_checkpoint,
    load_metadata,
    save_checkpoint,
    write_metadata_sidecar,
)
from fieldlens.models.factory import (
    ModelBuildError,
    build_model,
    count_parameters,
    freeze_backbone,
    preprocessing_spec,
    unfreeze_last_block,
)


def test_model_forwards_a_batch(tiny_model):
    import torch

    tiny_model.eval()
    with torch.inference_mode():
        out = tiny_model(torch.randn(2, 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE))
    assert out.shape == (2, len(TEST_CLASSES))


def test_head_is_replaced_for_the_requested_class_count():
    import torch

    config = ModelConfig("mobilenet_v3_small", "NONE", 32, 0.0)
    for num_classes in (2, 5, 12):
        model = build_model(config, num_classes, pretrained=False)
        model.eval()
        with torch.inference_mode():
            out = model(torch.randn(1, 3, 32, 32))
        assert out.shape == (1, num_classes)


def test_dropout_does_not_change_the_state_dict_layout():
    """Regression: an inserted Dropout shifted classifier indices.

    A checkpoint saved with dropout enabled could not be reloaded with dropout
    disabled, because the head's Linear moved from index 4 to 3. The layout must
    depend only on the backbone.
    """
    with_dropout = build_model(
        ModelConfig("mobilenet_v3_small", "NONE", 32, 0.3), 4, pretrained=False
    )
    without_dropout = build_model(
        ModelConfig("mobilenet_v3_small", "NONE", 32, 0.0), 4, pretrained=False
    )
    assert set(with_dropout.state_dict()) == set(without_dropout.state_dict())
    # The full head must still round-trip in both directions.
    with_dropout.load_state_dict(without_dropout.state_dict())
    without_dropout.load_state_dict(with_dropout.state_dict())


def test_classifier_head_keeps_its_original_depth():
    """The head must not grow a layer; only the final Linear is swapped."""
    reference = build_model(
        ModelConfig("mobilenet_v3_small", "NONE", 32, 0.0), 1000, pretrained=False
    )
    replaced = build_model(
        ModelConfig("mobilenet_v3_small", "NONE", 32, 0.2), 12, pretrained=False
    )
    assert len(reference.classifier) == len(replaced.classifier)


def test_parameter_count_is_small_enough_to_commit(tiny_model):
    counts = count_parameters(tiny_model)
    assert 1_000_000 < counts["total"] < 5_000_000, counts


def test_freeze_backbone_leaves_only_the_head_trainable(tiny_model):
    freeze_backbone(tiny_model)
    trainable = [n for n, p in tiny_model.named_parameters() if p.requires_grad]
    assert trainable, "head must remain trainable"
    assert all(n.startswith("classifier") for n in trainable)


def test_unfreeze_last_block_enables_more_parameters(tiny_model):
    freeze_backbone(tiny_model)
    frozen = sum(p.numel() for p in tiny_model.parameters() if p.requires_grad)
    unfrozen = unfreeze_last_block(tiny_model)
    assert unfrozen > frozen


def test_unsupported_backbone_is_rejected():
    config = ModelConfig("resnet9000", "NONE", 32, 0.0)
    with pytest.raises(ModelBuildError, match="unsupported backbone"):
        build_model(config, 3, pretrained=False)


def test_unknown_weights_name_is_rejected():
    config = ModelConfig("mobilenet_v3_small", "NOT_A_WEIGHT_SET", 32, 0.0)
    with pytest.raises(ModelBuildError, match="unknown weights"):
        build_model(config, 3)


def test_num_classes_below_two_is_rejected():
    config = ModelConfig("mobilenet_v3_small", "NONE", 32, 0.0)
    with pytest.raises(ModelBuildError, match="num_classes"):
        build_model(config, 1, pretrained=False)


def test_preprocessing_spec_reports_the_version():
    spec = preprocessing_spec("mobilenet_v3_small", "NONE", 224)
    assert spec.image_size == 224
    assert len(spec.mean) == 3 and len(spec.std) == 3
    assert "untrained" in spec.version


def test_checkpoint_round_trip_preserves_class_order(tiny_checkpoint):
    import torch

    path, classes = tiny_checkpoint
    model, metadata = load_checkpoint(path, device="cpu")
    assert tuple(metadata.class_names) == tuple(classes)
    assert metadata.class_index()["Grape___healthy"] == 1
    assert model.training is False, "loaded models must be in eval mode"
    with torch.inference_mode():
        out = model(torch.randn(1, 3, TEST_IMAGE_SIZE, TEST_IMAGE_SIZE))
    assert out.shape == (1, len(classes))


def test_reloaded_state_dict_matches_the_saved_one(tiny_checkpoint):
    import torch

    path, _ = tiny_checkpoint
    original = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    model, _ = load_checkpoint(path, device="cpu")
    reloaded = model.state_dict()
    assert set(original) == set(reloaded)
    for key in original:
        assert torch.equal(original[key], reloaded[key]), f"mismatch in {key}"


def test_loader_rejects_a_mismatched_class_list(tiny_checkpoint):
    """A stale checkpoint must fail loudly, never mislabel silently."""
    path, _ = tiny_checkpoint
    with pytest.raises(CheckpointError, match="does not match"):
        load_checkpoint(path, dataset_classes=("Something___else",), device="cpu")


def test_loader_accepts_the_matching_class_list(tiny_checkpoint):
    path, classes = tiny_checkpoint
    model, _ = load_checkpoint(path, dataset_classes=classes, device="cpu")
    assert model is not None


def test_metadata_only_load_does_not_build_the_model(tiny_checkpoint):
    path, classes = tiny_checkpoint
    metadata = load_metadata(path)
    assert tuple(metadata.class_names) == tuple(classes)


def test_missing_checkpoint_gives_actionable_message(tmp_path):
    with pytest.raises(CheckpointError, match="fieldlens train"):
        load_metadata(tmp_path / "absent.pt")


def test_checkpoint_metadata_rejects_empty_classes():
    with pytest.raises(CheckpointError, match="empty class list"):
        CheckpointMetadata(class_names=(), preprocessing={}, backbone="m", image_size=32)


def test_checkpoint_metadata_rejects_duplicate_classes():
    with pytest.raises(CheckpointError, match="duplicate"):
        CheckpointMetadata(
            class_names=("A___healthy", "A___healthy"),
            preprocessing={},
            backbone="m",
            image_size=32,
        )


def test_checkpoint_metadata_rejects_non_positive_temperature():
    with pytest.raises(CheckpointError, match="temperature"):
        CheckpointMetadata(
            class_names=TEST_CLASSES,
            preprocessing={},
            backbone="m",
            image_size=32,
            temperature=0.0,
        )


def test_unsupported_schema_version_is_rejected():
    with pytest.raises(CheckpointError, match="schema version"):
        CheckpointMetadata.from_dict(
            {
                "schema_version": 999,
                "class_names": list(TEST_CLASSES),
                "preprocessing": {"image_size": 32},
                "image_size": 32,
            }
        )


def test_metadata_sidecar_is_written(tiny_checkpoint, tmp_path):
    path, _ = tiny_checkpoint
    metadata = load_metadata(path)
    sidecar = write_metadata_sidecar(tmp_path / "copy.pt", metadata)
    assert sidecar.exists()
    assert "class_names" in sidecar.read_text(encoding="utf-8")


def test_save_and_load_preserves_metrics_and_provenance(tmp_path, tiny_model):
    from fieldlens.models.factory import preprocessing_spec

    spec = preprocessing_spec("mobilenet_v3_small", "NONE", 32)
    metadata = CheckpointMetadata(
        class_names=TEST_CLASSES,
        preprocessing=spec.as_dict(),
        backbone="mobilenet_v3_small",
        image_size=32,
        temperature=1.75,
        metrics={"test_macro_f1": 0.5},
        git_sha="deadbeef",
        seed=7,
    )
    save_checkpoint(tmp_path / "c.pt", tiny_model, metadata)
    loaded = load_metadata(tmp_path / "c.pt")
    assert loaded.temperature == 1.75
    assert loaded.metrics["test_macro_f1"] == 0.5
    assert loaded.git_sha == "deadbeef"
    assert loaded.seed == 7
