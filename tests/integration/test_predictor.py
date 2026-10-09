"""Integration tests for the Predictor and the confidence policy."""

from __future__ import annotations

import pytest
from tests.integration.conftest import TEST_CLASSES

from fieldlens.inference.predictor import (
    HIGH,
    LOW,
    MODERATE,
    ConfidencePolicy,
    PredictionError,
    Predictor,
)


@pytest.fixture
def predictor(tiny_checkpoint):
    path, classes = tiny_checkpoint
    return Predictor.from_checkpoint(path, dataset_classes=classes, device="cpu")


def test_prediction_is_a_valid_probability_distribution(predictor, saved_png):
    result = predictor.predict(saved_png)
    total = sum(prob for _name, prob in result.top_k)
    assert 0.0 <= total <= 1.0 + 1e-6
    assert 0.0 <= result.confidence <= 1.0
    assert result.num_classes == len(TEST_CLASSES)


def test_prediction_fields_are_present_and_bounded(predictor, saved_png):
    result = predictor.predict(saved_png)
    assert result.label in TEST_CLASSES
    assert 0.0 <= result.margin <= 1.0
    assert 0.0 <= result.entropy <= 1.0
    assert result.warning_level in {HIGH, MODERATE, LOW}
    assert result.temperature == 1.0


def test_top_k_is_ordered_and_capped(predictor, saved_png):
    result = predictor.predict(saved_png)
    probs = [prob for _name, prob in result.top_k]
    assert probs == sorted(probs, reverse=True)
    assert len(result.top_k) == min(3, len(TEST_CLASSES))


def test_predictions_are_deterministic(predictor, saved_png):
    first = predictor.predict(saved_png).to_dict()
    second = predictor.predict(saved_png).to_dict()
    assert first["label"] == second["label"]
    assert first["confidence"] == pytest.approx(second["confidence"], abs=1e-9)
    assert first["top_k"] == second["top_k"]


def test_same_input_image_object_matches_path_input(predictor, saved_png, rgb_image):
    from_path = predictor.predict(saved_png)
    from_object = predictor.predict(rgb_image)
    assert from_path.label == from_object.label
    assert from_path.confidence == pytest.approx(from_object.confidence, abs=1e-6)


def test_in_memory_bytes_input_works(predictor, saved_png):
    result = predictor.predict(saved_png.read_bytes())
    assert result.label in TEST_CLASSES


def test_missing_image_raises_actionable_error(predictor, tmp_path):
    with pytest.raises(PredictionError, match="image not found"):
        predictor.predict(tmp_path / "nope.png")


def test_non_image_file_raises(tmp_path, predictor):
    bad = tmp_path / "not_an_image.png"
    bad.write_text("this is not an image", encoding="utf-8")
    with pytest.raises(PredictionError, match="not a readable image"):
        predictor.predict(bad)


def test_batch_prediction_matches_single(predictor, saved_png, rgb_image):
    batch = predictor.predict_batch([saved_png, rgb_image])
    assert len(batch) == 2
    singles = [predictor.predict(saved_png), predictor.predict(rgb_image)]
    for from_batch, from_single in zip(batch, singles, strict=True):
        assert from_batch.label == from_single.label
        assert from_batch.confidence == pytest.approx(from_single.confidence, abs=1e-5)


def test_batch_of_nothing_returns_empty(predictor):
    assert predictor.predict_batch([]) == []


def test_explain_returns_json_serialisable_dict(predictor, saved_png):
    import json

    payload = predictor.explain(saved_png)
    json.dumps(payload)
    assert set(payload) >= {"label", "confidence", "top_k", "margin", "entropy", "warning_level"}


# --- Confidence policy -------------------------------------------------------
def test_policy_bands_high_for_confident_clear_predictions():
    policy = ConfidencePolicy()
    assert policy.classify(0.95, 0.90, 0.05) == HIGH


def test_policy_bands_low_when_confidence_is_weak():
    policy = ConfidencePolicy()
    assert policy.classify(0.40, 0.05, 0.90) == LOW


def test_policy_bands_moderate_at_the_middle():
    policy = ConfidencePolicy()
    assert policy.classify(0.60, 0.25, 0.60) == MODERATE


def test_policy_demotes_when_the_margin_is_too_small():
    """High probability alone is not enough; the runner-up must be clear."""
    policy = ConfidencePolicy()
    assert policy.classify(0.90, 0.02, 0.10) != HIGH


def test_policy_demotes_when_entropy_is_high():
    policy = ConfidencePolicy()
    assert policy.classify(0.95, 0.90, 0.95) != HIGH


@pytest.mark.parametrize(
    ("moderate", "high"),
    [(0.9, 0.5), (0.0, 0.8), (0.6, 1.5)],
)
def test_policy_rejects_inconsistent_thresholds(moderate, high):
    with pytest.raises(PredictionError):
        ConfidencePolicy(high_confidence=high, moderate_confidence=moderate)


def test_policy_from_dict_round_trips():
    policy = ConfidencePolicy.from_dict(
        {"high_confidence": 0.7, "moderate_confidence": 0.4, "min_margin": 0.1, "max_entropy": 0.5}
    )
    assert policy.high_confidence == 0.7
    assert policy.min_margin == 0.1


def test_temperature_from_checkpoint_is_applied(tmp_path, tiny_model, saved_png):
    """A checkpoint's stored temperature must actually change reported numbers.

    Temperature > 1 must soften the distribution (lower peak confidence, higher
    entropy) relative to a temperature < 1 on identical weights.
    """
    from tests.integration.conftest import TEST_CLASSES, TEST_IMAGE_SIZE

    from fieldlens.models.checkpoint import CheckpointMetadata, save_checkpoint
    from fieldlens.models.factory import preprocessing_spec

    spec = preprocessing_spec("mobilenet_v3_small", "NONE", TEST_IMAGE_SIZE)
    results = {}
    for label, temperature in (("sharp", 0.5), ("smooth", 3.0)):
        metadata = CheckpointMetadata(
            class_names=TEST_CLASSES,
            preprocessing=spec.as_dict(),
            backbone="mobilenet_v3_small",
            image_size=TEST_IMAGE_SIZE,
            weights="NONE",
            temperature=temperature,
        )
        path = tmp_path / f"{label}.pt"
        save_checkpoint(path, tiny_model, metadata)
        predictor = Predictor.from_checkpoint(
            path, dataset_classes=TEST_CLASSES, device="cpu"
        )
        results[label] = predictor.predict(saved_png)

    assert results["smooth"].confidence < results["sharp"].confidence
    assert results["smooth"].entropy > results["sharp"].entropy
    assert results["smooth"].temperature == 3.0


def test_caveats_are_always_described(predictor):
    caveats = predictor.described_caveats()
    assert len(caveats) >= 3
    joined = " ".join(caveats).lower()
    assert "not a field-condition diagnosis" in joined
    assert "reject" in joined


# --- CLI / UI parity ---------------------------------------------------------
def test_cli_prediction_matches_predictor(predictor, saved_png, tiny_checkpoint, capsys):
    """The CLI and the library must produce identical results.

    Both go through Predictor, which is what makes train/serve skew a testable
    property rather than an assumption.
    """
    import json

    from fieldlens.cli import main

    checkpoint, classes = tiny_checkpoint
    exit_code = main(["predict", str(saved_png), "--checkpoint", str(checkpoint), "--json"])
    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)

    direct = predictor.explain(saved_png)
    assert printed["label"] == direct["label"]
    assert printed["confidence"] == pytest.approx(direct["confidence"], abs=1e-9)
    assert printed["warning_level"] == direct["warning_level"]


def test_predictor_surfaces_no_nan(predictor, saved_png):
    import math

    result = predictor.predict(saved_png)
    assert not math.isnan(result.confidence)
    assert all(not math.isnan(prob) for _name, prob in result.top_k)
