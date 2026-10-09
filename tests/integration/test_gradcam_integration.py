"""Grad-CAM end-to-end against a real (untrained) torch model.

Uses the synthetic checkpoint so no dataset or network access is involved. The
properties asserted are the ones a diagnostic aid must satisfy: bounded output,
determinism, no leaked gradient state, and an overlay that matches the input.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from fieldlens.explain.gradcam import GradCAMExplainer
from fieldlens.inference.predictor import Predictor


@pytest.fixture
def tiny_predictor(tiny_checkpoint):
    path, class_names = tiny_checkpoint
    return Predictor.from_checkpoint(path), class_names


def _image(tmp_path, size=(96, 96)):
    array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    array[:, :, 0] = np.linspace(0, 255, size[0], dtype=np.uint8)
    array[:, :, 1] = np.linspace(255, 0, size[1], dtype=np.uint8)[:, None]
    array[20:40, 30:60, 2] = 200
    path = tmp_path / "leaf.png"
    Image.fromarray(array).save(path)
    return path


def test_explain_returns_a_bounded_heatmap(tiny_predictor, tmp_path) -> None:
    predictor, class_names = tiny_predictor
    with GradCAMExplainer(predictor) as explainer:
        result = explainer.explain(_image(tmp_path))

    assert result.heatmap.ndim == 2
    assert result.heatmap.shape[0] == result.heatmap.shape[1], "CAM should be square"
    assert np.isfinite(result.heatmap).all()
    assert float(result.heatmap.min()) >= 0.0
    assert float(result.heatmap.max()) <= 1.0
    assert result.heatmap.max() == pytest.approx(1.0, abs=1e-6), (
        "a non-constant CAM is normalised so its peak is exactly 1"
    )
    assert result.label in class_names
    assert 0.0 <= result.confidence <= 1.0
    assert result.target_class_index == class_names.index(result.label)


def test_explain_is_deterministic(tiny_predictor, tmp_path) -> None:
    predictor, _ = tiny_predictor
    image = _image(tmp_path)
    with GradCAMExplainer(predictor) as explainer:
        first = explainer.explain(image)
        second = explainer.explain(image)
    assert np.array_equal(first.heatmap, second.heatmap)


def test_explaining_a_specific_class_overrides_the_prediction(
    tiny_predictor, tmp_path
) -> None:
    predictor, class_names = tiny_predictor
    target = class_names.index(class_names[-1])
    with GradCAMExplainer(predictor) as explainer:
        result = explainer.explain(_image(tmp_path), target_class=target)
    assert result.target_class_index == target
    assert result.label == class_names[-1]


def test_explaining_each_class_yields_different_maps(tiny_predictor, tmp_path) -> None:
    """If every class produced the same map, the map would carry no information."""
    predictor, class_names = tiny_predictor
    image = _image(tmp_path)
    maps = []
    with GradCAMExplainer(predictor) as explainer:
        for index in (0, 1, 2):
            maps.append(explainer.explain(image, target_class=index).heatmap)
    assert not np.array_equal(maps[0], maps[1])
    assert not np.array_equal(maps[1], maps[2])


def test_overlay_matches_the_input_image_size(tiny_predictor, tmp_path) -> None:
    predictor, _ = tiny_predictor
    image = _image(tmp_path)
    with GradCAMExplainer(predictor) as explainer:
        result = explainer.explain(image)
    overlay = result.overlay_on(predictor.load_rgb(image))
    expected = predictor.load_rgb(image).size
    assert overlay.size == expected
    assert overlay.size != (result.heatmap.shape[1], result.heatmap.shape[0]), (
        "regression: the overlay must not be thumbnail-sized"
    )


def test_hooks_are_removed_on_close(tiny_predictor, tmp_path) -> None:
    predictor, _ = tiny_predictor
    explainer = GradCAMExplainer(predictor)
    assert len(explainer._handles) == 2
    explainer.explain(_image(tmp_path))
    explainer.close()
    assert explainer._handles == []
    # Explaining again must fail loudly rather than reuse stale captures.
    with pytest.raises(RuntimeError, match="closed"):
        explainer.explain(_image(tmp_path))


def test_explain_leaves_no_gradient_state_on_the_model(tiny_predictor, tmp_path) -> None:
    """The explainer must be safe to use next to normal inference."""
    predictor, _ = tiny_predictor
    with GradCAMExplainer(predictor) as explainer:
        explainer.explain(_image(tmp_path))
    assert all(p.grad is None for p in predictor.model.parameters())


def test_peak_region_is_inside_the_map(tiny_predictor, tmp_path) -> None:
    predictor, _ = tiny_predictor
    with GradCAMExplainer(predictor) as explainer:
        result = explainer.explain(_image(tmp_path))
    row, column = result.peak_region()
    assert 0 <= row < result.heatmap.shape[0]
    assert 0 <= column < result.heatmap.shape[1]


def test_confidence_matches_the_predictor(tiny_predictor, tmp_path) -> None:
    """The map's reported confidence must agree with the serving path."""
    predictor, _ = tiny_predictor
    image = _image(tmp_path)
    with GradCAMExplainer(predictor) as explainer:
        result = explainer.explain(image)
    prediction = predictor.predict(image)
    assert result.label == prediction.label
    assert result.confidence == pytest.approx(prediction.confidence, abs=1e-5)


def test_caveats_are_declared(tiny_predictor) -> None:
    predictor, _ = tiny_predictor
    with GradCAMExplainer(predictor) as explainer:
        caveats = explainer.described_caveats()
    assert caveats
    assert any("not lesion detection" in c or "not evidence of causation" in c for c in caveats)
