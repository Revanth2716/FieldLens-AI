"""Grad-CAM overlay rendering.

Unit-level and torch-free: the explainer itself is exercised in the integration
suite, but the compositing maths is pure NumPy and is where a real defect lived —
``overlay_heatmap`` used to resize the *image down to* the 7x7 CAM and return a
222-byte thumbnail instead of a full-size overlay. That is pinned here.
"""

from __future__ import annotations

import numpy as np
import pytest

from fieldlens.explain.gradcam import (
    GradCAMResult,
    default_target_layer,
    overlay_heatmap,
)


def _heatmap(size: int = 7, fill: float = 0.0) -> np.ndarray:
    return np.full((size, size), fill, dtype=np.float32)


def test_overlay_keeps_the_image_size_not_the_cam_size(rgb_image) -> None:
    """Regression: a coarse CAM must be upsampled, never the image downsampled."""
    overlay = overlay_heatmap(rgb_image, _heatmap(7))
    assert overlay.size == rgb_image.size == (128, 128)
    assert overlay.mode == "RGB"


def test_overlay_handles_non_square_images(rgb_image) -> None:
    wide = rgb_image.resize((200, 100))
    assert overlay_heatmap(wide, _heatmap(7)).size == (200, 100)


def test_overlay_output_is_a_dense_image(rgb_image) -> None:
    overlay = overlay_heatmap(rgb_image, np.linspace(0, 1, 49, dtype=np.float32).reshape(7, 7))
    colours = len(np.unique(np.asarray(overlay).reshape(-1, 3), axis=0))
    assert colours > 1000, "a smooth overlay should not collapse to a few colours"


def test_zero_alpha_returns_the_base_image(rgb_image) -> None:
    overlay = overlay_heatmap(rgb_image, _heatmap(7), alpha=0.0)
    assert np.array_equal(np.asarray(overlay), np.asarray(rgb_image.convert("RGB")))


def test_full_alpha_is_the_pure_colour_ramp(rgb_image) -> None:
    """alpha=1 makes the ramp directly observable: 0 is blue, 1 is red."""
    cold = np.asarray(overlay_heatmap(rgb_image, _heatmap(7, 0.0), alpha=1.0))
    hot = np.asarray(overlay_heatmap(rgb_image, _heatmap(7, 1.0), alpha=1.0))
    assert tuple(cold[0, 0]) == (0, 0, 255)
    assert tuple(hot[0, 0]) == (255, 0, 0)


def test_out_of_range_heatmaps_are_clipped(rgb_image) -> None:
    overlay = overlay_heatmap(rgb_image, _heatmap(7, 5.0), alpha=1.0)
    assert tuple(np.asarray(overlay)[0, 0]) == (255, 0, 0)


def test_heatmap_matching_the_image_size_is_used_as_is(rgb_image) -> None:
    """An already-sized CAM must not be resampled twice (which would blur it)."""
    square = _heatmap(128, 0.0)
    assert tuple(np.asarray(overlay_heatmap(rgb_image, square, alpha=1.0))[0, 0]) == (
        0,
        0,
        255,
    )


def test_result_overlay_on_delegates_to_the_same_renderer(rgb_image) -> None:
    result = GradCAMResult(
        heatmap=_heatmap(7, 1.0),
        label="Potato___healthy",
        confidence=0.9,
        target_class_index=2,
    )
    overlay = result.overlay_on(rgb_image)
    assert overlay.size == rgb_image.size


def test_peak_region_locates_the_maximum() -> None:
    heatmap = _heatmap(4, 0.0)
    heatmap[1, 3] = 1.0
    result = GradCAMResult(
        heatmap=heatmap, label="x", confidence=0.5, target_class_index=0
    )
    assert result.peak_region() == (1, 3)


def test_default_target_layer_requires_a_convolutional_backbone() -> None:
    class NoFeatures:
        pass

    with pytest.raises(ValueError, match="features"):
        default_target_layer(NoFeatures())

    class EmptyFeatures:
        features: list = []

    with pytest.raises(ValueError, match="features"):
        default_target_layer(EmptyFeatures())


def test_default_target_layer_picks_the_last_block() -> None:
    """Only the attribute layout matters here, so no torch import is needed."""

    class WithFeatures:
        features = ["block-one", "block-two"]

    assert default_target_layer(WithFeatures()) == "block-two"
