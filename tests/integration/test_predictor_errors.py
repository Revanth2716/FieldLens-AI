"""Bad inputs must produce guidance, not a traceback.

The CLI and the Streamlit page both funnel through ``Predictor``, so an uncaught
``OSError`` here becomes a stack trace on screen. A directory path used to raise
``PermissionError`` and a truncated file raised an uncaught ``OSError``; both are
pinned below.
"""

from __future__ import annotations

import pytest

from fieldlens.inference.predictor import PredictionError, Predictor


@pytest.fixture
def tiny_predictor(tiny_checkpoint):
    path, _ = tiny_checkpoint
    return Predictor.from_checkpoint(path)


def test_missing_file_is_reported_clearly(tiny_predictor, tmp_path) -> None:
    with pytest.raises(PredictionError, match="image not found"):
        tiny_predictor.predict(tmp_path / "nope.png")


def test_directory_is_rejected_before_reaching_pillow(tiny_predictor, tmp_path) -> None:
    with pytest.raises(PredictionError, match="directory"):
        tiny_predictor.predict(tmp_path)


def test_non_image_file_on_disk_is_rejected(tiny_predictor, tmp_path) -> None:
    fake = tmp_path / "not-an-image.png"
    fake.write_bytes(b"this is plainly not a PNG")
    with pytest.raises(PredictionError, match="not a readable image file"):
        tiny_predictor.predict(fake)


def test_truncated_image_is_rejected(tiny_predictor, tmp_path, rgb_image) -> None:
    """A half-written JPEG raises OSError, which must be converted too."""
    truncated = tmp_path / "half.png"
    rgb_image.save(truncated)
    truncated.write_bytes(truncated.read_bytes()[:40])
    with pytest.raises(PredictionError):
        tiny_predictor.predict(truncated)


def test_garbage_bytes_are_rejected(tiny_predictor) -> None:
    with pytest.raises(PredictionError, match="not a readable image"):
        tiny_predictor.predict(b"\x00\x01\x02 not an image")


def test_unsupported_input_type_is_rejected(tiny_predictor) -> None:
    with pytest.raises(PredictionError, match="unsupported image input type"):
        tiny_predictor.predict(12345)


def test_valid_image_still_works(tiny_predictor, saved_png) -> None:
    """The hardening must not reject legitimate input."""
    result = tiny_predictor.predict(saved_png)
    assert 0.0 <= result.confidence <= 1.0
    assert result.label
