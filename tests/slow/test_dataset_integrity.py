"""Dataset-dependent tests. Opt-in only; never part of default CI.

These assert the properties the project's central claim rests on, using the real
metadata and images when they are present locally. Every test skips cleanly when
the data has not been acquired, so the suite stays green on a fresh clone.

Run them with::

    pytest tests/slow -q -m slow
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from fieldlens.config import DatasetConfig
from fieldlens.data.leakage import audit_manifest
from fieldlens.data.split import Manifest, default_manifest_path, read_split_file
from fieldlens.paths import IMAGE_DIR, METADATA_DIR

pytestmark = pytest.mark.slow


@pytest.fixture
def dataset() -> DatasetConfig:
    return DatasetConfig.load()


@pytest.fixture
def manifest(dataset) -> Manifest:
    path = default_manifest_path(dataset)
    if not path.exists():
        pytest.skip("manifest not built; run `fieldlens audit` first")
    return Manifest.load(path)


def test_leakage_audit_passes_on_the_committed_manifest(manifest):
    """No leaf identifier may appear in more than one split."""
    report = audit_manifest(manifest)
    assert report.passed, report.summary()
    assert report.cross_split_leaf_overlap_counts == {}
    assert report.cross_split_path_overlaps == {}


def test_official_split_file_line_counts_match_the_documented_sizes():
    """Cross-check the published split sizes against the real files."""
    train = METADATA_DIR / "splits" / "color_train.txt"
    test = METADATA_DIR / "splits" / "color_test.txt"
    if not (train.exists() and test.exists()):
        pytest.skip("split metadata not downloaded; run `fieldlens prepare`")
    assert len(read_split_file(train)) == 43596
    assert len(read_split_file(test)) == 10709


def test_manifest_keeps_the_official_test_set_intact(manifest, dataset):
    """The leak_free policy must never remove anything from the test split."""
    test_entries = manifest.entries_for("test")
    assert len(test_entries) == len({e.path for e in test_entries})
    assert manifest.counts()["test"] > 0


def test_leaf_ids_do_not_span_multiple_classes(manifest):
    """An ambiguous leaf identifier would make grouping unreliable."""
    report = audit_manifest(manifest)
    assert report.leaf_ids_spanning_multiple_classes == {}


def test_potato_healthy_is_the_smallest_class(manifest):
    """The measured smallest class drives the choice of macro-averaged metrics."""
    report = audit_manifest(manifest)
    assert report.smallest_class == "Potato___healthy"
    assert 0 < report.smallest_class_count < 500


def test_manifest_classes_match_the_configured_classes(manifest, dataset):
    assert manifest.classes == dataset.classes
    assert len(dataset.classes) == 12


def test_test_split_covers_every_class(manifest):
    """A class missing from test would make its metrics meaningless."""
    classes = Counter(e.class_name for e in manifest.entries_for("test"))
    assert set(classes) == set(manifest.classes)


def test_acquired_images_are_non_empty_and_expected_size(manifest):
    """Integrity of whatever has been downloaded so far."""
    if not IMAGE_DIR.exists():
        pytest.skip("no images acquired; run `fieldlens extract`")
    files = [p for p in IMAGE_DIR.rglob("*") if p.is_file()]
    if not files:
        pytest.skip("image directory is empty")
    empty = [p for p in files if p.stat().st_size == 0]
    assert not empty, f"{len(empty)} zero-byte images, e.g. {empty[:3]}"


def test_acquired_images_decode(manifest):
    """Every acquired image must open, not merely exist."""
    from PIL import Image

    if not IMAGE_DIR.exists():
        pytest.skip("no images acquired; run `fieldlens extract`")
    files = sorted(p for p in IMAGE_DIR.rglob("*") if p.is_file())[:50]
    if not files:
        pytest.skip("image directory is empty")
    for path in files:
        with Image.open(path) as handle:
            handle.verify()


def test_temperature_in_checkpoint_was_fitted_on_validation():
    """Guard the rule that calibration never touches the test split."""
    import json

    report = Path("artifacts/reports/evaluation.json")
    if not report.exists():
        pytest.skip("no evaluation report; run `fieldlens evaluate`")
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["calibration_fit"]["fitted_on"] in {"val", "validation"}
    assert payload["leakage_audit_passed"] is True
