"""Unit tests for leaf identity and manifest construction."""

from __future__ import annotations

import pytest

from fieldlens.data.split import (
    ManifestError,
    apply_split_policy,
    build_manifest,
    class_name_from_path,
    default_manifest_path,
    leaf_id_from_path,
    read_split_file,
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (
            "raw/color/Tomato___healthy/a37f34f7-022d-461a-8a3d-95f5cd774e35___Mary_HL 9155.JPG",
            "mary_hl 9155",
        ),
        ("raw/color/Grape___Black_rot/x___leaf 7.jpg", "leaf 7"),
        # No separator: fall back to the bare filename so nothing is dropped.
        ("raw/color/Apple___healthy/plain.JPG", "plain"),
    ],
)
def test_leaf_id_from_path(path, expected):
    assert leaf_id_from_path(path) == expected


def test_class_name_from_path():
    assert (
        class_name_from_path("raw/color/Potato___Early_blight/file.JPG")
        == "Potato___Early_blight"
    )


def test_class_name_from_path_rejects_short_paths():
    with pytest.raises(ManifestError):
        class_name_from_path("only-one-segment")


def test_read_split_file_missing(tmp_path):
    with pytest.raises(ManifestError) as excinfo:
        read_split_file(tmp_path / "nope.txt")
    assert "fieldlens prepare" in str(excinfo.value)


def test_build_manifest_only_keeps_configured_classes(synthetic_manifest):
    assert set(synthetic_manifest.classes) == {
        "Corn_(maize)___healthy",
        "Grape___healthy",
        "Potato___Early_blight",
    }
    counts = synthetic_manifest.counts()
    # All 9 official-train images are retained; they are split between train and
    # val by leaf identity, so assert on the total rather than one side.
    assert counts["train"] + counts["val"] == 9
    assert counts["test"] == 5


def test_train_validation_split_by_leaf_not_image(dataset_config):
    """Every image of a leaf must land on the same side of the boundary."""
    manifest = build_manifest(dataset_config, policy="official")
    leaves_by_split: dict[str, set[str]] = {}
    for entry in manifest.entries:
        leaves_by_split.setdefault(entry.split, set()).add(entry.leaf_id)

    train_leaves = leaves_by_split["train"]
    val_leaves = leaves_by_split["val"]
    assert not (train_leaves & val_leaves), "validation shares a leaf with train"
    # Both images of "leaf 1" must travel together.
    leaf_1_splits = {e.split for e in manifest.entries if e.leaf_id == "leaf 1"}
    assert len(leaf_1_splits) == 1


def test_official_policy_preserves_the_leak(synthetic_manifest):
    assert synthetic_manifest.split_policy == "official"
    assert synthetic_manifest.dropped_leaky_entries == 0


def test_leak_free_policy_drops_leaky_training_images(dataset_config):
    manifest = build_manifest(dataset_config, policy="leak_free")
    assert manifest.split_policy == "leak_free"
    # "leaf 4" appears in train and test, so its single train image is dropped.
    assert manifest.dropped_leaky_entries == 1
    train_leaves = {e.leaf_id for e in manifest.entries if e.split == "train"}
    test_leaves = {e.leaf_id for e in manifest.entries if e.split == "test"}
    assert not (train_leaves & test_leaves)


def test_leak_free_policy_does_not_modify_the_test_set(synthetic_manifest):
    official_test = [e.path for e in synthetic_manifest.entries if e.split == "test"]
    leak_free = apply_split_policy(synthetic_manifest, "leak_free")
    leak_free_test = [e.path for e in leak_free.entries if e.split == "test"]
    assert official_test == leak_free_test


def test_unknown_policy_is_rejected(synthetic_manifest):
    with pytest.raises(ManifestError):
        apply_split_policy(synthetic_manifest, "whatever")


def test_manifest_round_trips_through_json(tmp_path, synthetic_manifest):
    from fieldlens.data.split import Manifest

    path = synthetic_manifest.save(tmp_path / "m.json")
    reloaded = Manifest.load(path)
    assert reloaded.counts() == synthetic_manifest.counts()
    assert reloaded.split_policy == synthetic_manifest.split_policy
    assert reloaded.classes == synthetic_manifest.classes


def test_manifest_load_missing_file(tmp_path):
    from fieldlens.data.split import Manifest

    with pytest.raises(ManifestError):
        Manifest.load(tmp_path / "absent.json")


def test_leaf_source_records_provenance(synthetic_manifest):
    sources = {e.leaf_source for e in synthetic_manifest.entries}
    # The fixture's leaf map only covers three leaves, so both paths are used.
    assert sources == {"leaf_map", "filename_tail"}


def test_manifest_path_includes_policy(dataset_config):
    dataset_config = dataset_config.__class__(
        **{**dataset_config.__dict__, "split_policy": "leak_free"}
    )
    assert "leak_free" in default_manifest_path(dataset_config).name


def test_entries_for_filters_by_split(synthetic_manifest):
    train = synthetic_manifest.entries_for("train")
    assert train and all(e.split == "train" for e in train)
