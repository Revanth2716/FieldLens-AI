"""Sweep configuration parsing.

The reliability sweeps are configured in ``configs/eval.yaml`` with named keys
(``brightness: {gains: [...]}``). A parser that silently accepted the mapping and
iterated its *keys* produced one bogus point labelled ``"gains"``, so the accepted
shapes are pinned here.
"""

from __future__ import annotations

import pytest

from fieldlens.eval.robustness import DEGRADATIONS, levels_for, sweep_metadata


def test_bare_list_is_accepted() -> None:
    assert levels_for("brightness", [0.3, 1.0]) == [0.3, 1.0]


def test_named_mapping_is_accepted() -> None:
    assert levels_for("brightness", {"gains": [0.5, 1.5]}) == [0.5, 1.5]
    assert levels_for("resolution", {"sizes": [32, 224]}) == [32.0, 224.0]
    assert levels_for("jpeg", {"qualities": [95, 5]}) == [95.0, 5.0]
    assert levels_for("blur", {"sigmas": [0.5, 4.0]}) == [0.5, 4.0]


def test_single_unknown_key_falls_back_to_its_value() -> None:
    """A one-key mapping is unambiguous, so it is read even if the key is novel."""
    assert levels_for("brightness", {"whatever": [1.0, 2.0]}) == [1.0, 2.0]


def test_levels_are_floats() -> None:
    assert all(isinstance(v, float) for v in levels_for("jpeg", {"qualities": [95]}))


def test_multi_key_mapping_without_a_known_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="single named list"):
        levels_for("brightness", {"a": [1.0], "b": [2.0]})


def test_scalar_and_none_are_rejected() -> None:
    with pytest.raises(ValueError, match="must be a list or mapping"):
        levels_for("brightness", 1.0)
    with pytest.raises(ValueError, match="must be a list or mapping"):
        levels_for("brightness", None)


def test_every_library_sweep_has_a_transformation_description() -> None:
    """A report entry must never be separable from what it measured."""
    for name in ("brightness", "resolution", "jpeg", "blur"):
        assert name in DEGRADATIONS
        assert DEGRADATIONS[name].strip()


def test_sweep_metadata_is_a_copy() -> None:
    metadata = sweep_metadata()
    metadata["brightness"] = "tampered"
    assert DEGRADATIONS["brightness"] != "tampered"
