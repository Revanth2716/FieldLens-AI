"""Unit tests for the leakage audit.

The audit is the project's central honesty mechanism, so it is tested both for
catching a leak and for clearing a clean split.
"""

from __future__ import annotations

import pytest

from fieldlens.data.leakage import assert_no_leakage, audit_manifest
from fieldlens.data.split import apply_split_policy


def test_audit_detects_the_injected_cross_split_leaf(synthetic_manifest):
    """The fixture plants 'leaf 4' in both train and test."""
    report = audit_manifest(synthetic_manifest)
    assert report.passed is False
    assert report.cross_split_leaf_overlap_counts.get("test|train") == 1
    assert "leaf 4" in report.cross_split_leaf_overlaps["test|train"]


def test_audit_passes_after_applying_leak_free_policy(synthetic_manifest):
    report = audit_manifest(apply_split_policy(synthetic_manifest, "leak_free"))
    assert report.passed is True
    assert report.cross_split_leaf_overlap_counts == {}


def test_no_cross_split_path_overlaps(synthetic_manifest):
    report = audit_manifest(synthetic_manifest)
    assert report.cross_split_path_overlaps == {}


def test_report_records_measured_counts(synthetic_manifest):
    report = audit_manifest(synthetic_manifest)
    assert report.split_counts["test"] == 5
    assert report.leaf_counts["test"] == 5
    assert set(report.class_counts) == {"train", "val", "test"}


def test_smallest_class_is_measured_not_assumed(synthetic_manifest):
    report = audit_manifest(synthetic_manifest)
    assert report.smallest_class
    # Grape holds 2 train-side images plus 1 test image in the fixture, so the
    # measured smallest-class total is 3.
    assert report.smallest_class_count == 3


def test_leaf_id_source_counts_are_recorded(synthetic_manifest):
    report = audit_manifest(synthetic_manifest)
    assert report.leaf_source_counts.get("filename_tail", 0) > 0
    assert report.leaf_source_counts.get("leaf_map", 0) > 0


def test_summary_mentions_failure_and_result(synthetic_manifest):
    text = audit_manifest(synthetic_manifest).summary()
    assert "CROSS-SPLIT LEAF OVERLAPS" in text
    assert "RESULT: FAIL" in text


def test_summary_reports_pass_for_a_clean_split(synthetic_manifest):
    text = audit_manifest(apply_split_policy(synthetic_manifest, "leak_free")).summary()
    assert "RESULT: PASS" in text


def test_assert_no_leakage_raises_on_failure(synthetic_manifest):
    report = audit_manifest(synthetic_manifest)
    with pytest.raises(AssertionError) as excinfo:
        assert_no_leakage(report, context="training")
    assert "training" in str(excinfo.value)
    assert "inflated" in str(excinfo.value)


def test_assert_no_leakage_silent_when_clean(synthetic_manifest):
    report = audit_manifest(apply_split_policy(synthetic_manifest, "leak_free"))
    assert_no_leakage(report)  # must not raise


def test_report_json_is_serialisable(synthetic_manifest):
    import json

    payload = audit_manifest(synthetic_manifest).to_json()
    text = json.dumps(payload)
    assert '"passed": false' in text
    assert payload["cross_split_leaf_overlap_counts"]["test|train"] == 1


def test_policy_and_dropped_counts_are_reported(synthetic_manifest):
    report = audit_manifest(apply_split_policy(synthetic_manifest, "leak_free"))
    assert report.split_policy == "leak_free"
    assert report.dropped_leaky_entries == 1
