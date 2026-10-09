"""Unit tests for cache redirection and the disk guards.

The disk guards are load-bearing on this machine (C: has ~10 GB free), so the
refusal path is tested, not just the happy path.
"""

from __future__ import annotations

import pytest

from fieldlens import env


def test_temp_vars_are_forced_onto_the_project_drive():
    """Windows always defines TMP/TEMP; they must still be redirected.

    Regression test: an earlier revision respected pre-existing values, which
    left temp files on the nearly-full system drive.
    """
    report = env.cache_redirection_report()
    for name in ("TMP", "TEMP", "TMPDIR"):
        assert report[name]["on_system_drive"] is False, (
            f"{name} still resolves onto the system drive: {report[name]}"
        )
        assert report[name]["actual"] == report[name]["expected"]


def test_no_cache_resolves_onto_the_system_drive():
    report = env.cache_redirection_report()
    offenders = {name: info for name, info in report.items() if info["on_system_drive"]}
    assert not offenders, f"caches on the system drive: {offenders}"


def test_redirect_points_inside_the_project_cache_root():
    report = env.cache_redirection_report()
    for name, info in report.items():
        assert str(env.CACHE_ROOT) in str(info["expected"]), f"{name} not under cache root"


def test_environment_report_reports_system_drive_caches():
    report = env.environment_report()
    assert "caches_on_system_drive" in report
    assert report["caches_on_system_drive"] == []


def test_format_bytes_units():
    assert env.format_bytes(512) == "512 B"
    assert env.format_bytes(1024).endswith("KB")
    assert env.format_bytes(5 * 1024**3).endswith("GB")


def test_free_bytes_is_positive():
    assert env.free_bytes(env.PROJECT_ROOT) > 0


def test_required_bytes_applies_headroom_and_reserve():
    transfer = 100 * 1024**2
    expected = int(transfer * env.HEADROOM_MULTIPLIER) + env.RESERVE_BYTES
    assert env.required_bytes(transfer) == expected


def test_guard_refuses_an_impossible_transfer():
    """A transfer larger than the drive must be refused, not attempted."""
    with pytest.raises(env.InsufficientSpaceError) as excinfo:
        env.guard_project_space(1_000_000 * 1024**2, "absurd transfer")
    message = str(excinfo.value)
    assert "disk guard" in message
    assert "absurd transfer" in message


def test_guard_allows_a_small_transfer():
    have = env.guard_project_space(1024, "tiny")
    assert have > 0


def test_guard_message_states_requirement_and_availability():
    with pytest.raises(env.InsufficientSpaceError) as excinfo:
        env.guard_project_space(1_000_000 * 1024**2, "huge")
    message = str(excinfo.value)
    assert "needs" in message and "available" in message


def test_guard_c_drive_trips_when_strict(monkeypatch):
    """Installs opt into the strict system-drive check and must be refused."""
    monkeypatch.setattr(env, "free_bytes", lambda path: 1)
    with pytest.raises(env.InsufficientSpaceError):
        env.guard_c_drive(min_free=10 * env.GB, strict=True)


def test_guard_c_drive_is_advisory_by_default(monkeypatch):
    """Project-local work must not be blocked by a system-drive condition.

    Caches and temp are redirected to the project drive, so these operations
    cannot worsen C:; refusing them would stall the entire project instead.
    """
    monkeypatch.setattr(env, "free_bytes", lambda path: 1)
    assert env.guard_c_drive(min_free=10 * env.GB) == 1


def test_system_drive_warning_is_reported_but_not_raised(monkeypatch):
    monkeypatch.setattr(env, "free_bytes", lambda path: 1)
    warning = env.system_drive_warning(min_free=10 * env.GB)
    assert warning is not None
    assert "advisory floor" in warning


def test_guard_still_refuses_an_impossible_project_transfer(monkeypatch):
    """The project-drive guard remains strictly enforced."""
    with pytest.raises(env.InsufficientSpaceError):
        env.guard_project_space(1_000_000 * 1024**2, "absurd")
