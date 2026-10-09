"""Environment redirection and resource guards for FieldLens AI.

Import this module **before** torch, matplotlib, huggingface_hub or any library
that resolves a user-level cache directory. It has two jobs:

1. Redirect every cache and temporary directory onto the project drive, so the
   nearly-full ``C:`` drive cannot be filled by wheels, datasets or temp files.
2. Provide disk guards that *refuse* to start a write-heavy operation when
   there is not enough space, rather than warning and continuing.

This module is deliberately stdlib-only so it can run before any dependency is
installed.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

GB = 1024**3
MB = 1024**2

#: Repository root (``src/fieldlens/env.py`` -> project root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Everything cache-like lives here, on the project drive.
CACHE_ROOT = PROJECT_ROOT / ".cache"

#: Refuse any operation that would leave C: below this.
MIN_C_FREE_BYTES = 10 * GB

#: Keep this much room free on the project drive after an operation.
RESERVE_BYTES = 20 * GB

#: Assume transfers can transiently need this multiple of their nominal size.
HEADROOM_MULTIPLIER = 1.5

#: Variables that must be redirected even when the environment already sets them.
#: Windows always defines TMP and TEMP, so a "respect the existing value" rule
#: would leave package-manager and wheel-unpacking temp files on the nearly-full
#: system drive. These are forced onto the project drive.
FORCE_REDIRECT = frozenset({"TMP", "TEMP", "TMPDIR", "MPLCONFIGDIR"})


class InsufficientSpaceError(RuntimeError):
    """Raised when a guarded operation must not start or continue."""


def _cache_env() -> dict[str, Path]:
    """Environment variables redirected onto the project drive."""
    tmp = CACHE_ROOT / "tmp"
    return {
        # Package managers used by setup scripts and CI.
        "UV_CACHE_DIR": CACHE_ROOT / "uv",
        "PIP_CACHE_DIR": CACHE_ROOT / "pip",
        "UV_PYTHON_INSTALL_DIR": CACHE_ROOT / "uv" / "python",
        # Framework + plotting caches.
        "TORCH_HOME": CACHE_ROOT / "torch",
        "MPLCONFIGDIR": CACHE_ROOT / "matplotlib",
        "XDG_CACHE_HOME": CACHE_ROOT,
        # Hugging Face (dataset archive + hub metadata).
        "HF_HOME": CACHE_ROOT / "huggingface",
        "HF_HUB_CACHE": CACHE_ROOT / "huggingface" / "hub",
        "HF_DATASETS_CACHE": CACHE_ROOT / "huggingface" / "datasets",
        # Temporary files (pip/uv/wheel unpacking lands here).
        "TMP": tmp,
        "TEMP": tmp,
        "TMPDIR": tmp,
    }


def redirect_caches(*, create: bool = True) -> dict[str, str]:
    """Point every cache/temp variable at the project drive.

    Only sets variables that are not already set, so an explicit override from
    the calling shell still wins. Returns the variables that were applied.
    """
    applied: dict[str, str] = {}
    for name, path in _cache_env().items():
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if name in FORCE_REDIRECT or not os.environ.get(name):
            os.environ[name] = str(path)
            applied[name] = str(path)
    return applied


# Caches must be redirected at import time: importing torch or matplotlib first
# would let them create their default (C:-drive) directories.
redirect_caches()


def cache_redirection_report() -> dict[str, dict[str, object]]:
    """Where each cache variable actually points, flagging the system drive.

    A cache that silently resolves onto C: defeats the point of the guard, so
    this is reported rather than assumed correct.
    """
    system_root = volume_root("C:/") if os.name == "nt" else None
    report: dict[str, dict[str, object]] = {}
    for name, target in _cache_env().items():
        actual = os.environ.get(name, "")
        on_system = False
        if actual and system_root is not None:
            try:
                on_system = Path(actual).resolve().anchor == str(system_root)
            except OSError:  # pragma: no cover - unreadable path
                on_system = True
        report[name] = {
            "expected": str(target),
            "actual": actual,
            "on_system_drive": on_system,
        }
    return report


def free_bytes(path: os.PathLike[str] | str) -> int:
    """Free bytes on the volume containing ``path``.

    The path itself need not exist; its closest existing ancestor is measured.
    """
    probe = Path(path)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def volume_root(path: os.PathLike[str] | str) -> Path:
    """Volume root (e.g. ``D:\\``) containing ``path``."""
    return Path(Path(path).resolve().anchor or Path.cwd().anchor)


def format_bytes(num: int) -> str:
    """Human-readable byte count."""
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def required_bytes(expected_transfer: int) -> int:
    """Total free space required on the project drive for a transfer."""
    return int(expected_transfer * HEADROOM_MULTIPLIER) + RESERVE_BYTES


def guard_project_space(
    expected_transfer: int,
    label: str,
    *,
    path: os.PathLike[str] | str | None = None,
) -> int:
    """Refuse an operation that would overfill the project drive.

    Raises :class:`InsufficientSpaceError` with a message naming the measured
    requirement and the measured availability.
    """
    target = Path(path) if path is not None else PROJECT_ROOT
    need = required_bytes(expected_transfer)
    have = free_bytes(target)
    if have < need:
        raise InsufficientSpaceError(
            f"[disk guard] refusing '{label}': needs {format_bytes(need)} free on "
            f"{volume_root(target)} "
            f"({format_bytes(expected_transfer)} transfer x{HEADROOM_MULTIPLIER} "
            f"+ {format_bytes(RESERVE_BYTES)} reserve) but only {format_bytes(have)} "
            f"is available."
        )
    return have


def guard_c_drive(*, min_free: int = MIN_C_FREE_BYTES, strict: bool = False) -> int:
    """Check the system drive, which is nearly full on this machine.

    Two distinct situations are deliberately treated differently:

    * **Package installs** can write to ``C:`` (unpacking, link caches), so they
      pass ``strict=True`` and are refused when the floor is breached.
    * **Project-local operations** write only to the project drive, because every
      cache and temp variable is redirected there. For these the check is
      advisory: refusing them would stop the whole project over a system-drive
      condition they cannot worsen. The shortfall is still reported loudly.

    Returns the measured free bytes on ``C:``.
    """
    if os.name != "nt":
        return 0
    root = Path("C:/")
    if not root.exists():  # pragma: no cover - non-standard Windows layout
        return 0
    have = free_bytes(root)
    if have < min_free and strict:
        raise InsufficientSpaceError(
            f"[disk guard] refusing to proceed: system drive C: has only "
            f"{format_bytes(have)} free, below the {format_bytes(min_free)} "
            f"minimum. This operation can write to C: (package install), so "
            f"free space before continuing."
        )
    return have


def system_drive_warning(*, min_free: int = MIN_C_FREE_BYTES) -> str | None:
    """Advisory message when the system drive is low, else ``None``."""
    if os.name != "nt":
        return None
    have = free_bytes("C:/")
    if have < min_free:
        return (
            f"[disk guard] warning: system drive C: has only {format_bytes(have)} "
            f"free (below the {format_bytes(min_free)} advisory floor). Project "
            f"caches and temp files are redirected to the project drive, so this "
            f"run will not write to C: - but free space before installing "
            f"packages or the system may become unstable."
        )
    return None


def guard(
    expected_transfer: int,
    label: str,
    *,
    path: os.PathLike[str] | str | None = None,
    enforce_system_drive: bool = False,
) -> dict[str, object]:
    """Apply the disk guards before a write-heavy operation.

    The project-drive check is always enforced. The system-drive check is
    advisory unless ``enforce_system_drive`` is set (installs), and any shortfall
    is printed so it cannot pass unnoticed.
    """
    result: dict[str, object] = {
        "project_free": guard_project_space(expected_transfer, label, path=path),
        "c_free": guard_c_drive(strict=enforce_system_drive),
    }
    warning = system_drive_warning()
    result["c_warning"] = warning
    if warning:
        print(warning, file=sys.stderr, flush=True)
    return result


def environment_report() -> dict[str, object]:
    """Measured environment facts, for the committed environment report."""
    import platform
    import sys

    report: dict[str, object] = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "project_root": str(PROJECT_ROOT),
        "project_drive_free_bytes": free_bytes(PROJECT_ROOT),
        "project_drive_free_human": format_bytes(free_bytes(PROJECT_ROOT)),
    }
    if os.name == "nt":
        c_free = free_bytes("C:/")
        report["c_drive_free_bytes"] = c_free
        report["c_drive_free_human"] = format_bytes(c_free)
    report["redirected_caches"] = {
        name: os.environ.get(name, "") for name in _cache_env()
    }
    caches = cache_redirection_report()
    report["caches_on_system_drive"] = sorted(
        name for name, info in caches.items() if info["on_system_drive"]
    )
    return report


if __name__ == "__main__":  # pragma: no cover - manual diagnostic
    import json

    print(json.dumps(environment_report(), indent=2))
