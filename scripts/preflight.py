#!/usr/bin/env python
"""Preflight: measure the environment and refuse to proceed if space is short.

Stdlib only, so it runs before any dependency is installed and before the
virtual environment exists. Run it before every write-heavy operation.

    python scripts/preflight.py
    python scripts/preflight.py --require-mb 2185   # assert room for a transfer
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

GB = 1024**3
MIN_C_FREE_BYTES = 10 * GB
RESERVE_BYTES = 20 * GB
HEADROOM_MULTIPLIER = 1.5


def free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def human(num: int) -> str:
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--require-mb",
        type=float,
        default=0.0,
        help="size of the transfer about to start, in MB",
    )
    parser.add_argument(
        "--label",
        default="operation",
        help="name of the operation, used in guard messages",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args(argv)

    project_free = free_bytes(PROJECT_ROOT)
    c_free = free_bytes(Path("C:/")) if os.name == "nt" else None

    problems: list[str] = []
    warnings: list[str] = []

    # Advisory, not blocking: this script writes only to the project drive, since
    # caches and temp are redirected there. The strict system-drive check applies
    # to package installs, which can write to C:.
    if c_free is not None and c_free < MIN_C_FREE_BYTES:
        warnings.append(
            f"system drive C: has {human(c_free)} free, below the advisory "
            f"{human(MIN_C_FREE_BYTES)} floor; caches and temp are redirected, "
            f"so this run will not write to C:"
        )

    required = 0
    if args.require_mb:
        required = int(args.require_mb * 1024 * 1024)
        need = int(required * HEADROOM_MULTIPLIER) + RESERVE_BYTES
        if project_free < need:
            problems.append(
                f"'{args.label}' needs {human(need)} free on the project drive "
                f"({human(required)} x{HEADROOM_MULTIPLIER} + {human(RESERVE_BYTES)} "
                f"reserve) but only {human(project_free)} is available"
            )

    report = {
        "project_root": str(PROJECT_ROOT),
        "project_drive": str(Path(PROJECT_ROOT).anchor),
        "project_drive_free": human(project_free),
        "project_drive_free_bytes": project_free,
        "system_drive_free": human(c_free) if c_free is not None else "n/a",
        "system_drive_free_bytes": c_free,
        "required_for_transfer": human(required) if required else "n/a",
        "caches_redirected_to": str(PROJECT_ROOT / ".cache"),
        "problems": problems,
        "warnings": warnings,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("FieldLens AI - preflight")
        print(f"  project root      : {report['project_root']}")
        print(f"  project drive     : {report['project_drive']} "
              f"({report['project_drive_free']} free)")
        print(f"  system drive C:   : {report['system_drive_free']} free")
        print(f"  caches redirected : {report['caches_redirected_to']}")
        if required:
            print(f"  transfer required : {report['required_for_transfer']}")
        if warnings:
            print()
            print("  WARNINGS (non-blocking):")
            for warning in warnings:
                print(f"    - {warning}")
        if problems:
            print("\n  REFUSING TO PROCEED:")
            for problem in problems:
                print(f"    - {problem}")
        else:
            print("\n  status: OK")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
