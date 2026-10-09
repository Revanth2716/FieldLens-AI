"""Project layout: every path used by FieldLens AI, resolved in one place.

Directories are created on demand via :func:`ensure_dirs`. Nothing here writes
to a user-level cache; see :mod:`fieldlens.env` for that.
"""

from __future__ import annotations

from pathlib import Path

from fieldlens.env import CACHE_ROOT, PROJECT_ROOT

# --- Source tree ---
SRC_DIR = PROJECT_ROOT / "src" / "fieldlens"
CONFIG_DIR = PROJECT_ROOT / "configs"
SCRIPT_DIR = PROJECT_ROOT / "scripts"
TEST_DIR = PROJECT_ROOT / "tests"

# --- Data (gitignored: CC-BY-SA-3.0 content must not be redistributed) ---
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
METADATA_DIR = DATA_DIR / "metadata"
IMAGE_DIR = DATA_DIR / "images"
IMAGE_CACHE_DIR = DATA_DIR / "image_cache"
#: Small acquired sample used for demos and smoke checks (see `fieldlens sample`).
SAMPLE_DIR = DATA_DIR / "sample"

# --- Artifacts ---
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
CHECKPOINT_DIR = ARTIFACT_DIR / "checkpoints"
REPORT_DIR = ARTIFACT_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
GALLERY_DIR = REPORT_DIR / "gallery"

#: Stable name for the promoted checkpoint referenced by docs and the UI.
PROMOTED_CHECKPOINT = CHECKPOINT_DIR / "fieldlens_mobilenetv3s_12class.pt"

_CRITICAL = (
    DATA_DIR,
    MANIFEST_DIR,
    METADATA_DIR,
    IMAGE_DIR,
    IMAGE_CACHE_DIR,
    ARTIFACT_DIR,
    CHECKPOINT_DIR,
    REPORT_DIR,
    FIGURE_DIR,
    GALLERY_DIR,
    CONFIG_DIR,
    CACHE_ROOT,
)


def ensure_dirs() -> None:
    """Create every runtime directory. Safe to call repeatedly."""
    for path in _CRITICAL:
        path.mkdir(parents=True, exist_ok=True)


def config_path(name: str) -> Path:
    """Path to a YAML config, e.g. ``config_path("dataset")``."""
    return CONFIG_DIR / f"{name}.yaml"


def report_path(name: str) -> Path:
    """Path to a committed JSON report artifact."""
    return REPORT_DIR / f"{name}.json"


def figure_path(name: str) -> Path:
    """Path to a generated figure."""
    return FIGURE_DIR / f"{name}.png"
