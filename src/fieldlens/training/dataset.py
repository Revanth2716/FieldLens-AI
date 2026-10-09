"""Manifest-backed image dataset.

Reads only paths listed in the committed manifest, so a training run can never
accidentally reach images outside its declared split. Images live under an
``image_root`` mirroring the archive layout (``<root>/raw/color/<Class>/<file>``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

__all__ = ["LeafImageDataset", "ImageMissingError", "class_weight_vector"]


class ImageMissingError(FileNotFoundError):
    """A manifest entry has no corresponding image on disk."""


class LeafImageDataset:
    """A torch Dataset over manifest entries.

    Implements the minimal mapping protocol by hand so the module imports without
    torch at module scope (useful for the metadata-only commands).
    """

    def __init__(
        self,
        entries: Sequence,
        class_index: dict[str, int],
        transform,
        image_root: Path,
        *,
        strict: bool = True,
    ) -> None:
        self.entries = list(entries)
        self.class_index = class_index
        self.transform = transform
        self.image_root = Path(image_root)
        self.strict = strict

    def __len__(self) -> int:
        return len(self.entries)

    def resolve(self, index: int) -> Path:
        """Absolute path of the image for a manifest entry."""
        return self.image_root / self.entries[index].path

    def label_of(self, index: int) -> int:
        entry = self.entries[index]
        try:
            return self.class_index[entry.class_name]
        except KeyError as exc:  # pragma: no cover - manifest/config mismatch
            raise KeyError(
                f"class {entry.class_name!r} is not in the configured class list"
            ) from exc

    def __getitem__(self, index: int):
        from PIL import Image, UnidentifiedImageError

        path = self.resolve(index)
        if not path.exists():
            if self.strict:
                raise ImageMissingError(
                    f"image listed in the manifest is missing on disk: {path}"
                )
            # Skip forward deterministically rather than returning garbage.
            return self[(index + 1) % len(self)]
        try:
            with Image.open(path) as handle:
                image = handle.convert("RGB")
        except UnidentifiedImageError as exc:
            raise ImageMissingError(f"unreadable image: {path}") from exc
        return self.transform(image), self.label_of(index)

    # --- diagnostics -----------------------------------------------------
    def labels(self) -> list[int]:
        """Labels for every entry, in order."""
        return [self.label_of(i) for i in range(len(self))]

    def missing_files(self) -> list[Path]:
        """Entries whose image is absent, for a pre-flight check."""
        return [self.resolve(i) for i in range(len(self)) if not self.resolve(i).exists()]

    def counts_by_class(self) -> dict[str, int]:
        counts: dict[str, int] = {name: 0 for name in self.class_index}
        for entry in self.entries:
            counts[entry.class_name] = counts.get(entry.class_name, 0) + 1
        return counts


def class_weight_vector(
    entries: Sequence, class_index: dict[str, int]
) -> list[float]:
    """Inverse-frequency weights, for the optional balanced-sampling path.

    Returned as a plain list so it can be handed to ``WeightedRandomSampler``
    without importing torch here.
    """
    totals: dict[str, int] = {name: 0 for name in class_index}
    for entry in entries:
        totals[entry.class_name] = totals.get(entry.class_name, 0) + 1
    n_classes = len(class_index)
    present = [count for count in totals.values() if count > 0]
    if not present:
        return []
    weights = []
    for name in class_index:
        count = totals.get(name, 0)
        weights.append(0.0 if count == 0 else (sum(present) / (n_classes * count)))
    return weights
