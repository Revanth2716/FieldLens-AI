"""The single inference entry point.

The CLI and the Streamlit UI both predict through this class, so train/serve
skew is a testable property rather than an assumption: a test asserts the two
paths return identical results for the same input.

Confidence is reported as three independent signals (max probability, top-1/top-2
margin, normalised entropy) rather than one opaque score, together with an
explicit verdict. The wording deliberately avoids "certain" and "safe": a softmax
probability is not a probability of being correct.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Prediction",
    "Predictor",
    "ConfidencePolicy",
    "PredictionError",
]


class PredictionError(RuntimeError):
    """The image could not be read or scored."""


#: Bands are labels, not guarantees. Thresholds may be replaced by values fitted
#: on the validation split (see the calibration step) and stored in the
#: checkpoint metadata; these defaults apply until then.
HIGH = "high"
MODERATE = "moderate"
LOW = "low"

_BAND_NOTE = {
    HIGH: "the model is confident and the runner-up is clearly behind",
    MODERATE: "the model leans one way but the alternatives are close",
    LOW: "the model is not confident; treat this prediction as unreliable",
}


@dataclass(frozen=True)
class ConfidencePolicy:
    """Thresholds that turn probabilities into a plain-language verdict."""

    high_confidence: float = 0.80
    moderate_confidence: float = 0.55
    min_margin: float = 0.20
    max_entropy: float = 0.60

    def __post_init__(self) -> None:
        if not 0 < self.moderate_confidence <= self.high_confidence <= 1:
            raise PredictionError(
                "confidence thresholds must satisfy "
                "0 < moderate_confidence <= high_confidence <= 1, got "
                f"{self.moderate_confidence} and {self.high_confidence}"
            )
        if not 0 <= self.min_margin <= 1:
            raise PredictionError(f"min_margin must be in [0, 1], got {self.min_margin}")
        if not 0 <= self.max_entropy <= 1:
            raise PredictionError(
                f"max_entropy must be in [0, 1] (normalised), got {self.max_entropy}"
            )

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ConfidencePolicy:
        data = data or {}
        return cls(
            high_confidence=float(data.get("high_confidence", 0.80)),
            moderate_confidence=float(data.get("moderate_confidence", 0.55)),
            min_margin=float(data.get("min_margin", 0.20)),
            max_entropy=float(data.get("max_entropy", 0.60)),
        )

    def classify(self, confidence: float, margin: float, entropy: float) -> str:
        """Verdict from the three signals. Any doubt demotes the band."""
        if (
            confidence >= self.high_confidence
            and margin >= self.min_margin
            and entropy <= self.max_entropy
        ):
            return HIGH
        if confidence >= self.moderate_confidence and margin > 0:
            return MODERATE
        return LOW


@dataclass
class Prediction:
    """One prediction, with everything needed to explain it."""

    label: str
    confidence: float
    top_k: list[tuple[str, float]]
    margin: float
    entropy: float
    warning_level: str
    temperature: float
    num_classes: int
    image_path: str | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def band_note(self) -> str:
        return _BAND_NOTE.get(self.warning_level, "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 6),
            "top_k": [[name, round(prob, 6)] for name, prob in self.top_k],
            "margin": round(self.margin, 6),
            "entropy": round(self.entropy, 6),
            "warning_level": self.warning_level,
            "band_note": self.band_note,
            "temperature": self.temperature,
            "num_classes": self.num_classes,
            "image_path": self.image_path,
            "notes": self.notes,
        }


def _softmax(logits):
    import torch

    return torch.softmax(logits, dim=-1)


def _normalised_entropy(probs) -> float:
    """Shannon entropy divided by log(n), so the range is [0, 1]."""
    import torch

    n = probs.shape[-1]
    if n <= 1:
        return 0.0
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum()
    return float(entropy / math.log(n))


class Predictor:
    """Loads a checkpoint once and scores images through a single code path."""

    def __init__(
        self,
        model,
        metadata,
        *,
        device: str = "cpu",
        policy: ConfidencePolicy | None = None,
    ) -> None:
        from fieldlens.data.transforms import build_eval_transform, spec_from_checkpoint

        self.model = model
        self.metadata = metadata
        self.device = device
        self.class_names = tuple(metadata.class_names)
        self.temperature = float(metadata.temperature)
        self.policy = policy or ConfidencePolicy()
        self._transform = build_eval_transform(spec_from_checkpoint(metadata))

    # --- construction ----------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        *,
        dataset_classes: Sequence[str] | None = None,
        device: str | None = None,
        policy: ConfidencePolicy | None = None,
    ) -> Predictor:
        """Load a checkpoint. Raises :class:`CheckpointError` with guidance."""
        import torch

        from fieldlens.models.checkpoint import load_checkpoint

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model, metadata = load_checkpoint(
            Path(path),
            dataset_classes=tuple(dataset_classes) if dataset_classes else None,
            device=device,
        )
        return cls(model, metadata, device=device, policy=policy)

    @property
    def image_size(self) -> int:
        return self.metadata.image_size

    # --- inference -------------------------------------------------------
    def _load_image(self, image: str | Path | bytes | Any):
        from PIL import Image

        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise PredictionError(f"image not found: {path}")
            if path.is_dir():
                raise PredictionError(f"expected an image file, got a directory: {path}")
            try:
                with Image.open(path) as handle:
                    return handle.convert("RGB")
            # OSError covers UnidentifiedImageError *and* truncated or
            # permission-denied files, which would otherwise escape as a traceback.
            except OSError as exc:
                raise PredictionError(f"not a readable image file: {path}") from exc
        if isinstance(image, bytes):
            import io

            try:
                with Image.open(io.BytesIO(image)) as handle:
                    return handle.convert("RGB")
            except OSError as exc:
                raise PredictionError("uploaded bytes are not a readable image") from exc
        if hasattr(image, "convert"):
            return image.convert("RGB")
        raise PredictionError(f"unsupported image input type: {type(image)!r}")

    def load_rgb(self, image):
        """Load ``image`` (path/bytes/PIL) as RGB at the model's input size.

        Public because Grad-CAM must preprocess through exactly the same path the
        model serves, rather than re-implementing the pipeline.
        """
        return self._load_image(image)

    def predict(self, image) -> Prediction:
        """Score one image and return a fully explained prediction."""
        import torch

        pil = self._load_image(image)
        tensor = self._transform(pil).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            # Temperature scaling is applied at inference so a calibrated
            # checkpoint reports calibrated probabilities.
            probs = _softmax(logits / self.temperature)
        return self._to_prediction(
            probs[0],
            image_path=str(image) if isinstance(image, (str, Path)) else None,
        )

    def predict_batch(self, images: Sequence[Any]) -> list[Prediction]:
        """Score several images in one forward pass."""
        import torch

        if not images:
            return []
        pils = [self._load_image(image) for image in images]
        batch = torch.stack([self._transform(pil) for pil in pils]).to(self.device)
        with torch.inference_mode():
            probs = _softmax(self.model(batch) / self.temperature)
        return [self._to_prediction(row) for row in probs]

    def _to_prediction(self, probs, image_path: str | None = None) -> Prediction:
        import torch

        top_k = min(3, len(self.class_names))
        values, indices = torch.topk(probs, top_k)
        pairs = [
            (self.class_names[int(i)], float(v))
            for v, i in zip(values, indices, strict=True)
        ]
        confidence = float(values[0])
        margin = float(values[0] - values[1]) if len(values) > 1 else confidence
        entropy = _normalised_entropy(probs)

        notes: list[str] = []
        if confidence > 0.999:
            notes.append(
                "Extremely high confidence is not proof of correctness; the model "
                "has no reject option for inputs unlike its training data."
            )
        if self.temperature != 1.0:
            notes.append(f"Probabilities calibrated with T={self.temperature:.3f}.")

        return Prediction(
            label=pairs[0][0],
            confidence=confidence,
            top_k=pairs,
            margin=margin,
            entropy=entropy,
            warning_level=self.policy.classify(confidence, margin, entropy),
            temperature=self.temperature,
            num_classes=len(self.class_names),
            image_path=image_path,
            notes=notes,
        )

    # --- presentation ----------------------------------------------------
    def explain(self, image) -> dict[str, Any]:
        """Machine-readable prediction, used by the CLI ``--json`` and the UI."""
        return self.predict(image).to_dict()

    def render(self, result: dict[str, Any] | Prediction) -> None:
        """Human-readable CLI rendering of a prediction."""
        data = result.to_dict() if isinstance(result, Prediction) else result
        label = data["label"]
        crop, _, state = label.partition("___")
        pretty_state = state.replace("_", " ").strip() or "unknown"

        print(f"image      : {data['image_path'] or '(in-memory)'}")
        print(f"prediction : {crop}  ->  {pretty_state}")
        print(f"confidence : {data['confidence']:.3f}")
        print(f"margin     : {data['margin']:.3f}  (top-1 minus top-2)")
        print(f"entropy    : {data['entropy']:.3f}  (0 = certain, 1 = uniform)")
        band = data["warning_level"].upper()
        print(f"verdict    : {band} - {data['band_note']}")
        print("top-3:")
        for name, prob in data["top_k"]:
            print(f"  {prob * 100:6.2f}%  {name}")
        for note in data.get("notes", []):
            print(f"note       : {note}")

    def described_caveats(self) -> list[str]:
        """Limitations shown alongside every prediction in the UI."""
        return [
            "Trained on single detached leaves photographed under controlled "
            "lighting with uniform backgrounds; not a field-condition diagnosis.",
            "No reject option: an image of something other than a leaf still "
            "receives a confident label.",
            "Calibration was measured on this dataset only and does not transfer "
            "to a different camera, lighting regime, or background.",
        ]
