"""Grad-CAM saliency.

Produces a coarse localisation map showing which regions most influenced a
predicted class, using gradients flowing into the last convolutional block.

**How to read this.** Grad-CAM is a *diagnostic aid*, not evidence of causation
and not lesion segmentation. It is deliberately used here as a way to inspect
failure modes — in particular whether the model is looking at the leaf or at the
background — and the report keeps a mix of correct and incorrect examples rather
than only flattering ones.

The map is deterministic for a given image, class and set of weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "GradCAMResult",
    "GradCAMExplainer",
    "default_target_layer",
    "overlay_heatmap",
]

#: Colour map applied to the normalised heatmap (RGB, 0-1), low to high.
_HEATMAP_STOPS = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float32,
)


def default_target_layer(model):
    """Last convolutional block of the backbone.

    Falls back through a couple of common layouts so a backbone swap does not
    require editing this module.
    """
    features = getattr(model, "features", None)
    if features is not None and len(features) > 0:
        return features[-1]
    raise ValueError(
        "cannot locate a convolutional feature block for Grad-CAM; "
        "expected the model to expose `features`"
    )


def _colourise(normalised: np.ndarray) -> np.ndarray:
    """Map a [0, 1] array to RGB using a small blue-to-red ramp."""
    scaled = np.clip(normalised, 0.0, 1.0) * (len(_HEATMAP_STOPS) - 1)
    low = np.floor(scaled).astype(int)
    high = np.minimum(low + 1, len(_HEATMAP_STOPS) - 1)
    weight = (scaled - low)[..., None]
    return (_HEATMAP_STOPS[low] * (1 - weight) + _HEATMAP_STOPS[high] * weight).astype(
        np.float32
    )


@dataclass
class GradCAMResult:
    """Heatmap plus the prediction it explains."""

    heatmap: np.ndarray  # float32 in [0, 1], model-input resolution
    label: str
    confidence: float
    target_class_index: int

    def overlay_on(self, pil_image, alpha: float = 0.45):
        """Original image with the heatmap composited over it."""
        return overlay_heatmap(pil_image, self.heatmap, alpha=alpha)

    def peak_region(self) -> tuple[int, int]:
        """(row, column) of the strongest activation, in heatmap coordinates."""
        index = int(np.argmax(self.heatmap))
        return divmod(index, self.heatmap.shape[1])


def _upsample(heatmap: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a coarse CAM up to the display size, staying in [0, 1]."""
    from PIL import Image

    as_bytes = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    resized = Image.fromarray(as_bytes, mode="L").resize(
        (width, height), Image.BICUBIC
    )
    return np.asarray(resized, dtype=np.float32) / 255.0


def overlay_heatmap(pil_image, heatmap: np.ndarray, alpha: float = 0.45):
    """Composite a heatmap onto an image, returning a new RGB image.

    The CAM is coarse (a 7x7 map for a 224 px MobileNet input), so it is
    upsampled *to the image* — never the image down to the CAM, which would
    silently produce a thumbnail-sized overlay.
    """
    from PIL import Image

    base = pil_image.convert("RGB")
    width, height = base.size
    if (int(heatmap.shape[1]), int(heatmap.shape[0])) != (width, height):
        heatmap = _upsample(heatmap, width, height)
    base_array = np.asarray(base, dtype=np.float32) / 255.0
    colour = _colourise(heatmap)
    blended = (1 - alpha) * base_array + alpha * colour
    return Image.fromarray((np.clip(blended, 0, 1) * 255).astype(np.uint8))


class GradCAMExplainer:
    """Computes Grad-CAM maps through the same preprocessing the model serves."""

    def __init__(self, predictor, target_layer=None) -> None:
        self.predictor = predictor
        self.model = predictor.model
        self.device = predictor.device
        self.target_layer = target_layer or default_target_layer(self.model)
        self._activations: Any = None
        self._gradients: Any = None
        self._handles = [
            self.target_layer.register_forward_hook(self._capture_activations),
            self.target_layer.register_full_backward_hook(self._capture_gradients),
        ]

    # --- hooks ----------------------------------------------------------
    def _capture_activations(self, _module, _inputs, output) -> None:
        self._activations = output

    def _capture_gradients(self, _module, _grad_input, grad_output) -> None:
        self._gradients = grad_output[0]

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
        # Drop captures too: otherwise a closed explainer would quietly return
        # the previous image's map, which is worse than failing.
        self._activations = None
        self._gradients = None

    def __enter__(self) -> GradCAMExplainer:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # --- computation ----------------------------------------------------
    def explain(self, image, *, target_class: int | None = None) -> GradCAMResult:
        """Compute a heatmap for ``image`` (optionally for a specific class)."""
        import torch

        if not self._handles:
            raise RuntimeError(
                "this GradCAMExplainer is closed; create a new one for each explanation"
            )

        # Clear first, so a hook that fails to fire is detected rather than
        # silently reusing the previous forward pass.
        self._activations = None
        self._gradients = None

        pil = self.predictor.load_rgb(image)
        tensor = self.predictor._transform(pil).unsqueeze(0).to(self.device)

        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())

        score = logits[0, target_class]
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Grad-CAM hooks did not fire; is the layer correct?")

        activations = self._activations.detach()[0]
        gradients = self._gradients.detach()[0]
        # Global-average the gradients into per-channel importance weights.
        weights = gradients.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=0))
        cam = cam.detach().float().cpu().numpy()

        # A constant map is honest — it says nothing stood out — so a zero
        # maximum normalises to zeros rather than dividing by zero.
        maximum = float(cam.max())
        cam = cam / maximum if maximum > 0 else np.zeros_like(cam)

        probabilities = torch.softmax(
            logits.detach() / self.predictor.temperature, dim=-1
        )[0]
        confidence = float(probabilities[target_class])

        self.model.zero_grad(set_to_none=True)
        return GradCAMResult(
            heatmap=cam.astype(np.float32),
            label=self.predictor.class_names[target_class],
            confidence=confidence,
            target_class_index=target_class,
        )

    def described_caveats(self) -> list[str]:
        return [
            "Grad-CAM is a coarse localisation aid, not lesion detection or "
            "evidence of causation.",
            "A plausible-looking map does not prove the model reasons correctly; "
            "it shows where gradient-weighted activation was strongest.",
            "Maps are deterministic for given weights and image but are only as "
            "meaningful as the model itself.",
        ]
