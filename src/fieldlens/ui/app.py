"""FieldLens AI — Streamlit interface.

Run with::

    fieldlens serve
    # or
    streamlit run src/fieldlens/ui/app.py

Design rules enforced here:

* The UI **computes nothing**. Every metric is read from a report artifact that
  the evaluation pipeline wrote, so a number on screen can always be traced to a
  committed command.
* Prediction goes through the same :class:`Predictor` the CLI uses.
* Missing model or missing reports produce a clear instruction, never a fake
  prediction or a stack trace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Redirect caches before anything else imports torch.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import streamlit as st  # noqa: E402

import fieldlens.env as env  # noqa: E402
from fieldlens.config import DatasetConfig  # noqa: E402
from fieldlens.paths import (  # noqa: E402
    FIGURE_DIR,
    IMAGE_DIR,
    PROMOTED_CHECKPOINT,
    REPORT_DIR,
    SAMPLE_DIR,
)

VERDICT_ICON = {"high": "✅", "moderate": "⚠️", "low": "🛑"}
VERDICT_TEXT = {
    "high": "High confidence",
    "moderate": "Uncertain — please review",
    "low": "Low confidence — unreliable",
}


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _one_per_class(root: Path, limit: int) -> list[Path]:
    """First image of each class directory, so a picker spans the classes."""
    if not root.is_dir():
        return []
    picked: list[Path] = []
    for class_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        candidate = next((f for f in sorted(class_dir.iterdir()) if f.is_file()), None)
        if candidate is not None:
            picked.append(candidate)
        if len(picked) >= limit:
            break
    return picked


def _bundled_samples(limit: int = 12) -> list[Path]:
    """Real dataset images offered as ready-made examples.

    Anchored to the project root rather than the working directory: Streamlit does
    not guarantee a cwd, and a relative literal silently produced an empty picker.
    Falls back to the extracted images when no sample has been acquired.
    """
    for root in (SAMPLE_DIR / "raw" / "color", IMAGE_DIR / "raw" / "color"):
        picked = _one_per_class(root, limit)
        if picked:
            return picked
    return []


@st.cache_resource(show_spinner="Loading model…")
def load_predictor(checkpoint_path: str):
    """Load the predictor once per session (first load is the slow one)."""
    from fieldlens.inference.predictor import Predictor

    return Predictor.from_checkpoint(Path(checkpoint_path))


def _predictor_or_message():
    """Return a predictor, or render guidance and return None."""
    if not PROMOTED_CHECKPOINT.exists():
        st.error("No trained model is available yet.")
        st.markdown(
            "Train one first:\n\n"
            "```bash\n"
            "fieldlens prepare      # dataset metadata (~8 MB)\n"
            "fieldlens audit        # build the split + leakage gate\n"
            "fieldlens extract --per-class-per-split 60\n"
            "fieldlens train\n"
            "```\n\n"
            f"Expected checkpoint: `{PROMOTED_CHECKPOINT}`"
        )
        return None
    try:
        return load_predictor(str(PROMOTED_CHECKPOINT))
    except Exception as exc:  # noqa: BLE001 - show guidance, never a traceback
        st.error(f"The checkpoint could not be loaded: {exc}")
        st.info("Retrain with `fieldlens train`, or check the checkpoint path.")
        return None


# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
def page_predict() -> None:
    st.header("Predict")
    st.caption(
        "Classifies a single leaf image and reports how much the prediction "
        "should be trusted."
    )

    predictor = _predictor_or_message()
    if predictor is None:
        return

    uploaded = st.file_uploader(
        "Upload a leaf image", type=["jpg", "jpeg", "png", "bmp", "webp"]
    )
    demo_files = _bundled_samples()
    choice = st.selectbox(
        "…or use a bundled sample",
        ["(none)"] + [str(p) for p in demo_files],
        format_func=lambda value: (
            value
            if value == "(none)"
            else f"{Path(value).parent.name}  ({Path(value).name[:24]})"
        ),
    )
    if demo_files:
        st.caption(
            f"{len(demo_files)} bundled examples, one per class — real "
            "PlantVillage images, not synthetic."
        )

    image_bytes = None
    label = None
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        label = uploaded.name
    elif choice != "(none)":
        image_bytes = Path(choice).read_bytes()
        label = Path(choice).name

    if image_bytes is None:
        st.info("Upload an image or pick a bundled sample to see a prediction.")
        _render_caveats(predictor)
        return

    st.image(image_bytes, caption=label, width=320)

    try:
        with st.spinner("Analysing image…"):
            result = predictor.predict(image_bytes)
    except Exception as exc:  # noqa: BLE001 - user-facing message
        st.error(f"That image could not be processed: {exc}")
        return

    icon = VERDICT_ICON.get(result.warning_level, "")
    st.subheader(f"{icon} {VERDICT_TEXT.get(result.warning_level, result.warning_level)}")

    crop, _, state = result.label.partition("___")
    st.markdown(f"### {crop} — {state.replace('_', ' ').strip() or 'unknown'}")

    left, middle, right = st.columns(3)
    left.metric("Confidence", f"{result.confidence:.1%}")
    middle.metric("Margin (top-1 − top-2)", f"{result.margin:.3f}")
    right.metric("Entropy (0 = certain)", f"{result.entropy:.3f}")

    st.markdown(f"*{result.band_note}*")

    st.markdown("**Top 3 predictions**")
    for name, probability in result.top_k:
        st.progress(min(1.0, probability), text=f"{probability:.1%} — {name}")

    st.caption(
        "A confidence score is not a probability of being correct. The model has "
        "no reject option, so an image that is not a leaf still receives a label."
    )
    for note in result.notes:
        st.info(note)

    _render_gradcam(predictor, image_bytes, result)
    _render_caveats(predictor)


def _render_gradcam(predictor, image_bytes: bytes, result) -> None:
    """Optional Grad-CAM side-by-side, for inspecting *where* the model looked.

    Opt-in because it costs a backward pass, and framed as a diagnostic rather
    than as evidence: a plausible map does not prove the model reasons correctly.
    """
    with st.expander("Grad-CAM — where the model looked"):
        st.caption(
            "A coarse localisation aid for inspecting failure modes, not lesion "
            "detection. Ideal for checking whether the model is attending to the "
            "leaf or to the background."
        )
        if not st.checkbox(
            "Compute a Grad-CAM map for this image", value=False, key="show_gradcam"
        ):
            return
        try:
            from fieldlens.explain.gradcam import GradCAMExplainer

            with (
                st.spinner("Computing Grad-CAM…"),
                GradCAMExplainer(predictor) as explainer,
            ):
                cam = explainer.explain(image_bytes)
                caveats = explainer.described_caveats()
        except (RuntimeError, ValueError) as exc:  # noqa: BLE001 - user-facing
            st.warning(f"Grad-CAM is unavailable for this model: {exc}")
            return

        overlay = cam.overlay_on(predictor.load_rgb(image_bytes))
        left, right = st.columns(2)
        left.image(image_bytes, caption="input", use_container_width=True)
        right.image(overlay, caption=f"explained: {cam.label}", use_container_width=True)

        rows, columns = cam.heatmap.shape
        row, column = cam.peak_region()
        st.caption(
            f"Peak activation at x={((column + 0.5) / columns):.2f}, "
            f"y={((row + 0.5) / rows):.2f} (fractions of the frame, 0–1). "
            f"Explained class: {cam.label} at {cam.confidence:.1%}."
        )
        if cam.label != result.label:
            st.info(
                f"The map explains `{cam.label}`, not the predicted `{result.label}`."
            )
        for caveat in caveats:
            st.caption(f"• {caveat}")


def _render_caveats(predictor) -> None:
    with st.expander("Limitations of this model"):
        for caveat in predictor.described_caveats():
            st.markdown(f"- {caveat}")


# --------------------------------------------------------------------------- #
# Reliability
# --------------------------------------------------------------------------- #
def page_reliability() -> None:
    st.header("Reliability")
    report = _load_json(REPORT_DIR / "evaluation.json")
    if report is None:
        st.warning("No evaluation has been run yet.")
        st.markdown(
            "Generate the numbers this page displays with:\n\n"
            "```bash\nfieldlens evaluate\n```"
        )
        return

    dataset = report["dataset"]
    st.caption(
        f"{report['model']['backbone']} · evaluated on "
        f"{dataset['test_images_evaluated']} test images "
        f"({dataset['images_evaluated_fraction_of_test']:.0%} of the "
        f"{dataset['test_entries_in_manifest']} in the manifest)"
    )
    if dataset["images_evaluated_fraction_of_test"] < 0.999:
        st.info(
            "The acquisition was capped, so only part of the official test split "
            "was evaluated. Rate the numbers below accordingly."
        )

    uncalibrated = report["results"]["uncalibrated"]
    calibrated = report["results"]["calibrated"]

    st.subheader("Test metrics")
    columns = st.columns(4)
    columns[0].metric("Accuracy", f"{calibrated['accuracy']:.3f}",
                      f"{calibrated['accuracy'] - uncalibrated['accuracy']:+.3f}")
    columns[1].metric("Macro F1", f"{calibrated['macro_f1']:.3f}",
                      f"{calibrated['macro_f1'] - uncalibrated['macro_f1']:+.3f}")
    columns[2].metric("Top-3 accuracy", f"{calibrated['top3_accuracy']:.3f}")
    columns[3].metric("Macro recall", f"{calibrated['macro_recall']:.3f}")

    st.subheader("Calibration")
    st.caption(
        "Temperature scaling is fitted on the validation split and applied "
        "unchanged to test."
    )
    fit = report["calibration_fit"]
    columns = st.columns(3)
    columns[0].metric("Temperature", f"{fit['temperature']:.3f}")
    columns[1].metric("ECE (uncalibrated)", f"{uncalibrated['ece']:.4f}")
    columns[2].metric("ECE (calibrated)", f"{calibrated['ece']:.4f}")
    if not fit["improved"]:
        st.warning(
            "Temperature scaling did not reduce validation NLL here. That is "
            "reported as-is rather than tuned away."
        )

    figure = FIGURE_DIR / "reliability_diagram.png"
    if figure.exists():
        st.image(str(figure), width=480)

    st.subheader("Selective prediction")
    st.caption(
        "Thresholds are chosen on validation; the accuracy shown is measured on "
        "test at the resulting coverage."
    )
    rows = []
    for coverage, entry in sorted(
        report["results"]["selective_prediction"]["thresholds"].items(),
        key=lambda item: float(item[0]),
    ):
        rows.append(
            {
                "target coverage": f"{float(coverage):.0%}",
                "threshold": round(entry["threshold"], 4),
                "test coverage": f"{entry['test_coverage']:.1%}",
                "accuracy on kept images": round(entry["test_accuracy_at_threshold"], 4),
                "images kept": entry["test_images_kept"],
            }
        )
    st.dataframe(rows, use_container_width=True)

    st.subheader("Confusion matrix")
    matrix_figure = FIGURE_DIR / "confusion_matrix.png"
    if matrix_figure.exists():
        st.image(str(matrix_figure), width=620)
    else:
        st.caption("Confusion matrix figure not generated (matplotlib unavailable).")

    st.subheader("Per-crop performance")
    st.dataframe(
        [
            {"crop": crop, **{k: round(v, 4) if isinstance(v, float) else v
                              for k, v in values.items()}}
            for crop, values in calibrated["per_crop"].items()
        ],
        use_container_width=True,
    )


# --------------------------------------------------------------------------- #
# Error analysis
# --------------------------------------------------------------------------- #
def page_errors() -> None:
    st.header("Error analysis")
    report = _load_json(REPORT_DIR / "evaluation.json")
    if report is None:
        st.warning("No evaluation has been run yet. Run `fieldlens evaluate` first.")
        return

    per_class = report["results"]["calibrated"]["per_class"]
    name_by_index = {int(row["index"]): row.pop("class_name") for row in per_class}

    st.subheader("Weakest classes")
    st.caption(
        "Macro F1 is the headline metric because the rarest class "
        f"({report['dataset']['classes'][0].split('___')[0]} onwards) is small "
        "enough for plain accuracy to hide it."
    )
    st.dataframe(
        [
            {
                "class": name_by_index[int(row["index"])],
                "precision": round(row["precision"], 4),
                "recall": round(row["recall"], 4),
                "f1": round(row["f1"], 4),
                "support": int(row["support"]),
            }
            for row in sorted(per_class, key=lambda r: r["f1"])
        ],
        use_container_width=True,
    )

    matrix = report["results"]["calibrated"]["confusion_matrix"]
    classes = report["dataset"]["classes"]
    confusions = []
    for true_index, row in enumerate(matrix):
        for predicted_index, count in enumerate(row):
            if true_index != predicted_index and count:
                confusions.append(
                    {
                        "true": classes[true_index],
                        "predicted": classes[predicted_index],
                        "count": count,
                    }
                )
    st.subheader("Most common confusions")
    if confusions:
        st.dataframe(
            sorted(confusions, key=lambda r: -r["count"])[:15],
            use_container_width=True,
        )
    else:
        st.caption("No off-diagonal confusion in the evaluated test images.")


# --------------------------------------------------------------------------- #
# Model card
# --------------------------------------------------------------------------- #
def page_model_card() -> None:
    st.header("Model card, dataset and limitations")
    dataset = DatasetConfig.load()
    report = _load_json(REPORT_DIR / "evaluation.json")
    leakage = _load_json(REPORT_DIR / "leakage_audit.json")

    st.subheader("Intended use")
    st.markdown(
        "- **Intended:** a portfolio demonstration of honest computer-vision "
        "evaluation — leakage-safe splits, calibration, and selective prediction.\n"
        "- **Not intended:** diagnosis, advisory, or any decision about a real crop.\n"
        "- **Not a field-condition system:** the data is single detached leaves "
        "photographed under controlled lighting with uniform backgrounds."
    )

    st.subheader("Dataset")
    st.markdown(
        f"- Source: `{dataset.repo_id}` (config `{dataset.config}`)\n"
        f"- License: **{dataset.license}** — share-alike, attribution required\n"
        f"- Classes: {dataset.num_classes}\n"
        f"- Citation: {dataset.citation}"
    )
    st.caption(
        "Images are not redistributed with this project; only split manifests and "
        "derived metrics are committed."
    )

    st.subheader("Leakage audit")
    if leakage is None:
        st.caption("No audit recorded yet. Run `fieldlens audit`.")
    else:
        measured = leakage.get("cross_split_leaf_overlap_counts", {})
        st.markdown(
            f"- Split policy: `{leakage.get('split_policy')}`\n"
            f"- Result: **{'PASS' if leakage.get('passed') else 'FAIL'}**\n"
            f"- Leaky train/val images dropped: {leakage.get('dropped_leaky_entries')}\n"
            f"- Measured cross-split leaf overlaps: {measured or 'none'}\n"
            f"- Smallest class: `{leakage.get('smallest_class')}` "
            f"({leakage.get('smallest_class_count')} images)"
        )
        st.caption(
            "The upstream dataset card states its splits preserve leaf grouping. "
            "This independent audit measured 5 leaf identifiers present in both "
            "the official train and test lists, which the leak_free policy removes."
        )

    st.subheader("Measured metrics")
    if report is None:
        st.caption("No evaluation has been run yet.")
    else:
        calibrated = report["results"]["calibrated"]
        st.markdown(
            f"- Accuracy: {calibrated['accuracy']:.4f}\n"
            f"- Macro F1: {calibrated['macro_f1']:.4f}\n"
            f"- Top-3 accuracy: {calibrated['top3_accuracy']:.4f}\n"
            f"- ECE (calibrated): {calibrated['ece']:.4f}\n"
            f"- Test images evaluated: {report['dataset']['test_images_evaluated']}"
        )

    st.subheader("Known limitations")
    if report:
        for caveat in report["caveats"]:
            st.markdown(f"- {caveat}")
    st.markdown(
        "- No reject option: non-leaf images still receive a confident label.\n"
        "- Softmax confidence is not a probability of correctness.\n"
        "- Calibration does not transfer across cameras, lighting, or backgrounds.\n"
        "- Only temperature scaling is used; no ensembles or MC-dropout are in scope, "
        "so uncertainty reflects a single deterministic forward pass."
    )


PAGES = {
    "Predict": page_predict,
    "Reliability": page_reliability,
    "Error analysis": page_errors,
    "Model card": page_model_card,
}


def main() -> None:
    st.set_page_config(page_title="FieldLens AI", page_icon="🌿", layout="wide")
    st.sidebar.title("FieldLens AI")
    st.sidebar.caption("Plant leaf classification with confidence checks")
    choice = st.sidebar.radio("Page", list(PAGES), label_visibility="collapsed")
    st.sidebar.divider()
    st.sidebar.caption(
        f"Caches on: {Path(env.CACHE_ROOT).anchor}\n\n"
        "Not a field-condition diagnostic tool."
    )
    PAGES[choice]()


main()
