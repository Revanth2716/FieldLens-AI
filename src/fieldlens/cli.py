"""Command line interface for FieldLens AI.

Heavy dependencies (torch, streamlit, matplotlib) are imported lazily inside the
command handlers, so metadata-only commands such as ``audit`` run without the
model stack installed.

    fieldlens env                     environment and disk report
    fieldlens prepare                 fetch dataset metadata (a few MB)
    fieldlens inspect                 probe the remote archive (range-read check)
    fieldlens audit                   build the manifest and run the leakage gate
    fieldlens sample                  acquire a small real-image sample
    fieldlens train                   train the classifier
    fieldlens evaluate                metrics, calibration, reliability sweeps
    fieldlens predict <image>         classify one image with a confidence verdict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fieldlens.env import environment_report
from fieldlens.paths import REPORT_DIR, ensure_dirs

__all__ = ["main", "build_parser"]


def _dataset_config():
    from fieldlens.config import DatasetConfig

    return DatasetConfig.load()


def _eval_config():
    from fieldlens.config import EvalConfig

    return EvalConfig.load()


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_env(args: argparse.Namespace) -> int:
    """Print measured environment facts and disk headroom."""
    report = environment_report()
    from fieldlens.env import format_bytes, required_bytes
    from fieldlens.paths import PROJECT_ROOT

    report["required_headroom_for_full_archive"] = format_bytes(
        required_bytes(2185 * 1024 * 1024)
    )
    report["artifacts_dir"] = str(PROJECT_ROOT / "artifacts")
    if args.json:
        _print_json(report)
        return 0

    print("FieldLens AI - environment")
    print(f"  python           : {report['python_version']}")
    print(f"  executable       : {report['python_executable']}")
    print(f"  project root     : {report['project_root']}")
    print(f"  project drive    : {report['project_drive_free_human']} free")
    if "c_drive_free_human" in report:
        print(f"  system drive C:  : {report['c_drive_free_human']} free")
    print("  redirected caches:")
    for name, value in report["redirected_caches"].items():
        print(f"    {name:<20} {value}")
    try:
        import torch

        cuda = torch.cuda.is_available()
        device = torch.cuda.get_device_name(0) if cuda else "CPU"
        if cuda:
            props = torch.cuda.get_device_properties(0)
            memory = f"{props.total_memory / 1024**3:.2f} GiB"
        else:
            memory = "n/a"
        print(f"  torch            : {torch.__version__}")
        print(f"  cuda available   : {cuda}")
        print(f"  device           : {device} ({memory})")
        report["torch"] = {
            "version": torch.__version__,
            "cuda_available": cuda,
            "device": device,
            "device_memory": memory,
        }
    except Exception as exc:  # noqa: BLE001 - report, never crash the env command
        print(f"  torch            : not importable ({exc})")
        report["torch"] = {"error": str(exc)}
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    """Fetch the small split/leaf-map metadata files."""
    from fieldlens.data.acquire import fetch_metadata

    dataset = _dataset_config()
    expected = sum(dataset.metadata_files.values())
    result = fetch_metadata(dataset, expected_mb=expected, overwrite=args.force)
    print(f"metadata files in {dataset.metadata_dir}")
    for name, path in result.files.items():
        print(f"  {name:<32} {path.stat().st_size / 1e6:.2f} MB")
    print(f"downloaded this run: {result.bytes_downloaded / 1e6:.2f} MB")
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Probe the remote archive without downloading it."""
    from fieldlens.data.acquire import RangeReadNotSupported, inspect_remote_archive

    dataset = _dataset_config()
    try:
        info = inspect_remote_archive(dataset)
    except RangeReadNotSupported as exc:
        print(f"range-read unavailable: {exc}")
        print("fallback: `fieldlens prepare-full` downloads the archive once.")
        return 2
    _print_json(info)
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    """Build the manifest and run the leakage gate."""
    from fieldlens.data.leakage import audit_manifest
    from fieldlens.data.split import (
        ManifestError,
        build_manifest,
        default_manifest_path,
    )

    dataset = _dataset_config()
    try:
        manifest = build_manifest(dataset)
    except ManifestError as exc:
        print(f"cannot build manifest: {exc}", file=sys.stderr)
        print("hint: run `fieldlens prepare` first.", file=sys.stderr)
        return 2

    report = audit_manifest(manifest)

    ensure_dirs()
    manifest_path = default_manifest_path(dataset)
    manifest.save(manifest_path)
    leakage_path = REPORT_DIR / "leakage_audit.json"
    leakage_path.write_text(json.dumps(report.to_json(), indent=2), encoding="utf-8")

    print(report.summary())
    print()
    print("measured class counts (all splits):")
    totals: dict[str, int] = {}
    for split_counts in report.class_counts.values():
        for name, count in split_counts.items():
            totals[name] = totals.get(name, 0) + count
    for name in sorted(totals, key=lambda n: (totals[n], n)):
        print(f"  {name:<52} {totals[name]:>5}")
    print()
    print(f"manifest  : {manifest_path} ({manifest_path.stat().st_size / 1e6:.2f} MB)")
    print(f"audit     : {leakage_path}")
    return 0 if report.passed else 1


def cmd_sample(args: argparse.Namespace) -> int:
    """Acquire a small, class-balanced sample of real images."""
    from fieldlens.data.acquire import RangeReadNotSupported, acquire_sample
    from fieldlens.data.split import Manifest, default_manifest_path

    dataset = _dataset_config()
    manifest_path = default_manifest_path(dataset)
    if not manifest_path.exists():
        print("manifest missing; run `fieldlens audit` first.", file=sys.stderr)
        return 2

    manifest = Manifest.load(manifest_path)
    # Include the validation split so a sample-trained checkpoint has something
    # to early-stop on. The official test split is deliberately excluded: it must
    # stay untouched until evaluation.
    sample_entries = [e for e in manifest.entries if e.split in ("train", "val")]
    dest = Path(args.dest) if args.dest else Path("data/sample")

    try:
        stats = acquire_sample(
            sample_entries,
            dataset=dataset,
            dest_dir=dest,
            per_class=args.per_class,
            seed=dataset.seed,
            progress=args.progress,
        )
    except RangeReadNotSupported as exc:
        print(f"selective read unavailable: {exc}", file=sys.stderr)
        print(
            "fallback: download the full archive once with `fieldlens prepare-full`.",
            file=sys.stderr,
        )
        return 2

    _print_json(stats)
    (REPORT_DIR / "sample_acquisition.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """Extract manifest images (optionally a stratified subset of all splits)."""
    from fieldlens.data.acquire import (
        RangeReadNotSupported,
        extract_manifest_images,
        stratified_subset,
    )
    from fieldlens.data.split import Manifest, default_manifest_path
    from fieldlens.paths import IMAGE_DIR

    dataset = _dataset_config()
    manifest_path = default_manifest_path(dataset)
    if not manifest_path.exists():
        print("manifest missing; run `fieldlens audit` first.", file=sys.stderr)
        return 2

    manifest = Manifest.load(manifest_path)
    entries = list(manifest.entries)
    cap = args.per_class_per_split
    if cap:
        entries = stratified_subset(
            entries, per_class_per_split=cap, seed=dataset.seed
        )
    dest = Path(args.dest) if args.dest else IMAGE_DIR

    print(
        f"extracting {len(entries)} images to {dest} "
        f"({'stratified cap ' + str(cap) + ' per class per split' if cap else 'ALL'})",
        flush=True,
    )
    try:
        stats = extract_manifest_images(entries, dataset=dataset, dest_dir=dest)
    except RangeReadNotSupported as exc:
        print(f"selective read unavailable: {exc}", file=sys.stderr)
        print(
            "fallback: download the archive once with `fieldlens fetch-archive`.",
            file=sys.stderr,
        )
        return 2

    stats["per_class_per_split_cap"] = cap
    _print_json(stats)
    (REPORT_DIR / "extraction.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Classify one image and report a confidence verdict."""
    from fieldlens.inference.predictor import PredictionError, Predictor
    from fieldlens.models.checkpoint import CheckpointError
    from fieldlens.paths import PROMOTED_CHECKPOINT

    checkpoint = Path(args.checkpoint) if args.checkpoint else PROMOTED_CHECKPOINT
    try:
        predictor = Predictor.from_checkpoint(checkpoint)
    except CheckpointError as exc:
        print(f"cannot load a model: {exc}", file=sys.stderr)
        return 2
    try:
        result = predictor.explain(Path(args.image))
    except PredictionError as exc:
        print(f"cannot read the image: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _print_json(result)
        return 0
    predictor.render(result)
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    """Render a Grad-CAM map for one image, for failure-mode inspection."""
    from fieldlens.explain.gradcam import GradCAMExplainer
    from fieldlens.inference.predictor import PredictionError, Predictor
    from fieldlens.models.checkpoint import CheckpointError
    from fieldlens.paths import FIGURE_DIR, PROMOTED_CHECKPOINT

    checkpoint = Path(args.checkpoint) if args.checkpoint else PROMOTED_CHECKPOINT
    try:
        predictor = Predictor.from_checkpoint(checkpoint)
    except CheckpointError as exc:
        print(f"cannot load a model: {exc}", file=sys.stderr)
        return 2

    image_path = Path(args.image)
    try:
        prediction = predictor.predict(image_path)
    except PredictionError as exc:
        print(f"cannot read the image: {exc}", file=sys.stderr)
        return 2

    target_class = None
    if args.target_class:
        if args.target_class not in predictor.class_names:
            options = ", ".join(predictor.class_names)
            print(
                f"unknown class {args.target_class!r}; choose one of: {options}",
                file=sys.stderr,
            )
            return 2
        target_class = predictor.class_names.index(args.target_class)

    try:
        with GradCAMExplainer(predictor) as explainer:
            cam = explainer.explain(image_path, target_class=target_class)
            caveats = explainer.described_caveats()
    except (RuntimeError, ValueError) as exc:
        print(f"Grad-CAM unavailable for this model: {exc}", file=sys.stderr)
        return 3

    out = Path(args.out) if args.out else FIGURE_DIR / f"gradcam_{image_path.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cam.overlay_on(predictor.load_rgb(image_path)).save(out)

    row, column = cam.peak_region()
    width = cam.heatmap.shape[1]
    height = cam.heatmap.shape[0]
    # Reported as a normalised position so the figure size cannot make it stale.
    peak = {
        "x_fraction": round((column + 0.5) / width, 4),
        "y_fraction": round((row + 0.5) / height, 4),
    }

    if args.json:
        _print_json(
            {
                "image": str(image_path),
                "predicted": prediction.label,
                "confidence": prediction.confidence,
                "warning_level": prediction.warning_level,
                "explained_class": cam.label,
                "explained_class_is_prediction": cam.label == prediction.label,
                "gradcam_confidence": cam.confidence,
                "heatmap_shape": [int(height), int(width)],
                "peak": peak,
                "overlay": str(out),
                "caveats": caveats,
            }
        )
        return 0

    print(f"  image         : {image_path}")
    print(f"  prediction    : {prediction.label}  ({prediction.confidence:.1%})")
    print(f"  verdict       : {prediction.warning_level}")
    print(f"  explained     : {cam.label}")
    print(f"  peak region   : x={peak['x_fraction']:.2f} y={peak['y_fraction']:.2f} (0-1)")
    print(f"  overlay       : {out}")
    print()
    for caveat in caveats:
        print(f"  note: {caveat}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from fieldlens.training.train import train_from_config

    return train_from_config(
        epochs_override=args.epochs,
        batch_size_override=args.batch_size,
        image_root=Path(args.image_root) if args.image_root else None,
        run_name=args.run_name,
        only_present=args.only_present,
    )


def cmd_robustness(args: argparse.Namespace) -> int:
    """Run the controlled reliability sweeps on frozen weights."""
    import json

    import torch

    from fieldlens.data.split import Manifest, default_manifest_path
    from fieldlens.data.transforms import build_eval_transform, spec_from_checkpoint
    from fieldlens.eval.robustness import figure_sweeps, run_sweeps, sweep_metadata
    from fieldlens.models.checkpoint import load_checkpoint
    from fieldlens.paths import FIGURE_DIR, IMAGE_DIR, PROMOTED_CHECKPOINT

    dataset = _dataset_config()
    eval_config = _eval_config()
    manifest_path = default_manifest_path(dataset)
    if not manifest_path.exists():
        print("manifest missing; run `fieldlens audit` first.", file=sys.stderr)
        return 2

    checkpoint = Path(args.checkpoint) if args.checkpoint else PROMOTED_CHECKPOINT
    if not checkpoint.exists():
        print(f"checkpoint missing: {checkpoint}; run `fieldlens train` first.")
        return 2

    manifest = Manifest.load(manifest_path)
    image_root = Path(args.image_root) if args.image_root else IMAGE_DIR
    test_entries = [
        e for e in manifest.entries_for("test") if (image_root / e.path).exists()
    ]
    if not test_entries:
        print(f"no test images under {image_root}; run `fieldlens extract`.")
        return 2

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, metadata = load_checkpoint(
        checkpoint, dataset_classes=dataset.classes, device=device
    )
    transform = build_eval_transform(spec_from_checkpoint(metadata))

    print("FieldLens AI - reliability sweeps")
    print(f"  images  : {len(test_entries)} held-out test images")
    print(f"  device  : {device}")
    print(f"  legend  : {sweep_metadata()}")
    print()

    report = run_sweeps(
        model,
        test_entries,
        dataset.class_index(),
        transform,
        image_root,
        device=device,
        degradation=eval_config.degradation,
        temperature=float(metadata.temperature),
        num_classes=dataset.num_classes,
        batch_size=args.batch_size,
        max_images=args.max_images,
    )
    report["temperature"] = float(metadata.temperature)
    report["checkpoint"] = str(checkpoint)

    figure_ok = figure_sweeps(report, FIGURE_DIR / "robustness_sweeps.png")
    report["figure"] = figure_ok

    out = REPORT_DIR / "robustness.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print()
    for name, payload in report.get("sweeps", {}).items():
        if "levels" not in payload:
            continue
        drop = payload.get("macro_f1_drop_from_baseline")
        print(
            f"  {name:<12} worst macro F1 {payload['worst_macro_f1']:.4f} at "
            f"{payload['worst_level']}"
            + (f"  (drop {drop:.4f} from baseline)" if drop is not None else "")
        )
    print()
    print(f"  report : {out}")
    print(f"  figure : {figure_ok}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    """Launch the Streamlit interface."""
    import subprocess

    from fieldlens.ui import app as _app  # noqa: F401 - ensures the app imports

    app_path = Path(__file__).parent / "ui" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    if args.port:
        cmd += ["--server.port", str(args.port)]
    cmd += ["--server.headless", "true"]
    print("starting:", " ".join(cmd))
    return subprocess.call(cmd)


def cmd_evaluate(args: argparse.Namespace) -> int:
    from fieldlens.eval.run import run_evaluation

    return run_evaluation(skip_sweeps=args.skip_sweeps)


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fieldlens",
        description="Plant leaf classification with confidence checks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_env = sub.add_parser("env", help="environment and disk report")
    p_env.add_argument("--json", action="store_true")
    p_env.set_defaults(func=cmd_env)

    p_prep = sub.add_parser("prepare", help="fetch dataset metadata (a few MB)")
    p_prep.add_argument("--force", action="store_true", help="re-download")
    p_prep.set_defaults(func=cmd_prepare)

    p_inspect = sub.add_parser("inspect", help="probe the remote archive")
    p_inspect.set_defaults(func=cmd_inspect)

    p_audit = sub.add_parser("audit", help="build manifest and run leakage gate")
    p_audit.set_defaults(func=cmd_audit)

    p_sample = sub.add_parser("sample", help="acquire a small real-image sample")
    p_sample.add_argument("--per-class", type=int, default=20)
    p_sample.add_argument("--dest", default=None)
    p_sample.add_argument(
        "--progress", type=int, default=25, help="print progress every N images"
    )
    p_sample.set_defaults(func=cmd_sample)

    p_train = sub.add_parser("train", help="train the classifier")
    p_train.add_argument("--epochs", type=int, default=None)
    p_train.add_argument("--batch-size", type=int, default=None)
    p_train.add_argument(
        "--image-root",
        default=None,
        help="directory holding the manifest's image paths (default: data/images)",
    )
    p_train.add_argument("--run-name", default=None)
    p_train.add_argument(
        "--only-present",
        action="store_true",
        help="train only on images already on disk (reports how many were kept)",
    )
    p_train.set_defaults(func=cmd_train)

    p_robust = sub.add_parser(
        "robustness", help="reliability sweeps (brightness, resolution, JPEG, blur)"
    )
    p_robust.add_argument("--checkpoint", default=None)
    p_robust.add_argument("--image-root", default=None)
    p_robust.add_argument("--batch-size", type=int, default=32)
    p_robust.add_argument(
        "--max-images",
        type=int,
        default=300,
        help="cap the images swept per level (0 = all held-out images)",
    )
    p_robust.set_defaults(func=cmd_robustness)

    p_serve = sub.add_parser("serve", help="launch the Streamlit interface")
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.set_defaults(func=cmd_serve)

    p_eval = sub.add_parser("evaluate", help="metrics, calibration and sweeps")
    p_eval.add_argument("--skip-sweeps", action="store_true")
    p_eval.set_defaults(func=cmd_evaluate)

    p_extract = sub.add_parser(
        "extract", help="extract manifest images (range reads, resumable)"
    )
    p_extract.add_argument(
        "--per-class-per-split",
        type=int,
        default=None,
        help="cap images per class per split; omit to extract everything",
    )
    p_extract.add_argument("--dest", default=None)
    p_extract.set_defaults(func=cmd_extract)

    p_expl = sub.add_parser(
        "explain", help="Grad-CAM overlay for one image (failure-mode inspection)"
    )
    p_expl.add_argument("image")
    p_expl.add_argument("--checkpoint", default=None)
    p_expl.add_argument(
        "--target-class",
        default=None,
        help="explain this class instead of the top-1 prediction",
    )
    p_expl.add_argument("--out", default=None, help="output PNG path")
    p_expl.add_argument("--json", action="store_true")
    p_expl.set_defaults(func=cmd_explain)

    p_pred = sub.add_parser("predict", help="classify one image")
    p_pred.add_argument("image")
    p_pred.add_argument("--checkpoint", default=None)
    p_pred.add_argument("--json", action="store_true")
    p_pred.set_defaults(func=cmd_predict)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
