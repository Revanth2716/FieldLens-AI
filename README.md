# FieldLens AI — Plant Health Classification with Confidence Checks

A computer-vision project that treats **honest uncertainty** as the deliverable.
It classifies plant leaf images into 12 classes across 4 crops, and — more
importantly — it measures *whether its own confidence can be trusted*.

The classifier is a vehicle. The interesting parts are leakage-safe evaluation,
calibration, selective prediction, and a set of limitations that are **measured
rather than asserted**.

---

## What this project actually demonstrates

| Capability | How it shows up here |
|---|---|
| Leakage-safe evaluation | Splits are built by **leaf identity**, not by image. An independent audit found the upstream split was not perfectly clean (see below). |
| Honest metrics | Macro F1 is the headline, because the smallest class has only 152 images. |
| Calibration | Temperature scaling fitted on **validation only** cut ECE by **74%**. |
| Selective prediction | Accuracy rises from 91.6% to **99.7%** when the model is allowed to answer only its 50% most confident cases. |
| Reliability work | A confidence audit that finds **confidently-wrong** predictions instead of hiding them. |

---

## Measured results

Everything below is produced by committed commands and stored in
[`artifacts/reports/`](artifacts/reports). Numbers are not estimated.

**Model:** MobileNetV3-Small, 1,530,156 parameters (659,596 trainable), 224x224
input, ImageNet-pretrained, fine-tuned on the GTX 1650.

| Metric (test split) | Uncalibrated | After temperature scaling |
|---|---|---|
| Accuracy | 0.9162 | 0.9162 |
| Macro precision | 0.9161 | 0.9161 |
| Macro recall | 0.9194 | 0.9194 |
| **Macro F1** | 0.9140 | 0.9140 |
| Top-3 accuracy | 0.9942 | 0.9942 |
| **ECE (calibration error)** | **0.0933** | **0.0244** |
| Brier score | 0.1459 | 0.1279 |

Temperature `T = 0.5788`, fitted on the validation split
(NLL 0.3112 to 0.2431).

> **Important caveat on the evaluation size.** The test metrics are computed on
> **692 of the 2,454** official test images (28.2%). Acquisition was capped to
> keep the archive transfer small; the images that *were* evaluated are genuine
> held-out data. Re-run `fieldlens extract` without a cap to evaluate the full
> split. The evaluation report records this ratio
> (`dataset.images_evaluated_fraction_of_test`).

### Per-crop performance

| Crop | Accuracy | Macro F1 | Images |
|---|---|---|---|
| Corn (maize) | 0.9889 | 0.9889 | 180 |
| Potato | 0.9384 | 0.9335 | 146 |
| Grape | 0.9218 | 0.9204 | 179 |
| **Tomato** | **0.8851** | **0.8820** | 174 |

### Where the model fails

| Class | F1 | Recall | n |
|---|---|---|---|
| Tomato — Early blight | 0.772 | 0.733 | 60 |
| Potato — Late blight | 0.844 | 0.767 | 60 |
| Tomato — Late blight | 0.862 | 0.833 | 60 |
| Grape — Black rot | 0.870 | 0.783 | 60 |

These are the **Early vs Late blight** confusions the class set was deliberately
chosen to expose. A model reporting only aggregate accuracy would have hidden
them.

### The value of the confidence system

Thresholds are selected on validation at a target coverage; the accuracy is then
measured on test at whatever coverage actually results.

| Target coverage | Threshold | Test coverage | Accuracy on kept images |
|---|---|---|---|
| 50% | 0.989 | 48.6% | **0.9970** |
| 60% | 0.974 | 60.5% | 0.9976 |
| 70% | 0.938 | 69.7% | 0.9834 |
| 80% | 0.849 | 81.2% | 0.9715 |
| 90% | 0.659 | 91.0% | 0.9508 |

Answering only the most confident half raises accuracy from 91.6% to 99.7% —
which is the entire point of exposing confidence rather than a bare label.

### But confidence is not a guarantee

A spot check of 18 held-out images found 2 errors: one was correctly flagged
`low` confidence (0.368), and **one was wrong at 0.943 confidence**. Softmax
confidence is not a probability of being correct, and temperature scaling cannot
repair a systematically confident mistake.

---

## A finding worth highlighting: the upstream split was not perfectly clean

The dataset card states its splits "preserve the leaf grouping logic (ensuring
images of the same leaf do not appear in both sets)".

This project **audited that claim instead of trusting it** and measured
**5 leaf identifiers present in both the official train and test lists**
(`gh_hl leaf 220`, `gh_hl leaf 466.1`, `ghlb2 leaf 8999`, ...). Small — 0.2% of
test leaves — but real.

The `leak_free` split policy keeps the official test set **untouched** and drops
those 5 leaves from train/validation instead. The audit result is committed in
[`artifacts/reports/leakage_audit.json`](artifacts/reports/leakage_audit.json)
and surfaced in the UI. Reproducing the upstream boundary is still possible with
`split_policy: official`.

The audit also measured **1,207 `copy`-suffixed leaves** — image files whose
names differ only by a `copy`/`copy 2`/`copy 3` suffix. These are a
near-duplicate risk that path-level disjointness alone cannot prove distinctness
for, and this project did not complete a perceptual-hash audit across split
boundaries. (LIMITATIONS.md item 9; DATASET.md and
`artifacts/reports/leakage_audit.json`.)

---

## Quickstart

### 1. Environment

This pins **every** cache and temporary directory onto the project drive
(`D:\Revanth projects\FieldLens AI\.cache`) before any dependency is imported,
because the system drive `C:` on this machine is nearly full and cannot be
filled by wheels, downloaded datasets, or temp files.

```bash
# One-time: create the virtual environment on D: (if not already present)
# This reads setup_env.sh, which sets UV_CACHE_DIR/PIP_CACHE_DIR/etc.
# before installing anything, so nothing lands on C:.
python .ci/local_env/bootstrap_venv.py

# For interactive use, source the helper so your shell inherits the same
# project-drive redirections as the installed tooling:
. scripts/env.sh   # or `.\scripts\env.ps1` on PowerShell
```

After either of the above, the package is available as `fieldlens` and every
cache/temp variable points at `D:\...\FieldLens AI\.cache`.

### 2. Prepare the dataset metadata

```bash
fieldlens prepare
```

This downloads only the zip's *central directory* (~27 MB), not the 2.18 GB
archive. It writes `data/metadata/` and the dataset config in `configs/dataset.yaml`.

### 3. Build the manifest and run the leakage gate

```bash
fieldlens audit
```

This builds the split manifest and independently re-runs the leaf-identity
leakage check whose result is reported below. The manifest is written to
`data/manifests/`.

### 4. Acquire real images

```bash
# Read individual zip members by HTTP range request into data/images/.
fieldlens extract
```

This reuses one HTTPS connection and reads each image by range, so acquiring the
full 12-class stratified split cost **38 MB and under three minutes** instead of
the 2.18 GB full download. The extraction receipt is at
`artifacts/reports/extraction.json`.

```bash
# A small sample for smoke checks and demos:
fieldlens sample --per-class 20
```

### 5. Train

```bash
fieldlens train
```

The default config (`configs/train.yaml`) trains in two stages: head-only
first, then fine-tune the last feature block. Checkpoints land in
`artifacts/checkpoints/`.

### 6. Evaluate (metrics, calibration, reliability)

```bash
fieldlens evaluate
```

This writes every number below to `artifacts/reports/evaluation.json` and
`artifacts/reports/robustness.json`, plus the figures under
`artifacts/reports/figures/`.

### 7. Predict an image

```bash
fieldlens predict path/to/image.jpg
```

### 8. Open the UI

```bash
fieldlens serve
```

Opens the Streamlit interface on port 8501. Upload an image or pick a bundled
sample to see a prediction with confidence verdict, top-k, and an optional
Grad-CAM overlay.

---

## Commands

| Command | What it does |
|---|---|
| `fieldlens env` | Environment and disk report (cache redirections, free space) |
| `fieldlens prepare` | Download dataset metadata only (~27 MB) |
| `fieldlens inspect` | Probe the remote archive without downloading |
| `fieldlens audit` | Build manifest + run the leakage gate |
| `fieldlens sample --per-class N` | Acquire a small sample for demos/smoke checks |
| `fieldlens extract [--per-class-per-split N]` | Acquire images by range request |
| `fieldlens train` | Train the classifier (staged) |
| `fieldlens evaluate` | Metrics, calibration, reliability sweeps |
| `fieldlens predict <image>` | Classify one image with confidence verdict |
| `fieldlens explain <image>` | Grad-CAM overlay for failure-mode inspection |
| `fieldlens robustness` | Reliability sweeps (brightness, resolution, JPEG, blur) |
| `fieldlens serve` | Launch the Streamlit UI |

---

## Dataset

Source: [`mohanty/PlantVillage`](https://huggingface.co/datasets/mohanty/PlantVillage)
(Hugging Face), `color` config. License: **CC-BY-SA-3.0**.

The image payload is a **single 2.18 GB `data.zip`**. There is **no official
streaming path**: `load_dataset(...)` downloads and extracts the whole archive,
which is why this project reads individual members by HTTP range request instead.

Split manifests are generated by `fieldlens audit`, not vendored. The dataset
itself lives in `data/images/` and is gitignored; only the manifests and
derived metrics are kept in the repository.

12 classes across 4 crops:

- Corn (maize): Cercospora leaf spot / Gray leaf spot, Common rust, healthy
- Grape: Black rot, Esca (Black measles), healthy
- Potato: Early blight, Late blight, healthy
- Tomato: Early blight, Late blight, healthy

Smallest class: **Potato___healthy**, 152 images. This is why macro F1 and
per-class recall are the headline numbers rather than plain accuracy.

---

## Leakage audit (measured, not assumed)

The dataset card claims the splits preserve leaf grouping. An independent audit
measured the official split directly and found it does **not** fully hold for
these 12 classes:

- **5 leaf identifiers** appear in both the official train and test lists
  (`gh_hl leaf 220`, `gh_hl leaf 466.1`, `ghlb2 leaf 8999`, ...).
- 0.2% of test leaves are affected.
- Train/validation overlap: 0. Test/validation overlap: 0.
- No leaf spans multiple classes.

The default `leak_free` split policy keeps the official test set untouched and
drops those 5 leaves from train/validation (5 images removed). The resulting
audit passes with zero cross-split leaf or path overlap.

The audit also measured **1,207 `copy`-suffixed leaves** — image files whose
names differ only by a `copy`/`copy 2`/`copy 3` suffix. These are a
near-duplicate risk that path-level disjointness alone cannot prove distinctness
for, and this project did not complete a perceptual-hash audit across split
boundaries. (See LIMITATIONS.md item 9; DATASET.md and
`artifacts/reports/leakage_audit.json`.)

Set `split_policy: official` in `configs/dataset.yaml` to reproduce the upstream
boundary.

---

## Tests

```bash
# Default suite: unit + integration + model tests (no dataset, no network).
.venv/Scripts/pytest.exe tests/ -q -m "not slow and not network"

# Dataset-dependent (slow) tests — opt-in only:
.venv/Scripts/pytest.exe tests/ -q -m slow
```

- Default suite: 161 tests pass.
- Slow suite: tests the full manifest and the extraction/resumption path.
- Ruff: `ruff check src tests scripts`.

---

## Reproducibility

```bash
source scripts/env.sh       # or .ps1 — pins caches/temp to D:
fieldlens prepare
fieldlens audit
fieldlens extract
fieldlens train
fieldlens evaluate
```

Every number in this README is regenerated by these commands. The run manifest
records seed, git SHA, library versions, epoch timings, and peak VRAM.

---

## Limitations

- The data is single detached leaves on uniform backgrounds — not field imagery.
- The model has **no reject option**; a non-leaf image still receives a label.
- Softmax confidence is not a probability of being correct.
- Calibration is specific to this data; it does not transfer across cameras,
  lighting, or backgrounds.
- Only temperature scaling is used; uncertainty is a single deterministic forward
  pass.
- `Potato___healthy` has only 152 images; statistical noise on the rare class
  remains high.
- The metrics were computed on 692 of 2,454 official test images (28.2%) because
  acquisition was capped. The evaluation report records this fraction.
- An independent audit found **5 leaf identifiers** in both the official train and
  test lists; the `leak_free` policy removes them (test set untouched).
- The audit also measured **1,207 `copy`-suffixed leaves**, a near-duplicate risk
  that path-level checks alone cannot resolve; a perceptual-hash audit across
  split boundaries was not completed.
- No field-condition, no out-of-distribution retrain, no background-compositing
  experiment, and no perceptual-hash duplicate audit were completed in this
  revision.

See [LIMITATIONS.md](LIMITATIONS.md) for the full list.

---

## License and attribution

The model code and this repository are provided under the terms in [LICENSE](LICENSE).
The dataset is **CC-BY-SA-3.0**; attribution and share-alike obligations are
described in [ATTRIBUTION.md](ATTRIBUTION.md) and [DATASET.md](DATASET.md).

The dataset is **not redistributed** in this repository (`data/` is gitignored).
Only split manifests, derived metrics, and trained weights (reproducible) are
kept locally.

---

## Files in this repository

| Path | Purpose |
|---|---|
| `src/fieldlens/` | Main package: data, models, training, eval, explain, inference, UI |
| `configs/` | YAML configs: dataset, model, train, eval |
| `scripts/` | Bootstrap, environment helper, local env setup |
| `tests/` | Unit, integration, model, and slow (dataset-dependent) tests |
| `artifacts/reports/` | JSON reports + figures (metrics, calibration, sweeps) |
| `artifacts/checkpoints/` | Trained checkpoints (gitignored; reproducible from `fieldlens train`) |
| `data/` | Gitignored: metadata, manifests, images, sample |
| `.cache/` | Gitignored: pip/uv/torch/huggingface/matplotlib/tmp caches |
| `.github/workflows/ci.yml` | CI: CPU-only torch, lint, tests, CLI checks |
| `README.md` | This file |
| `MODEL_CARD.md` | Model card |
| `DATASET.md` | Dataset provenance, license, measured counts |
| `LIMITATIONS.md` | Full limitations list |
| `ATTRIBUTION.md` | Dataset attribution |
| `LICENSE` | Repository license |
