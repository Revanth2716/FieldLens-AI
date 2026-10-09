# Model Card — FieldLens AI 12-class leaf classifier

## Model details

| Field | Value |
|---|---|
| Architecture | MobileNetV3-Small (torchvision), ImageNet-pretrained backbone |
| Parameters | 1,530,156 total; 659,596 trainable during fine-tuning |
| Input | 224×224 RGB; normalisation constants taken from the pretrained weights' own metadata |
| Output | 12 classes (softmax over 4 crops × 3 states) |
| Checkpoint | `artifacts/checkpoints/fieldlens_mobilenetv3s_12class.pt` (~6 MB) |
| Checkpoint schema | version 1 — carries class order, preprocessing spec, temperature, provenance |
| Framework | PyTorch 2.14.1+cu130 |
| Training hardware | NVIDIA GeForce GTX 1650, 4 GB (compute capability 7.5) |

The classifier head is replaced **in place**. An earlier revision inserted an
extra dropout layer, which shifted the head's positional indices and made
checkpoints saved with one dropout setting fail to load under another. The
loader now rejects a mismatched class list loudly, and a regression test asserts
the state-dict layout is independent of dropout.

## Intended use

**Intended.** A demonstration of applied-ML judgement: leakage-safe dataset
handling, calibration measured rather than assumed, selective prediction, and
limitations backed by measurements. Suitable as a portfolio artifact and for
reasoning about model behaviour.

**Not intended.** Any decision about a real plant. This is not a diagnostic, not
an advisory tool, and not a field-condition system.

## Out-of-scope / non-goals

- No agronomic, diagnostic, or treatment recommendation.
- No severity grading or multi-lesion localisation.
- No field, drone, or smartphone-camera imagery.
- No rejection of non-leaf inputs (see below).
- No ensembles or MC-dropout: uncertainty reflects a single deterministic forward pass.

## Training data

`mohanty/PlantVillage` (`color` config), license **CC-BY-SA-3.0**. 12 classes
across corn, grape, potato and tomato. See [DATASET.md](DATASET.md).

Training used a **capped stratified subset** of the archive: 720 train / 625
validation images. Evaluation used 692 held-out images from the official test
list. The full archive was deliberately not downloaded — see the acquisition
notes in DATASET.md.

## Measured metrics

Test split, 692 images. Uncalibrated probabilities come from the raw softmax;
calibrated probabilities apply a temperature fitted on validation only.

| Metric | Uncalibrated | Calibrated |
|---|---|---|
| Accuracy | 0.9162 | 0.9162 |
| Macro precision | 0.9161 | 0.9161 |
| Macro recall | 0.9194 | 0.9194 |
| Macro F1 | 0.9140 | 0.9140 |
| Top-3 accuracy | 0.9942 | 0.9942 |
| Expected calibration error | 0.0933 | **0.0244** |
| Brier score | 0.1459 | 0.1279 |

Temperature `T = 0.5788` (validation NLL 0.3112 → 0.2431).

Temperature scaling does **not** change accuracy, and that is expected: it
reorders nothing, it only makes the stated confidence honest. Here it reduced
calibration error by roughly 74%.

### Per-crop

| Crop | Accuracy | Macro F1 | n |
|---|---|---|---|
| Corn (maize) | 0.9889 | 0.9889 | 180 |
| Potato | 0.9384 | 0.9335 | 146 |
| Grape | 0.9218 | 0.9204 | 179 |
| Tomato | 0.8851 | 0.8820 | 174 |

## Selective prediction

Thresholds are fitted on validation at a target coverage; accuracy is then
measured on test at the resulting coverage.

| Target | Threshold | Test coverage | Accuracy | Images kept |
|---|---|---|---|---|
| 50% | 0.989 | 48.6% | 0.9970 | 336 |
| 60% | 0.974 | 60.5% | 0.9976 | 419 |
| 70% | 0.938 | 69.7% | 0.9834 | 482 |
| 80% | 0.849 | 81.2% | 0.9715 | 562 |
| 90% | 0.659 | 91.0% | 0.9508 | 630 |

## Known failure modes

1. **Confidently wrong.** A spot check of 18 held-out images produced 2 errors:
   one flagged `low` (0.368), the other wrong at **0.943** confidence. High
   confidence does not imply correctness.
2. **Early vs Late blight.** Tomato Early blight (F1 0.772), Potato Late blight
   (0.844), Tomato Late blight (0.862) are the weakest classes — the genuinely
   confusable pairs in this class set.
3. **Grape Black rot** (F1 0.870) is confused with Grape Esca and Grape healthy.
4. **No reject option.** An image that is not a leaf still receives a label.
5. **Background dependence.** Training images are single detached leaves on
   uniform backgrounds; performance on cluttered backgrounds is not measured here.

## Ethical and practical considerations

- The confidence verdict is deliberately worded to avoid "certain" and "safe".
- The UI states the intended use and limitations on the model-card page.
- No data about individuals is processed; no personal data is collected.
- Metrics are reported together with the size of the evaluation set, so a small
  or capped evaluation cannot be mistaken for a comprehensive one.

## Reproducing

```bash
source scripts/setup_env.sh
.venv/Scripts/fieldlens.exe prepare
.venv/Scripts/fieldlens.exe audit
.venv/Scripts/fieldlens.exe extract --per-class-per-split 60
.venv/Scripts/fieldlens.exe train --only-present --run-name full_run --batch-size 32
.venv/Scripts/fieldlens.exe evaluate
```

Every number above is regenerated by these commands; the run manifest records
seed, git SHA, library versions, epoch timings and peak VRAM.
