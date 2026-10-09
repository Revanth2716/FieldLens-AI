# Limitations

Written from measured results. Nothing here is hypothetical padding — each item
names the evidence behind it.

## 1. This is not a field-condition system

The dataset contains **single detached leaves photographed under controlled
lighting with uniform backgrounds**. Every metric in this repository is measured
on that kind of image.

No claim is made about photographs taken in a field: different lighting,
cluttered backgrounds, multiple leaves, occlusion, different cameras, or different
plant varieties. PlantVillage-style benchmarks are widely known to be far easier
than field imagery, and this project does not pretend otherwise.

**What would be needed to claim otherwise:** evaluation on a field-condition
dataset with its own held-out split. That is out of scope here.

## 2. Calibration does not transfer

Temperature `T = 0.5788` was fitted on the validation split of *this* dataset and
reduced ECE from 0.0933 to 0.0244 — a real improvement, measured.

That value is specific to this data. A different camera, exposure regime, or
background will change the confidence distribution, and the stored temperature
will no longer be correct. The checkpoint records which split the temperature was
fitted on precisely so this is not mistaken for a universal constant.

## 3. Confidence is not correctness — and can be badly wrong

A spot check of 18 held-out images found 2 errors:

- one flagged **low** confidence (0.368) — the system worked as intended;
- one **wrong at 0.943** confidence — the system did not.

Softmax confidence is not a probability of being correct. Temperature scaling
improves aggregate calibration but cannot repair a systematically confident
mistake. Selective prediction raises accuracy to 0.9970 at 50% coverage, but the
remaining 0.3% of errors are not all flagged.

## 4. No reject option (out-of-distribution inputs)

The model has **no reject class**. An image of a hand, a rock, or a different
plant still receives one of the 12 labels, possibly confidently. There is no
out-of-distribution detector in this project.

## 5. Smallest class is scarce

`Potato___healthy` has **152 images** — roughly 2.8× fewer than the next smallest
class. Macro F1 (0.9140) and per-class recall are reported for this reason, but
statistical noise on the rare class remains high. The per-class table in the UI
shows support counts so this is visible rather than buried.

## 6. Two of twelve classes are genuinely hard

| Class | F1 | Recall |
|---|---|---|
| Tomato — Early blight | 0.772 | 0.733 |
| Potato — Late blight | 0.844 | 0.767 |
| Tomato — Late blight | 0.862 | 0.833 |

Early and Late blight are agronomically confusable and visually similar in this
dataset. Aggregate accuracy (0.9162) hides that a quarter of Tomato Early blight
images are missed.

## 7. Evaluation covers part of the test set

Metrics were computed on **692 of 2,454** official test images (28.2%), because
acquisition was capped to keep the transfer small. Those images are genuinely
held out, but the evaluation is narrower than the full split. The report records
the exact fraction so this cannot be misread as comprehensive.

## 8. Uncertainty is a single forward pass

Only temperature scaling is in scope. There are **no ensembles and no
MC-dropout**, so the reported uncertainty reflects what one deterministic forward
pass can express. Genuinely hard epistemic uncertainty is not captured.

## 9. Upstream splits were not perfectly clean

An independent audit measured **5 leaf identifiers** present in both the official
train and test lists, contradicting the dataset card's claim. The `leak_free`
policy removes them (5 images dropped, test set untouched), and the measured
result is published rather than smoothed over.

The audit also found **1,207 `copy`-suffixed leaves**. Path-level disjointness
does not prove pixel-level distinctness; a perceptual-hash audit across split
boundaries is the natural next step and was not completed here.

## 10. Not implemented in this revision

- Robustness sweeps (brightness, resolution, JPEG, blur)
- Background-dependency experiments (colour → segmented, synthetic compositing)
- Held-out-crop out-of-distribution retraining
- Grad-CAM saliency overlays
- Perceptual-hash duplicate audit over image pixels

These are defined in the project plan as optional extensions. They are **absent,
not failed** — and nothing in the README claims otherwise.
