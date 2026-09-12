# Model Sources — Pink Edge AI

Honesty convention carried over from the original project's own `MODEL_DOCUMENTATION.md`: every
modality below states exactly what is running behind it, including where it gets things wrong.
Nothing here should be read as a clinically validated product.

Each modality tries multiple methods in order, per `predict_tb()` / `predict_maternal()` /
`predict_mammography()` in `inference.py` — the order isn't fixed to "online first": it's set by
**measured accuracy against real held-out ground truth**, not by assumption.

| Modality | Try order | Measured accuracy |
|---|---|---|
| Mammography | Roboflow Workflow `breastcancer-yolov8-78tni` → offline pixel-diff heuristic → **locally-trained classifier** → local weights (rare) → Simulated | Roboflow: one ground-truth sample correctly flagged (90.9% confidence — not a % accuracy over many samples, see below). Offline heuristic: **98%** (49/50 held-out). Locally-trained classifier: **98.7%** (77/78 held-out) — tied with the heuristic within sample-size noise. |
| Tuberculosis | **Locally-trained classifier** → offline pixel-diff heuristic → Roboflow model `tuberculosis-tp2pv/1` → offline HF ViT | Locally-trained classifier: **82.5%** (80 held-out samples) — best measured of the 4. Offline heuristic: 74%. Offline HF ViT: 62%. Roboflow: 0% on healthy samples — **known issue, see below**. |
| Maternal Health | Roboflow model `hash-maternal-health/1` → offline pixel-diff heuristic (currently always skips — no negative data, see below) → locally-trained classifier (not trained yet — same reason) → offline HF CNN | Roboflow: one ground-truth sample correctly flagged (88.3% confidence). |

Before any real triage tier runs, every modality is also checked for **out-of-domain input** — "does
this even look like the right kind of scan at all" — see the section near the bottom of this file.

The Roboflow calls need a key (`roboflow_key.txt` / `ROBOFLOW_API_KEY` / Streamlit `st.secrets`) and
internet; the offline pixel-diff heuristic (`offline_cv.py`), the locally-trained classifier
(`Models/<Modality>/local_model.pt`), and the Hugging Face models need neither, ever, once their
weights/datasets are on disk.

## The locally-trained classifier (`train_local_model.py`) — a real model, trained on this project's own data

Run `python train_local_model.py` to (re)train it. Architecture: torchvision's MobileNetV3-Small
with an ImageNet-pretrained backbone (frozen) and a linear head trained from scratch — practical on
CPU-only hardware. Trained on the dataset's own `train`+`valid` splits (capped at 150 images/class,
same sample budget as `offline_cv.py`'s templates) plus anything manually dropped into
`Models/<Modality>/positive|negative/`, and evaluated on the dataset's own held-out `test` split —
**never seen during training**, exactly like every other accuracy figure in this file. Saves
`Models/<Modality>/local_model.pt` + `local_model_metadata.json` (architecture, image size, sample
counts, measured accuracy, training date — so the file documents its own provenance).

This is a genuine **addition** to the lineup, not an assumed upgrade: `inference.py` only moves it
ahead of an existing tier when it measurably beats that tier on the same held-out methodology — true
for TB (82.5% > offline heuristic's 74%); for Mammography it's statistically tied with the offline
heuristic (98.7% vs. 98%, on 78 vs. 50 samples — not a real difference), so it stays second, right
after the heuristic, since the heuristic needs no torch/torchvision at all. Maternal Health has no
trained copy yet: the dataset has zero negative images (see below), and the training script refuses
to train on one-class-only data rather than silently producing a model that always says "positive."

⚠️ **A labeling bug was found and fixed here** (2026-09-13): `Models/Mammography/positive/` and
`negative/` had been manually pre-populated with real images using the **opposite** convention this
project uses everywhere else (COCO categories, `offline_cv.py`, this script) — 81 healthy scans
sitting in `positive/` and 156 cancer scans sitting in `negative/`. Every template/model trained
before the fix was quietly contaminated with mislabeled data pulled from those folders. Caught
before shipping, fixed by swapping the two folders' contents to match the standard convention
(`positive` = disease/abnormal, `negative` = healthy — see each folder's own `README.md`), then
retraining/recalibrating everything that reads from them — the numbers in this file are all
POST-fix. Mammography's accuracy actually improved once fixed (offline heuristic 96%→98%, local
classifier 82.1%→98.7%), consistent with "mislabeled training data hurts, not helps."

## The offline pixel-diff heuristic (`offline_cv.py`) — no model, no internet, ever

Built per your request for a genuinely offline method: no trained weights, no API, nothing but
files already in this repo. For each modality it averages images from the project's own local
labeled dataset (`Models/<Modality>/Data Set/*.coco`) into a "typical positive" and "typical
negative" reference template (grayscale, resized, histogram-equalized), then for a new image:
grayscale → resize → histogram-equalize → try identity vs. horizontal-flip and keep whichever
correlates better with a generic reference (handles left/right laterality without full image
registration) → pixel-wise absolute difference against **both** templates → the region that's
furthest from the negative template and closest to the positive one becomes the highlighted
bounding box (real detected coordinates — `draw_bbox()` in `GUI.py` now draws these when present,
not just an illustrative fixed position) → the mean of that "leans positive" difference map becomes
the confidence score.

**Measured, not assumed** — `python offline_cv.py` builds templates from train+valid only and
tests against the held-out `test` split (proper train/test split, not testing on training data):

```
[tb]           positive 16/25 | negative 21/25 | overall 37/50 (74%)
[mammography]  positive 25/25 | negative 24/25 | overall 49/50 (98%)
[maternal]     no dataset found — see below
```

**Maternal Health has no negative reference images at all** — every image in the local dataset is
an annotated `abnormal` case (arachnoid cyst, Chiari malformation, etc.); there's no "normal" class
to average into a negative template. `offline_cv.predict("maternal", ...)` correctly returns `None`
(not a crash, not a guess) and the dispatcher falls through to the next method. Fixable only by
adding genuinely normal/healthy maternal-health images to the local dataset.

## ⚠️ Known issue: `tuberculosis-tp2pv/1` (Roboflow) has a high false-positive rate

Validated against the project's own ground-truth COCO annotations
(`Models/TB/Data Set/tuberculosis.coco/test/_annotations.coco.json`), not just "does it run":
correctly flagged a `Tüberküloz` (active TB)-annotated sample as positive (91.2%), but called
**all 3** `Sağlıklı` (healthy)-annotated samples tested TB Positive too, at **98–99% confidence** —
high-confidence wrong, not a borderline call a threshold tweak would fix. Only one version of this
project exists on Roboflow (checked via the API). Likely a training-quality issue aggravated by
class imbalance (432 `Tüberküloz` vs. 159 `Sağlıklı` annotations in just the test split), not an
integration bug. **This is why TB's try-order puts offline methods first** — it's not "prefer
offline on principle," it's that both the locally-trained classifier (82.5%) and the offline
heuristic (74%) measurably beat the Roboflow model (0% on healthy) and the offline HF ViT (62%) on
the same held-out data. Revisit this ordering if the Roboflow model is retrained with better class
balance.

## Details

### Mammography — Workflow `breastcancer-yolov8-78tni`
Grounded via a real call: detections come back as
`{"x", "y", "width", "height", "confidence", "class", "class_id"}` inside
`result[0]["predictions"]["predictions"]`; class `"cancer"` seen on a real positive sample, empty
list on real negatives. No local trained-weight export was found for the deepest fallback
(`b-davmu/breastcancer-yolov8`, a *different*, public Roboflow Universe project used only as a
placeholder before the user's own workspace/model was available) — falls to SIMULATED only if
Roboflow is unreachable *and* the offline heuristic has no local dataset either.

### Tuberculosis — offline heuristic (primary) / Model `tuberculosis-tp2pv/1` / `sukhmani1303/tuberculosis-vit-model`
Real class taxonomy for the Roboflow model, grounded from the project's own COCO export (Turkish
labels): `Sağlıklı`=Healthy, `Sekelli`=old healed sequelae, `Gizli`=latent infection,
`Hastalıklı`=diseased (non-specific), `Tüberküloz`=active TB. Its API response returns these
**ASCII-folded** (`"Tuberkuloz"`, not `"Tüberküloz"`) — `inference.py`'s `_normalize_class_name()`
accent-strips both sides before matching rather than hardcoding one spelling.
Offline HF fallback: binary ViT classifier (`Normal`/`Tuberculosis`), TorchScript-exported, loaded
via `torch.jit.load`. Preprocessing (ported exactly from the repo's `handler.py`): grayscale →
CLAHE → Gaussian blur → resize to 224×224 → back to RGB → per-image z-score normalization.
Apache-2.0.

### Maternal Health — Model `hash-maternal-health/1` (primary) / `shr3m/fetal-brain-plane-cnn` (fallback)
Grounded from the project's own COCO export — a single-class detector (`"abnormal"`; a second
same-named category id in the export is an unused Roboflow placeholder with zero real annotations).
Same semantics as Mammography: any detection is a genuine flagged finding, no detection = normal.
Offline HF fallback: classifies fetal-**brain** ultrasound images into one of 4 standard planes —
narrower than full obstetric triage, but the closest public match found. CC BY 4.0, research/
education use only per its model card.

### Locally-trained classifier — `Models/<Modality>/local_model.pt`
Produced by `train_local_model.py`, not downloaded from anywhere — see the section above. TB:
82.5% (80 held-out samples: 40 positive + 40 negative). Mammography: 98.7% (78 held-out samples: 40
positive + 38 negative). Maternal Health: not trained (dataset has zero negative images — see the
offline-heuristic section above; same root cause).

### Considered and rejected: `Astaxanthin/KEEP` (suggested cancer model)
KEEP is a vision-language foundation model for zero-shot cancer diagnosis on **histopathology**
(H&E-stained biopsy slide) images — a different imaging modality from digital **mammography**
(radiographic breast X-ray). Left out rather than force-fit onto the wrong image type.

## Out-of-domain detection — "does this even look like the right kind of scan at all"

Checked by `inference.py`'s `check_image_domain()` before any real triage tier runs (and before any
Roboflow API call, so an obviously-wrong upload doesn't cost a network round-trip) — if flagged, the
UI shows "Wrong Image Type" instead of a triage verdict, with no bounding box/localization claim
(see `draw_bbox()` in `GUI.py`). Fully offline: `offline_cv.py`'s `domain_score()` reuses the same
templates as the triage heuristic — normalized cross-correlation between the uploaded image and the
modality's own generic reference image (the average of whichever of the positive/negative templates
exist), independent of best-orientation search.

**Measured, not assumed** (`calibrate_domain()` in `offline_cv.py`, run via `python offline_cv.py`):
each modality's own held-out test-split images vs. the OTHER two modalities' held-out test-split
images, used as a **deliberately hard** "wrong kind of scan" stand-in (a real medical grayscale
image, not an easy case like a random color photo):

| Modality | Threshold | Real scans correctly pass | Wrong-modality images correctly caught |
|---|---|---|---|
| Tuberculosis | 0.47 | 95% | 97.5% |
| Maternal Health | 0.37 | 80% | 82.5% |
| Mammography | 0.03 | 90% | 37.5% |

**Mammography's catch rate is deliberately weak** — its scans vary far more in crop/zoom/laterality
than TB's standardized front-on chest X-rays, so this pixel-correlation method can't separate them
as cleanly; the threshold was picked to protect real mammograms (90% pass) rather than to maximize
catch rate. Tested against a genuinely unrelated image (random noise or a solid color, not another
medical scan) all three modalities score near 0 — well below every threshold — so a real-world
"wrong image" upload (a photo, a document, a screenshot) should be caught far more reliably than
this hard cross-modality proxy suggests; that easier case just isn't independently measured here.
