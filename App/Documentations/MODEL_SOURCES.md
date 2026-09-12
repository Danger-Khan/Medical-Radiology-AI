# Model Sources — Pink Edge AI

Honesty convention carried over from the original project's own `MODEL_DOCUMENTATION.md`: every
modality below states exactly what is running behind it, including where it gets things wrong.
Nothing here should be read as a clinically validated product.

Each modality tries multiple methods in order, per `predict_tb()` / `predict_maternal()` /
`predict_mammography()` in `inference.py` — the order isn't fixed to "online first": it's set by
**measured accuracy against real held-out ground truth**, not by assumption.

| Modality | Try order | Measured accuracy |
|---|---|---|
| Mammography | Roboflow Workflow `breastcancer-yolov8-78tni` → offline pixel-diff heuristic → local weights (rare) → Simulated | Roboflow: ground-truth sample correctly flagged (90.9%). Offline heuristic: **96%** (50 held-out samples). |
| Tuberculosis | **Offline pixel-diff heuristic** → Roboflow model `tuberculosis-tp2pv/1` → offline HF ViT | Offline heuristic: **74%**. Offline HF ViT: 62%. Roboflow: 0% on healthy samples — **known issue, see below**. |
| Maternal Health | Roboflow model `hash-maternal-health/1` → offline pixel-diff heuristic (currently always skips — no negative data, see below) → offline HF CNN | Roboflow: ground-truth sample correctly flagged (88.3%). |

The Roboflow calls need a key (`roboflow_key.txt` / `ROBOFLOW_API_KEY` / Streamlit `st.secrets`) and
internet; the offline pixel-diff heuristic (`offline_cv.py`) and the Hugging Face models need
neither, ever, once their weights/datasets are on disk.

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
[mammography]  positive 25/25 | negative 23/25 | overall 48/50 (96%)
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
integration bug. **This is why TB's try-order puts the offline heuristic first** — it's not "prefer
offline on principle," it's that the offline heuristic (74%) measurably beats both the Roboflow
model (0% on healthy) and the offline HF ViT (62%) on the same held-out data. Revisit this ordering
if the Roboflow model is retrained with better class balance.

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

### Considered and rejected: `Astaxanthin/KEEP` (suggested cancer model)
KEEP is a vision-language foundation model for zero-shot cancer diagnosis on **histopathology**
(H&E-stained biopsy slide) images — a different imaging modality from digital **mammography**
(radiographic breast X-ray). Left out rather than force-fit onto the wrong image type.
