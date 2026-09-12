# Model Sources — Pink Edge AI

Honesty convention carried over from the original project's own `MODEL_DOCUMENTATION.md`: every
modality below states exactly what is running behind it, including where it gets things wrong.
Nothing here should be read as a clinically validated product.

Each modality now tries **your own trained model on Roboflow** first (workspace
`imaad-ullah-khan-yameen`), falling back to an offline Hugging Face model (TB, Maternal) or the
simulated scenario picker (Mammography only, if no local weights either) when Roboflow is
unreachable — no key configured, offline, or the call fails. See `predict_tb()` /
`predict_maternal()` / `predict_mammography()` in `inference.py`.

| Modality | Primary (Roboflow, hosted) | Fallback (offline) |
|---|---|---|
| Mammography | Workflow `breastcancer-yolov8-78tni` | Simulated (no local weights present) |
| Tuberculosis | Model `tuberculosis-tp2pv/1` — **⚠️ known accuracy issue, see below** | [sukhmani1303/tuberculosis-vit-model](https://huggingface.co/sukhmani1303/tuberculosis-vit-model) (Hugging Face) |
| Maternal Health | Model `hash-maternal-health/1` | [shr3m/fetal-brain-plane-cnn](https://huggingface.co/shr3m/fetal-brain-plane-cnn) (Hugging Face) |

All three Roboflow calls require a key in `roboflow_key.txt` / `ROBOFLOW_API_KEY` / Streamlit
`st.secrets` — see `README.md`. **This makes those three modalities cloud-dependent at inference
time when the key is present**, unlike the fully-offline Hugging Face fallbacks — a deliberate
tradeoff the user chose (real, purpose-trained predictions now; offline export is a follow-up).

## ⚠️ Known issue: `tuberculosis-tp2pv/1` has a high false-positive rate

Validated against the project's own ground-truth COCO annotations
(`Models/TB/Data Set/tuberculosis.coco/test/_annotations.coco.json`), not just "does it run":
correctly flagged a `Tüberküloz` (active TB)-annotated sample as positive (91.2%), but called
**all 3** `Sağlıklı` (healthy)-annotated samples tested TB Positive too, at **98–99% confidence** —
high-confidence wrong, not a borderline call a confidence-threshold tweak would fix. There is only
one version of this project on Roboflow (checked via the API — no newer version to switch to).
This looks like a real training-quality issue in the deployed model (likely aggravated by class
imbalance: 432 `Tüberküloz` vs. 159 `Sağlıklı` annotations in just the test split), not an
integration bug — `Validation/validate.py`'s TB ground-truth check is left failing on purpose to
keep surfacing this rather than being quietly loosened until it passes. Suggested next steps, in
order of effort: retrain on Roboflow with better class balance, try excluding low-value classes
(`Gizli`/`Sekelli` are sparse), or treat the offline Hugging Face ViT as the more trustworthy path
for TB until the Roboflow model improves — flip the try-order in `predict_tb()` if so.

## Details

### Mammography — Workflow `breastcancer-yolov8-78tni`
Grounded via a real call: detections come back as
`{"x", "y", "width", "height", "confidence", "class", "class_id"}` inside
`result[0]["predictions"]["predictions"]`; class `"cancer"` seen on a real positive sample,
empty list on real negatives. Verified against ground truth (a COCO-annotated cancer sample →
correctly flagged, 90.9%). No local trained-weight export was found for the fallback path
(`b-davmu/breastcancer-yolov8`, a *different*, public Roboflow Universe project used only as a
placeholder before the user's own workspace/model was available) — mammography is SIMULATED if no
Roboflow key is present.

### Tuberculosis — Model `tuberculosis-tp2pv/1` (primary) / `sukhmani1303/tuberculosis-vit-model` (fallback)
**Primary**: real class taxonomy grounded from the project's own COCO export (Turkish labels):
`Sağlıklı`=Healthy, `Sekelli`=old healed sequelae, `Gizli`=latent infection, `Hastalıklı`=diseased
(non-specific), `Tüberküloz`=active TB. The deployed model's API response returns these
**ASCII-folded** (`"Tuberkuloz"`, not `"Tüberküloz"`) — `inference.py`'s `_normalize_class_name()`
accent-strips both sides before matching rather than hardcoding one spelling. See the accuracy
caveat above.
**Fallback**: binary ViT classifier (`Normal`/`Tuberculosis`), TorchScript-exported, loaded via
`torch.jit.load` — no custom architecture code needed. Preprocessing (ported exactly from the repo's
`handler.py`): grayscale → CLAHE (clipLimit 2.0, tile 8×8) → Gaussian blur (5×5) → resize to 224×224
→ back to RGB → per-image z-score normalization. Apache-2.0.

### Maternal Health — Model `hash-maternal-health/1` (primary) / `shr3m/fetal-brain-plane-cnn` (fallback)
**Primary**: grounded from the project's own COCO export
(`Models/Maternal/Data Set/HASH Maternal Health.coco/*/_annotations.coco.json`) — a single-class
detector (`"abnormal"`; a second same-named category id in the export is an unused Roboflow
placeholder with zero real annotations). Same semantics as Mammography: any detection is a genuine
flagged finding (BI-RADS-style "4B - Moderate suspicion", critical), no detection = normal.
Verified against a ground-truth `abnormal`-annotated sample (88.3%, correctly flagged).
**Fallback**: classifies fetal-**brain** ultrasound images into one of 4 standard planes
(Trans-thalamic, Trans-cerebellum, Trans-ventricular, Other) — narrower than full obstetric
"maternal health" triage, but the closest public match found on Hugging Face. CC BY 4.0, research/
education use only per its model card.

### Considered and rejected: `Astaxanthin/KEEP` (suggested cancer model)
KEEP is a vision-language foundation model for zero-shot cancer diagnosis on **histopathology**
(H&E-stained biopsy slide) images — a different imaging modality from digital **mammography**
(radiographic breast X-ray). Left out rather than force-fit onto the wrong image type.
