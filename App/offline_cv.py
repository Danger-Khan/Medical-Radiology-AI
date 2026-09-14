#!/usr/bin/env python3
"""
Medical Radiology AI — offline, non-neural triage fallback (classical pixel-difference comparison).
================================================================================================
No trained model, no internet, no API key, no huggingface_hub/torch/ultralytics dependency at
all: builds a "typical positive" and "typical negative" reference template per modality by
averaging images from the project's own local labeled datasets (Models/<Modality>/Data
Set/*.coco — the same ones inference.py's Roboflow ground-truth checks use), then classifies a
new image by:
  1. grayscale + resize to a canonical size + histogram-equalize (normalizes exposure/contrast)
  2. auto-orientation: try identity vs. horizontal-flip, keep whichever correlates better with
     a generic (positive+negative averaged) reference — handles left/right laterality without
     full image registration
  3. pixel-wise absolute difference against the positive template AND the negative template
  4. a per-pixel "leans positive" map (diff-from-negative minus diff-from-positive); its largest
     connected region above threshold becomes the highlighted bounding box, and its mean/coverage
     become the confidence score

This is deliberately simple and fully explainable — a nearest-mean-template heuristic, not a
substitute for the trained models. It exists as a dependency-light, always-available offline
tier: see CALIBRATION.md-equivalent notes below and Documentations/MODEL_SOURCES.md for measured
accuracy against held-out samples from the same datasets — be honest with yourself about what
those numbers mean before trusting this over the real models.

Besides the COCO-annotated datasets, each modality also has Models/<Modality>/positive/,
negative/, and validate/ folders (see their own README.md) for manually dropping in extra images
with no annotation step required — positive/negative feed straight into template-building
alongside the labeled dataset (cache auto-invalidates when those folders change), validate/ is a
no-ground-truth spot-check folder reported by `python offline_cv.py`.
"""
import json
import math
import os
import random
import shutil

import numpy as np
from PIL import Image

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")
CANON_SIZE = 256
MAX_TEMPLATE_SAMPLES = 150
CONFIDENCE_SCALE = 6.0  # picked empirically — see calibrate() / MODEL_SOURCES.md

# Dataset location + category grounding per modality — reuses the exact taxonomies grounded in
# inference.py (see Documentations/MODEL_SOURCES.md) rather than re-guessing them.
_DATASETS = {
    "tb": {
        "models_subdir": "TB", "dataset_dir_name": "tuberculosis.coco",
        "positive_categories": ["Tüberküloz", "Hastalıklı"],
        "negative_categories": ["Sağlıklı", "Sekelli"],
    },
    "mammography": {
        "models_subdir": "Mammography", "dataset_dir_name": "BreastCancer-YOLOv8.coco",
        "positive_categories": ["cancer"],
        "negative_categories": ["normal"],
    },
    # Bone is the one modality where positive/negative do NOT mean disease/healthy -- the source
    # dataset (yakin/bone-fracture-tn84w, CC BY 4.0) has zero normal/healthy bone X-rays annotated
    # (every one of its 2147 images has at least one fracture-type box), so there's no "negative"
    # class to build an OK reference from. Instead these slots are reused for a genuine SUB-TYPE
    # distinction between two real, well-populated classes: "positive" = Dislocation ("Shift"),
    # "negative" = every other specific fracture-type category ("Crack"). See
    # Documentations/MODEL_SOURCES.md for why, and inference.py's predict_bone() for how presence-of
    # -fracture-at-all (OK vs. not) is determined separately by the real-time detector tiers.
    "bone": {
        "models_subdir": "Bone", "dataset_dir_name": "Bone-Fracture.coco",
        "positive_categories": ["Dislocation"],
        "negative_categories": ["Avulsion", "Comminuted", "Fracture", "Greenstick", "Hairline",
                                 "Impacted", "Longitudinal", "Oblique", "Pathological", "Spiral"],
        # Excluded on purpose: "Bone fracture detection - v1 2023-03-05 5-51pm" and lowercase
        # "fracture" -- generic/duplicate boxes carried over from the dataset's earlier single-class
        # annotation pass (same "unused Roboflow placeholder" pattern noted for Maternal's COCO
        # export); they don't distinguish Shift from Crack, so including them would only add noise.
    },
}

_TEMPLATE_CACHE = {}  # modality -> (positive_template, negative_template) | (None, None)
_MANUAL_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _dataset_dir(modality):
    cfg = _DATASETS[modality]
    return os.path.join(MODELS_DIR, cfg["models_subdir"], "Data Set", cfg["dataset_dir_name"])


def _modality_dir(modality):
    return os.path.join(MODELS_DIR, _DATASETS[modality]["models_subdir"])


def _manual_image_paths(modality, sub):
    """Images a user drops directly into Models/<Modality>/positive|negative|validate/ — no COCO
    annotation needed, picked up automatically alongside the labeled dataset. `sub` is "positive",
    "negative", or "validate"."""
    d = os.path.join(_modality_dir(modality), sub)
    if not os.path.isdir(d):
        return []
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(_MANUAL_IMAGE_EXTS))


def _manual_signature(modality):
    """Cheap on-disk-cache invalidation: (filename, mtime) pairs for every manually-added
    positive/negative image, so dropping a new file into those folders triggers a template rebuild
    on the next call instead of silently using a stale cached template forever."""
    sig = []
    for sub in ("positive", "negative"):
        for p in _manual_image_paths(modality, sub):
            try:
                sig.append([os.path.relpath(p, MODELS_DIR), os.path.getmtime(p)])
            except OSError:
                continue
    return sorted(sig)


def _load_coco_splits(dataset_dir):
    for split in ("train", "valid", "test"):
        split_dir = os.path.join(dataset_dir, split)
        ann_path = os.path.join(split_dir, "_annotations.coco.json")
        if os.path.isfile(ann_path):
            with open(ann_path, "r", encoding="utf-8") as fh:
                yield split, split_dir, json.load(fh)


def _collect_image_paths(dataset_dir, positive_categories, negative_categories, exclude_split=None):
    """Returns (positive_paths, negative_paths). `exclude_split` ('test', etc.) lets calibrate()
    hold a split out of template-building so it can validate against unseen images."""
    positive, negative = set(), set()
    for split, split_dir, coco in _load_coco_splits(dataset_dir):
        if split == exclude_split:
            continue
        id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
        cat_name_by_id = {c["id"]: c["name"] for c in coco["categories"]}
        pos_ids = {cid for cid, name in cat_name_by_id.items() if name in positive_categories}
        neg_ids = {cid for cid, name in cat_name_by_id.items() if name in negative_categories}
        annotated_ids = set()
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            annotated_ids.add(img_id)
            file_name = id_to_file.get(img_id)
            if file_name is None:
                continue
            path = os.path.join(split_dir, file_name)
            if ann["category_id"] in pos_ids:
                positive.add(path)
            elif ann["category_id"] in neg_ids:
                negative.add(path)
        if not negative_categories:
            for im in coco["images"]:
                if im["id"] not in annotated_ids:
                    negative.add(os.path.join(split_dir, im["file_name"]))
    return sorted(positive), sorted(negative)


def _preprocess_canonical(pil_image):
    import cv2

    gray = np.asarray(pil_image.convert("L").resize((CANON_SIZE, CANON_SIZE)), dtype=np.uint8)
    return cv2.equalizeHist(gray)


def _build_template(image_paths, seed):
    if not image_paths:
        return None
    rng = random.Random(seed)
    sample = image_paths if len(image_paths) <= MAX_TEMPLATE_SAMPLES else rng.sample(image_paths, MAX_TEMPLATE_SAMPLES)
    acc, n = np.zeros((CANON_SIZE, CANON_SIZE), dtype=np.float64), 0
    for p in sample:
        try:
            acc += _preprocess_canonical(Image.open(p)).astype(np.float64)
            n += 1
        except Exception:
            continue
    return (acc / n).astype(np.uint8) if n else None


def _get_templates(modality, exclude_split=None, use_cache=True):
    """`exclude_split` bypasses the on-disk cache (used only by calibrate() for a proper
    train/test split — normal predict calls always use the cached, all-data templates).

    Positive/negative images placed directly in Models/<Modality>/positive|negative/ (no COCO
    annotation needed — see that folder's own README.md) are folded in alongside the labeled
    dataset every time templates are (re)built, and their mtimes are checked against the cache on
    every call so dropping in a new image triggers a rebuild rather than silently going unused."""
    cfg = _DATASETS.get(modality)
    if cfg is None:
        return None, None
    manual_sig = _manual_signature(modality)

    if use_cache and modality in _TEMPLATE_CACHE:
        cached_pos, cached_neg, cached_sig = _TEMPLATE_CACHE[modality]
        if cached_sig == manual_sig:
            return cached_pos, cached_neg

    tpl_dir = os.path.join(MODELS_DIR, cfg["models_subdir"], "templates")
    pos_path, neg_path = os.path.join(tpl_dir, "positive.png"), os.path.join(tpl_dir, "negative.png")
    meta_path = os.path.join(tpl_dir, "metadata.json")

    if use_cache and os.path.isfile(pos_path) and os.path.isfile(neg_path):
        cached_meta_sig = None
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as fh:
                    cached_meta_sig = json.load(fh).get("manual_signature")
            except (OSError, json.JSONDecodeError):
                pass
        if cached_meta_sig == manual_sig:
            result = (np.asarray(Image.open(pos_path).convert("L")), np.asarray(Image.open(neg_path).convert("L")))
            _TEMPLATE_CACHE[modality] = (result[0], result[1], manual_sig)
            return result
        # Manual folders changed since this cache was written -- fall through and rebuild.

    dataset_dir = _dataset_dir(modality)
    pos_paths, neg_paths = set(), set()
    if os.path.isdir(dataset_dir):
        coco_pos, coco_neg = _collect_image_paths(
            dataset_dir, cfg["positive_categories"], cfg["negative_categories"], exclude_split=exclude_split)
        pos_paths.update(coco_pos)
        neg_paths.update(coco_neg)
    pos_paths.update(_manual_image_paths(modality, "positive"))
    neg_paths.update(_manual_image_paths(modality, "negative"))
    pos_paths, neg_paths = sorted(pos_paths), sorted(neg_paths)

    pos_tpl = _build_template(pos_paths, seed=1) if pos_paths else None
    neg_tpl = _build_template(neg_paths, seed=2) if neg_paths else None

    if use_cache and pos_tpl is not None and neg_tpl is not None:
        os.makedirs(tpl_dir, exist_ok=True)
        Image.fromarray(pos_tpl).save(pos_path)
        Image.fromarray(neg_tpl).save(neg_path)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({
                "positive_samples_available": len(pos_paths), "negative_samples_available": len(neg_paths),
                "positive_samples_used": min(len(pos_paths), MAX_TEMPLATE_SAMPLES),
                "negative_samples_used": min(len(neg_paths), MAX_TEMPLATE_SAMPLES),
                "manual_signature": manual_sig,
            }, fh, indent=2)

    if use_cache:
        _TEMPLATE_CACHE[modality] = (pos_tpl, neg_tpl, manual_sig)
    return pos_tpl, neg_tpl


def _best_orientation(gray, reference):
    """Returns (best_image, orientation_name, correlation_score) — the score (normalized
    cross-correlation, roughly -1..1) also doubles as a cheap "does this even look like the right
    kind of scan at all" signal, independent of orientation — see domain_score()/
    is_out_of_domain() below."""
    ref = reference.astype(np.float64)
    ref_c = ref - ref.mean()
    ref_norm = np.linalg.norm(ref_c) or 1.0
    best_img, best_name, best_score = gray, "identity", -np.inf
    for name, cand in (("identity", gray), ("hflip", np.fliplr(gray))):
        c = cand.astype(np.float64)
        c_c = c - c.mean()
        score = float(np.sum(ref_c * c_c) / (ref_norm * (np.linalg.norm(c_c) or 1.0)))
        if score > best_score:
            best_img, best_name, best_score = cand, name, score
    return best_img, best_name, best_score


def _analyze(modality, pil_image, positive_tpl, negative_tpl):
    """Core comparison. Returns (change_score, change_pct, bbox_normalized_xywh_or_None, orientation)."""
    import cv2

    gray = _preprocess_canonical(pil_image)
    generic_ref = ((positive_tpl.astype(np.float64) + negative_tpl.astype(np.float64)) / 2).astype(np.uint8)
    gray, orientation, _ = _best_orientation(gray, generic_ref)

    diff_pos = np.abs(gray.astype(np.float64) - positive_tpl.astype(np.float64))
    diff_neg = np.abs(gray.astype(np.float64) - negative_tpl.astype(np.float64))
    lean_map = diff_neg - diff_pos  # >0 pixel looks more like the positive/abnormal template

    change_score = float(lean_map.mean())
    change_pct = float((lean_map > 0).mean() * 100)

    threshold = max(float(lean_map.std()) * 0.75, 3.0)
    mask = (lean_map > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bbox = None
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) >= 25:
            x, y, w, h = cv2.boundingRect(largest)
            bbox = (x / CANON_SIZE, y / CANON_SIZE, w / CANON_SIZE, h / CANON_SIZE)
    return change_score, change_pct, bbox, orientation


def _confidence(change_score):
    return 100.0 / (1.0 + math.exp(-change_score / CONFIDENCE_SCALE))


# ============================================================
# OUT-OF-DOMAIN DETECTION — "does this even look like the right kind of scan at all", checked
# BEFORE running any real triage tier (see inference.py's predict_tb/mammography/bone). Fully
# offline, reuses the same templates/correlation machinery as the triage heuristic above rather
# than adding a new dependency: a real chest X-ray correlates reasonably well with the TB generic
# reference even when it disagrees on diagnosis; a photo of a cat or a document does not correlate
# well with ANY of the three modalities' references. See DOMAIN_THRESHOLDS' comment for how the
# per-modality cutoffs below were picked, and calibrate_domain() to re-measure them yourself.
# ============================================================
DOMAIN_THRESHOLDS = {
    # Picked via calibrate_domain() (run `python offline_cv.py` to reproduce): each modality's own
    # held-out test-split images vs. the OTHER two modalities' held-out test-split images used as
    # a deliberately HARD "wrong kind of scan" stand-in (another real medical grayscale image, not
    # an easy case like a random color photo) — see Documentations/MODEL_SOURCES.md for the full
    # measured score distributions and why mammography's threshold is biased toward never blocking
    # a real mammogram rather than maximizing catch rate (its scans vary too much in crop/zoom for
    # this pixel-correlation method to separate as cleanly as TB's more standardized X-rays do).
    "tb": 0.47,          # measured: 95% real TB scans pass, 97% wrong-modality images caught
    "mammography": 0.03,  # measured: 90% real mammograms pass, only ~33% wrong-modality images
                           # caught at this lenient setting — see MODEL_SOURCES.md before relying on it
    # Measured via calibrate_domain("bone"): in-domain scores (mean 0.28) and out-of-domain scores
    # (mean 0.30) almost entirely OVERLAP for this modality -- bone X-rays vary too much in framing
    # (wrist vs. skull vs. shoulder) for one generic reference template to separate them from other
    # radiograph types this way, worse than Mammography's already-weak case. A negative threshold
    # would pass 100% of real scans but ALSO passes genuinely unrelated images (random noise scores
    # ~0, and the worst real bone sample measured -0.11 -- noise scores HIGHER than that real image,
    # so no threshold can both pass every real scan and reject noise here). 0.0 is the honest
    # compromise actually used: catches noise/blank images reliably while still passing ~93%
    # (14/15 measured) of real bone X-rays -- see MODEL_SOURCES.md before relying on it.
    "bone": 0.0,
}


def _generic_reference(modality):
    """Whichever of the positive/negative templates exist, averaged — a "what does this modality's
    scan look like at all" reference for domain checking, independent of _get_templates()'s
    requirement that BOTH classes exist (out-of-domain detection only needs "some reference",
    not a full positive/negative split — useful for a modality with only one class populated)."""
    pos, neg = _get_templates(modality)
    if pos is not None and neg is not None:
        return ((pos.astype(np.float64) + neg.astype(np.float64)) / 2).astype(np.uint8)
    return pos if pos is not None else neg


def domain_score(modality, pil_image):
    """Best-orientation normalized cross-correlation against modality's generic reference image
    (roughly -1..1, higher = looks more like this modality's scans). Returns None if no reference
    is available at all (no dataset and nothing manually added for this modality)."""
    reference = _generic_reference(modality)
    if reference is None:
        return None
    gray = _preprocess_canonical(pil_image)
    _, _, score = _best_orientation(gray, reference)
    return score


def is_out_of_domain(modality, pil_image):
    """True if `pil_image` doesn't look like the right kind of scan for `modality` at all. False
    (never flags anything) if no reference exists yet to compare against — an unavailable check
    should never block triage, it should just not run."""
    score = domain_score(modality, pil_image)
    if score is None:
        return False, None
    return score < DOMAIN_THRESHOLDS.get(modality, 0.3), score


_MODALITY_VOCAB = {
    "tb": {
        "pos_verdict": "TB Positive", "neg_verdict": "TB Negative",
        "pos_bi_rads_by_tier": ["S1 - Minimal (unilateral, no cavity)", "S2 - Moderate (bilateral / cavity < 2 cm)",
                                 "S3 - Advanced (large cavity / miliary pattern)"],
        "neg_bi_rads": "S0 - No active disease", "acr_pos": "Upper Zone", "acr_neg": "Bilateral",
        "sms_pos": "TB:POS", "sms_neg": "TB:NEG",
    },
    "mammography": {
        "pos_verdict": "BI-RADS 5", "neg_verdict": "Normal",
        "pos_bi_rads_by_tier": ["BI-RADS 4A - Low suspicion (Biopsy recommended)", "BI-RADS 4B - Moderate suspicion",
                                 "BI-RADS 5 - Highly suggestive of malignancy"],
        "neg_bi_rads": "BI-RADS 1 - Negative", "acr_pos": "C - Heterogeneously dense",
        "acr_neg": "B - Scattered fibroglandular density", "sms_pos": "BI-RADS:5", "sms_neg": "BI-RADS:1",
    },
}


def _to_bone_result(confidence, is_shift, bbox):
    """Bone reuses the positive/negative offline-heuristic machinery for a Shift-vs-Crack SUB-TYPE
    call, not disease-vs-healthy like every other modality here -- both outcomes are real, abnormal
    findings (a fracture either way), so both branches are flagged danger/critical, unlike the
    negative=healthy/success branch below. This tier never returns "OK" on its own -- presence-of
    -fracture-at-all is decided by inference.py's real-time detector tiers before this sub-typer
    ever runs (see predict_bone())."""
    source = "Offline pixel-comparison heuristic vs. local Bone dataset (no model, no internet)"
    if is_shift:
        return {
            "bi_rads": "Shift - Dislocation / displaced fracture", "acr": "Dislocation",
            "verdict": "Bone Shift (Dislocation)",
            "sub": "Offline heuristic: image pattern leans toward the Dislocation reference set",
            "css": "danger", "loc": "See bounding box", "extra": "Shift (Dislocation)",
            "vicon": "⚠️", "confidence": confidence, "sms": "BONE:SHIFT", "is_critical": True,
            "source": source, "bbox": bbox,
        }
    return {
        "bi_rads": "Crack - Fracture detected", "acr": "Fracture (non-dislocation)",
        "verdict": "Bone Crack (Fracture)",
        "sub": "Offline heuristic: image pattern leans toward the non-dislocation fracture reference set",
        "css": "danger", "loc": "See bounding box", "extra": "Crack (Fracture)",
        "vicon": "⚠️", "confidence": confidence, "sms": "BONE:CRACK", "is_critical": True,
        "source": source, "bbox": bbox,
    }


def _to_result(modality, confidence, is_positive, bbox):
    if modality == "bone":
        return _to_bone_result(confidence, is_positive, bbox)
    v = _MODALITY_VOCAB[modality]
    source = f"Offline pixel-comparison heuristic vs. local {modality} dataset (no model, no internet)"
    if is_positive:
        tier = 2 if confidence >= 85 else (1 if confidence >= 65 else 0)
        return {
            "bi_rads": v["pos_bi_rads_by_tier"][tier], "acr": v["acr_pos"], "verdict": v["pos_verdict"],
            "sub": "Offline heuristic: image pattern leans toward the positive reference set",
            "css": "danger", "loc": "See bounding box", "extra": v["pos_bi_rads_by_tier"][tier],
            "vicon": "⚠️", "confidence": confidence, "sms": v["sms_pos"], "is_critical": True,
            "source": source, "bbox": bbox,
        }
    return {
        "bi_rads": v["neg_bi_rads"], "acr": v["acr_neg"], "verdict": v["neg_verdict"],
        "sub": "Offline heuristic: image pattern leans toward the negative/healthy reference set",
        "css": "success", "loc": "No notable deviation found", "extra": v["neg_bi_rads"],
        "vicon": "✅", "confidence": confidence, "sms": v["sms_neg"], "is_critical": False,
        "source": source, "bbox": bbox,
    }


def available(modality: str) -> bool:
    pos, neg = _get_templates(modality)
    return pos is not None and neg is not None


def predict(modality: str, pil_image: Image.Image):
    """modality: 'tb' | 'bone' | 'mammography'. Returns a result-dict (with an extra `bbox`
    field — normalized (x, y, w, h) fractions, or None) or None if no local dataset is present
    to build templates from."""
    positive_tpl, negative_tpl = _get_templates(modality)
    if positive_tpl is None or negative_tpl is None:
        return None
    change_score, change_pct, bbox, orientation = _analyze(modality, pil_image, positive_tpl, negative_tpl)
    confidence = _confidence(change_score)
    is_positive = confidence >= 50.0
    result = _to_result(modality, confidence if is_positive else 100.0 - confidence, is_positive, bbox)
    result["sub"] += f" ({change_pct:.1f}% of pixels leaning positive, orientation={orientation})"
    return result


# ============================================================
# Calibration / self-test — not used at runtime, run directly to (re)validate:
#   python offline_cv.py
# Builds templates from train+valid only, tests against the held-out `test` split, reports real
# accuracy so this method's confidence isn't just taken on faith.
# ============================================================
def calibrate(modality: str, max_per_class=25):
    cfg = _DATASETS[modality]
    dataset_dir = _dataset_dir(modality)
    pos_tpl, neg_tpl = _get_templates(modality, exclude_split="test", use_cache=False)
    if pos_tpl is None or neg_tpl is None:
        print(f"  [{modality}] no dataset found, skipping")
        return

    # Held-out test-split images only, by ground-truth category.
    test_dir = os.path.join(dataset_dir, "test")
    ann_path = os.path.join(test_dir, "_annotations.coco.json")
    if not os.path.isfile(ann_path):
        print(f"  [{modality}] no test split, skipping")
        return
    with open(ann_path, "r", encoding="utf-8") as fh:
        coco = json.load(fh)
    id_to_file = {im["id"]: im["file_name"] for im in coco["images"]}
    cat_name_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    pos_ids = {cid for cid, n in cat_name_by_id.items() if n in cfg["positive_categories"]}
    neg_ids = {cid for cid, n in cat_name_by_id.items() if n in cfg["negative_categories"]}
    annotated = set()
    pos_files, neg_files = [], []
    for ann in coco["annotations"]:
        annotated.add(ann["image_id"])
        fn = id_to_file.get(ann["image_id"])
        if fn is None:
            continue
        if ann["category_id"] in pos_ids:
            pos_files.append(fn)
        elif ann["category_id"] in neg_ids:
            neg_files.append(fn)
    if not cfg["negative_categories"]:
        neg_files = [im["file_name"] for im in coco["images"] if im["id"] not in annotated]

    rng = random.Random(0)
    pos_files = rng.sample(pos_files, min(max_per_class, len(pos_files)))
    neg_files = rng.sample(neg_files, min(max_per_class, len(neg_files)))

    def _run(files, expect_positive):
        correct = 0
        for fn in files:
            img = Image.open(os.path.join(test_dir, fn))
            score, _, _, _ = _analyze(modality, img, pos_tpl, neg_tpl)
            predicted_positive = _confidence(score) >= 50.0
            correct += predicted_positive == expect_positive
        return correct, len(files)

    pc, pn = _run(pos_files, True)
    nc, nn = _run(neg_files, False)
    total_c, total_n = pc + nc, pn + nn
    print(f"  [{modality}] positive: {pc}/{pn} correct | negative: {nc}/{nn} correct | "
          f"overall: {total_c}/{total_n} ({100 * total_c / total_n:.0f}%)")


def _test_split_images(modality, limit=15, seed=0):
    """Held-out test-split image paths for `modality`, regardless of category (used by
    calibrate_domain() as "definitely a real X" / "definitely NOT an X" stand-ins)."""
    cfg = _DATASETS.get(modality)
    if cfg is None:
        return []
    dataset_dir = _dataset_dir(modality)
    if not os.path.isdir(dataset_dir):
        return []
    pos, neg = _collect_image_paths(dataset_dir, cfg["positive_categories"], cfg["negative_categories"], exclude_split=None)
    test_marker = os.sep + "test" + os.sep
    all_test = sorted({p for p in (pos + neg) if test_marker in p})
    if len(all_test) > limit:
        all_test = random.Random(seed).sample(all_test, limit)
    return all_test


def calibrate_domain(modality, samples_per_class=15):
    """Measures how well domain_score() separates real in-domain images (this modality's own
    held-out test split) from out-of-domain images (the OTHER two modalities' held-out test
    splits, used as "definitely the wrong kind of scan" stand-ins — a chest X-ray fed to the
    Mammography check, etc.) — prints the score distributions so DOMAIN_THRESHOLDS above is a
    measured pick, not a guess. Run via `python offline_cv.py`."""
    in_domain_paths = _test_split_images(modality, limit=samples_per_class)
    if not in_domain_paths:
        print(f"  [{modality}] no test-split images available, skipping domain calibration")
        return
    out_domain_paths = []
    for other in _DATASETS:
        if other != modality:
            out_domain_paths += _test_split_images(other, limit=samples_per_class)
    if not out_domain_paths:
        print(f"  [{modality}] no other-modality images available for comparison, skipping")
        return

    in_scores = [domain_score(modality, Image.open(p)) for p in in_domain_paths]
    out_scores = [domain_score(modality, Image.open(p)) for p in out_domain_paths]
    threshold = DOMAIN_THRESHOLDS.get(modality, 0.3)
    in_pass = sum(s >= threshold for s in in_scores)
    out_caught = sum(s < threshold for s in out_scores)
    print(f"  [{modality}] in-domain (n={len(in_scores)}): min={min(in_scores):.2f} "
          f"mean={sum(in_scores) / len(in_scores):.2f} max={max(in_scores):.2f} -- "
          f"{in_pass}/{len(in_scores)} correctly pass threshold {threshold}")
    print(f"  [{modality}] out-of-domain (n={len(out_scores)}): min={min(out_scores):.2f} "
          f"mean={sum(out_scores) / len(out_scores):.2f} max={max(out_scores):.2f} -- "
          f"{out_caught}/{len(out_scores)} correctly caught below threshold {threshold}")


def gather_from_dataset(modality, target, count=15, seed=0):
    """Copies real images from the modality's own Data Set into Models/<Modality>/<target>/
    (target: "positive", "negative", or "validate") -- so those folders can hold actual files to
    look at/build on instead of staying empty, without requiring anyone to hunt down extra images
    by hand. "positive"/"negative" pull only from the dataset's train/valid splits (never `test`)
    so gathering samples can never quietly leak held-out data into the template and inflate
    calibrate()'s accuracy numbers; "validate" pulls specifically FROM the held-out `test` split
    instead (genuinely unseen-by-the-template images are exactly what a spot-check folder wants).
    Only ever adds files that aren't already present (by filename) -- never overwrites or
    duplicates anything a user has dropped in themselves. See __main__'s `--gather` flag."""
    cfg = _DATASETS.get(modality)
    if cfg is None:
        return []
    dataset_dir = _dataset_dir(modality)
    if not os.path.isdir(dataset_dir):
        return []

    if target in ("positive", "negative"):
        pos_paths, neg_paths = _collect_image_paths(
            dataset_dir, cfg["positive_categories"], cfg["negative_categories"], exclude_split="test")
        candidates = pos_paths if target == "positive" else neg_paths
    else:  # validate -- draw only from the held-out test split, no label needed either way
        pos_paths, neg_paths = _collect_image_paths(
            dataset_dir, cfg["positive_categories"], cfg["negative_categories"], exclude_split=None)
        train_valid_pos, train_valid_neg = _collect_image_paths(
            dataset_dir, cfg["positive_categories"], cfg["negative_categories"], exclude_split="test")
        test_only = (set(pos_paths) | set(neg_paths)) - set(train_valid_pos) - set(train_valid_neg)
        candidates = sorted(test_only)
    if not candidates:
        return []

    rng = random.Random(seed)
    sample = candidates if len(candidates) <= count else rng.sample(candidates, count)

    out_dir = os.path.join(_modality_dir(modality), target)
    os.makedirs(out_dir, exist_ok=True)
    copied = []
    for src in sample:
        dst = os.path.join(out_dir, os.path.basename(src))
        if os.path.exists(dst):
            continue
        try:
            shutil.copy2(src, dst)
            copied.append(dst)
        except OSError:
            continue
    return copied


def check_validate_folder(modality):
    """Runs predict() against every image dropped into Models/<Modality>/validate/ and prints the
    verdict for manual eyeballing. Unlike calibrate(), these images carry no ground-truth label
    (that's the point — it's for "I found some extra images, what does the heuristic say about
    them" spot-checks, not a formal accuracy measurement), so this reports predictions, not a score."""
    paths = _manual_image_paths(modality, "validate")
    if not paths:
        return
    print(f"  [{modality}] validate/ spot-check ({len(paths)} image(s), no ground truth — "
          f"predictions only):")
    for p in paths:
        try:
            result = predict(modality, Image.open(p))
        except Exception as exc:
            print(f"    {os.path.basename(p)}: ERROR ({exc})")
            continue
        if result is None:
            print(f"    {os.path.basename(p)}: no templates available")
        else:
            print(f"    {os.path.basename(p)}: {result['verdict']} ({result['confidence']:.1f}%)")


if __name__ == "__main__":
    import sys

    if "--gather" in sys.argv:
        count = 15
        if "--count" in sys.argv:
            idx = sys.argv.index("--count")
            if idx + 1 < len(sys.argv) and sys.argv[idx + 1].isdigit():
                count = int(sys.argv[idx + 1])
        print(f"Gathering up to {count} real sample(s) from each modality's own Data Set, into "
              f"positive/negative/validate/ folders that are still empty (leaves anything you've "
              f"already added there alone):")
        for m in _DATASETS:
            for target in ("positive", "negative", "validate"):
                if _manual_image_paths(m, target):
                    print(f"  [{m}/{target}] already has files — leaving as-is")
                    continue
                copied = gather_from_dataset(m, target, count=count)
                print(f"  [{m}/{target}] gathered {len(copied)} sample(s) from Data Set"
                      if copied else f"  [{m}/{target}] nothing to gather (no Data Set found)")
        print("\nRe-run without --gather to see updated calibration/spot-check numbers.")
        raise SystemExit(0)

    print("Offline CV heuristic — calibration against held-out test-split data (not used at runtime):")
    print("(tip: `python offline_cv.py --gather` seeds empty positive/negative/validate/ folders "
          "with real sample images pulled from each modality's own Data Set)")
    for m in _DATASETS:
        calibrate(m)
    print("\nOut-of-domain detection calibration (does DOMAIN_THRESHOLDS actually separate "
          "real scans from the other two modalities' images?):")
    for m in _DATASETS:
        calibrate_domain(m)
    print("\nManual validate/ folder spot-checks (drop extra images into Models/<Modality>/validate/):")
    for m in _DATASETS:
        check_validate_folder(m)
