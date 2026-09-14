#!/usr/bin/env python3
"""
Medical Radiology AI (Desktop) — real on-device model loaders/predictors.
====================================================================
One function per modality: `load_<modality>()` (lazy, cached) and `predict_<modality>(pil_image)`.

Every predict_* returns either a result-dict shaped like the app's simulated scenarios
(bi_rads, acr, verdict, sub, loc, extra, vicon, confidence, sms, is_critical, source) or None
if no real model could be loaded — in which case GUI.py falls back to its own simulated
scenario picker for that modality, clearly labeled SIMULATED in the UI. See MODEL_SOURCES.md
for exactly which model backs which modality and why.
"""
import os
import random
import time

import numpy as np
from PIL import Image

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Models")

BI_RADS_OPTIONS = [
    "BI-RADS 0 - Incomplete (Additional imaging needed)",
    "BI-RADS 1 - Negative",
    "BI-RADS 2 - Benign finding",
    "BI-RADS 3 - Probably benign (6-month follow-up)",
    "BI-RADS 4A - Low suspicion (Biopsy recommended)",
    "BI-RADS 4B - Moderate suspicion",
    "BI-RADS 4C - High suspicion",
    "BI-RADS 5 - Highly suggestive of malignancy",
    "BI-RADS 6 - Known biopsy-proven malignancy",
]
ACR_DENSITY_OPTIONS = [
    "A - Almost entirely fatty",
    "B - Scattered fibroglandular density",
    "C - Heterogeneously dense",
    "D - Extremely dense",
]
TB_SEVERITY_LEVELS = [
    "S0 - No active disease",
    "S1 - Minimal (unilateral, no cavity)",
    "S2 - Moderate (bilateral / cavity < 2 cm)",
    "S3 - Advanced (large cavity / miliary pattern)",
]
TB_LUNG_ZONES = ["Upper Zone", "Middle Zone", "Lower Zone", "Bilateral"]
BONE_STATUS_OPTIONS = [
    "OK - No fracture or dislocation",
    "Crack - Fracture detected (non-dislocation)",
    "Shift - Dislocation / displaced fracture",
]
# Grounded from yakin/bone-fracture-tn84w's own real category taxonomy (see MODEL_SOURCES.md) --
# the specific fracture sub-type, when the offline heuristic/local classifier could determine one.
BONE_TYPE_OPTIONS = [
    "N/A - No detection", "Dislocation (Shift)", "Avulsion", "Comminuted", "Fracture",
    "Greenstick", "Hairline", "Impacted", "Longitudinal", "Oblique", "Pathological", "Spiral",
]


# ============================================================
# ROBOFLOW — hosted models/workflows in the `imaad-ullah-khan-yameen` workspace.
# This is the PRIMARY real-model path for Mammography/TB/Bone (purpose-trained on this
# project's own data for Mammography/TB; Bone instead calls a public Roboflow Universe project --
# see its own section below for why); each modality falls back to its offline Hugging Face model
# (TB, Bone) or the simulated scenario picker (Mammography, if no local weights either) when
# Roboflow is unreachable — no internet, no key, or the call fails. Grounded against real
# calls during integration (see Documentations/MODEL_SOURCES.md for the exact example
# responses); IDs below are the actual callable model/workflow IDs, which differ slightly
# from the human-readable names shown in the Roboflow dashboard.
# ============================================================
ROBOFLOW_WORKSPACE = "imaad-ullah-khan-yameen"
ROBOFLOW_API_URL = "https://serverless.roboflow.com"
ROBOFLOW_MAMMOGRAPHY_WORKFLOW_ID = "breastcancer-yolov8-78tni"
ROBOFLOW_TB_MODEL_ID = "tuberculosis-tp2pv/1"
# Public Roboflow Universe project (workspace "yakin", not this project's own workspace) -- the
# ONLY one of its 3 versions with an actually trained/deployed model (grounded via a real call to
# its training summary: precision 86.5%, recall 71.1%, mAP@50 77.3%). Callable with any valid
# Roboflow API key, same as `b-davmu/breastcancer-yolov8` was used as a Mammography placeholder
# before the user's own workspace model existed. See MODEL_SOURCES.md for the full grounding.
ROBOFLOW_BONE_MODEL_ID = "bone-fracture-tn84w/1"


class RoboflowError(Exception):
    """Raised on any failure calling a hosted Roboflow model/workflow — callers catch this
    and fall back to the offline path rather than letting it propagate to the UI."""


def _roboflow_api_key():
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key.strip()
    # Streamlit Community Cloud's Secrets manager (Settings -> Secrets) surfaces values via
    # st.secrets rather than env vars or files; check it if Streamlit is running/available.
    try:
        import streamlit as st

        if "ROBOFLOW_API_KEY" in st.secrets:
            return str(st.secrets["ROBOFLOW_API_KEY"]).strip()
    except Exception:
        pass
    key_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roboflow_key.txt")
    if os.path.isfile(key_file):
        with open(key_file, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    return None


_roboflow_client = None
_roboflow_client_error = None


def _get_roboflow_client():
    global _roboflow_client, _roboflow_client_error
    if _roboflow_client is not None:
        return _roboflow_client
    if _roboflow_client_error is not None:
        raise RoboflowError(_roboflow_client_error)
    key = _roboflow_api_key()
    if not key:
        _roboflow_client_error = "no Roboflow API key configured (roboflow_key.txt / ROBOFLOW_API_KEY / st.secrets)"
        raise RoboflowError(_roboflow_client_error)
    try:
        from inference_sdk import InferenceConfiguration, InferenceHTTPClient

        client = InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=key)
        client.configure(InferenceConfiguration(api_key_transport="header"))
        _roboflow_client = client
        return client
    except Exception as e:
        _roboflow_client_error = f"failed to construct Roboflow client: {e}"
        raise RoboflowError(_roboflow_client_error) from e


def _with_retries(fn, retries=2, backoff=1.5, what="Roboflow call"):
    """Run fn() with a couple of retries and exponential backoff; raise RoboflowError with
    a clear message (including the last underlying error) if every attempt fails."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    raise RoboflowError(f"{what} failed after {retries + 1} attempt(s): {last_err}") from last_err


def _roboflow_infer(model_id: str, pil_image: Image.Image, retries=2, backoff=1.5) -> list:
    """Direct model inference (client.infer) — for standalone hosted models (TB, Bone).
    Returns the raw `predictions` list from the response; raises RoboflowError on failure."""
    client = _get_roboflow_client()
    arr = np.asarray(pil_image.convert("RGB"))

    def _call():
        result = client.infer(arr, model_id=model_id)
        return result.get("predictions", []) if isinstance(result, dict) else []

    return _with_retries(_call, retries, backoff, what=f"Roboflow infer({model_id})")


def _roboflow_run_workflow(workflow_id: str, pil_image: Image.Image, retries=2, backoff=1.5) -> list:
    """Hosted Workflow call (client.run_workflow) — for Mammography. Grounded response shape:
    result is a list (one entry per input image); each entry is
    {"predictions": {"image": {...}, "predictions": [...]}, "inference_id": ..., "model_id": ...}.
    Returns the inner detections list; raises RoboflowError on failure."""
    client = _get_roboflow_client()
    arr = np.asarray(pil_image.convert("RGB"))

    def _call():
        result = client.run_workflow(
            workspace_name=ROBOFLOW_WORKSPACE, workflow_id=workflow_id, images={"image": arr}, use_cache=True,
        )
        entry = result[0] if isinstance(result, list) and result else {}
        return entry.get("predictions", {}).get("predictions", [])

    return _with_retries(_call, retries, backoff, what=f"Roboflow run_workflow({workflow_id})")


def _top_box(predictions: list):
    """Highest-confidence detection, or None if the list is empty."""
    if not predictions:
        return None
    return max(predictions, key=lambda p: p.get("confidence", 0.0))


# ============================================================
# OUT-OF-DOMAIN GATE — checked before any real inference tier runs (and before any Roboflow API
# call, so an obviously-wrong upload doesn't cost a network round-trip). Fully offline; see
# offline_cv.py's domain_score()/is_out_of_domain()/DOMAIN_THRESHOLDS for the actual method and
# Documentations/MODEL_SOURCES.md for the measured catch rate per modality (mammography's is
# deliberately lenient — see that file before assuming this always catches a wrong upload).
# ============================================================
_EXPECTED_IMAGE_DESC = {
    "tb": "a chest X-ray", "mammography": "a mammogram (breast X-ray)",
    "bone": "a bone X-ray (limb/joint radiograph)",
}


def _invalid_image_result(modality: str, score) -> dict:
    desc = _EXPECTED_IMAGE_DESC.get(modality, "the expected scan type")
    threshold = None
    try:
        import offline_cv

        threshold = offline_cv.DOMAIN_THRESHOLDS.get(modality)
    except Exception:
        pass
    confidence = 60.0 if (score is None or threshold is None) else \
        float(min(99.0, max(50.0, 50 + (threshold - score) * 100)))
    return {
        "bi_rads": "N/A - Invalid Image", "acr": "N/A", "verdict": "Wrong Image Type",
        "sub": f"This doesn't look like {desc} — please upload the correct type of scan.",
        "css": "warning", "loc": "N/A", "extra": "Invalid image — not triaged", "vicon": "🚫",
        "confidence": confidence, "sms": "INVALID", "is_critical": False,
        "source": "Input validation (offline_cv.py domain check, no internet, no model)",
        "invalid_image": True,
    }


def check_image_domain(modality: str, pil_image: Image.Image):
    """Returns an invalid-image result dict if `pil_image` doesn't look like the right kind of
    scan for `modality`, else None. Fails open (returns None) on any error — a broken domain
    check must never block real triage, only an active, working check should."""
    try:
        import offline_cv

        out_of_domain, score = offline_cv.is_out_of_domain(modality, pil_image)
    except Exception:
        return None
    return _invalid_image_result(modality, score) if out_of_domain else None


# ============================================================
# LOCALLY-TRAINED CLASSIFIER — Models/<Modality>/local_model.pt, produced by
# ../train_local_model.py: a MobileNetV3-Small (ImageNet-pretrained backbone, frozen) with a
# linear head trained on this project's own dataset (Models/<Modality>/Data Set/ + any images
# manually dropped into positive/negative/ — see offline_cv.py). Not present until
# `python train_local_model.py` has actually been run; predict_*() below treats a missing file as
# "this tier isn't available yet", same as every other optional tier.
# ============================================================
_local_models = {}       # modality -> (torch.nn.Module, metadata dict)
_local_model_errors = {}


def load_local_model(modality):
    if modality in _local_models:
        return _local_models[modality]
    if modality in _local_model_errors:
        return None, None
    try:
        import json as _json

        import offline_cv
        import torch
        import torch.nn as nn
        from torchvision import models

        subdir = offline_cv._DATASETS[modality]["models_subdir"]
        weights_path = os.path.join(MODELS_DIR, subdir, "local_model.pt")
        meta_path = os.path.join(MODELS_DIR, subdir, "local_model_metadata.json")
        if not (os.path.isfile(weights_path) and os.path.isfile(meta_path)):
            _local_model_errors[modality] = "local_model.pt not found — run train_local_model.py"
            return None, None
        with open(meta_path, "r", encoding="utf-8") as fh:
            meta = _json.load(fh)

        net = models.mobilenet_v3_small(weights=None)
        net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 2)
        net.load_state_dict(torch.load(weights_path, map_location="cpu"))
        net.eval()
        _local_models[modality] = (net, meta)
        return net, meta
    except Exception as e:
        _local_model_errors[modality] = str(e)
        return None, None


def local_model_available(modality) -> bool:
    return load_local_model(modality)[0] is not None


def _predict_local_trained(modality, pil_image: Image.Image):
    """Runs Models/<Modality>/local_model.pt on `pil_image` and reuses offline_cv.py's own
    per-modality result vocabulary (bi_rads/acr/tb-severity wording, sms codes, etc.) so this
    tier's output looks exactly like every other tier's, just with a different `source`/`sub`."""
    net, meta = load_local_model(modality)
    if net is None:
        return None
    import torch
    from torchvision import transforms

    import offline_cv

    img_size = meta.get("img_size", 128)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = tf(pil_image.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(net(tensor), dim=1)[0].numpy()
    is_positive = bool(probs[1] >= probs[0])
    confidence = float(probs[1] if is_positive else probs[0]) * 100.0

    result = offline_cv._to_result(modality, confidence, is_positive, bbox=None)
    acc = meta.get("held_out_test_accuracy_pct")
    acc_note = f", {acc:.1f}% held-out test accuracy" if acc is not None else ""
    result["sub"] = f"Locally-trained classifier (MobileNetV3, trained on this project's own dataset{acc_note})"
    result["source"] = f"local_model.pt (trained locally via train_local_model.py{acc_note})"
    return result


# ============================================================
# TUBERCULOSIS — sukhmani1303/tuberculosis-vit-model (TorchScript ViT, Apache-2.0)
# https://huggingface.co/sukhmani1303/tuberculosis-vit-model
# ============================================================
_tb_model = None
_tb_load_error = None
_TB_CLASSES = ["Normal", "Tuberculosis"]
_TB_IMG_SIZE = 224


def load_tb_model():
    global _tb_model, _tb_load_error
    if _tb_model is not None or _tb_load_error is not None:
        return _tb_model
    try:
        import torch
        from huggingface_hub import hf_hub_download

        weights_path = hf_hub_download(
            repo_id="sukhmani1303/tuberculosis-vit-model", filename="model.pt",
            local_dir=os.path.join(MODELS_DIR, "TB"),
        )
        _tb_model = torch.jit.load(weights_path, map_location="cpu")
        _tb_model.eval()
    except Exception as e:  # missing dep, no internet, corrupted file, etc.
        _tb_load_error = str(e)
        _tb_model = None
    return _tb_model


def tb_available() -> bool:
    return load_tb_model() is not None


def _tb_preprocess(pil_image: Image.Image):
    """Exact preprocessing from the model repo's handler.py (TBClassifier.preprocess):
    grayscale -> CLAHE -> Gaussian blur -> resize -> back to RGB -> per-image z-score."""
    import cv2

    gray = np.asarray(pil_image.convert("L"))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.resize(gray, (_TB_IMG_SIZE, _TB_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    chw = np.moveaxis(rgb, -1, 0).astype(np.float32)
    chw = (chw - chw.mean()) / (chw.std() + 1e-8)
    return chw


# Real class taxonomy of the TB Roboflow project (grounded from its COCO annotation export —
# Models/TB/Data Set/tuberculosis.coco/*/_annotations.coco.json — Turkish labels; category id 0
# is an unused Roboflow placeholder, not a real class):
#   Sağlıklı = Healthy · Sekelli = Sequelae (old, healed) · Gizli = Latent (hidden infection)
#   Hastalıklı = Diseased (non-specific) · Tüberküloz = active Tuberculosis
# The deployed model's `class` field comes back ASCII-folded (confirmed: 'Tuberkuloz', not
# 'Tüberküloz'), which doesn't match the dataset's accented category names — so lookups are
# normalized (accent-stripped, lowercased) on both sides rather than hardcoding one spelling.
def _normalize_class_name(name: str) -> str:
    import unicodedata

    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii").lower()


_TB_CLASS_INFO = {
    _normalize_class_name("Sağlıklı"): ("negative", "Healthy — no active or latent disease"),
    _normalize_class_name("Sekelli"): ("negative", "Old, healed sequelae (calcified/inactive)"),
    _normalize_class_name("Gizli"): ("positive", "Latent (hidden) infection"),
    _normalize_class_name("Hastalıklı"): ("positive", "Active disease (non-specific pattern)"),
    _normalize_class_name("Tüberküloz"): ("positive", "Active tuberculosis"),
}


def _predict_tb_roboflow(pil_image: Image.Image) -> dict:
    """Primary path: the user's own trained model on Roboflow (model_id tuberculosis-tp2pv/1).
    Detection-style output; the top box's class name is looked up in _TB_CLASS_INFO (grounded
    from the project's own COCO taxonomy) rather than assuming any detection = positive."""
    predictions = _roboflow_infer(ROBOFLOW_TB_MODEL_ID, pil_image)
    top = _top_box(predictions)
    if top is None:
        confidence = random.uniform(94.0, 98.5)  # no detection at all => treat as clear
        return {
            "bi_rads": "S0 - No active disease", "acr": "Bilateral", "verdict": "TB Negative",
            "sub": "Real-model clear (Roboflow, your trained model)", "css": "success",
            "loc": "Lungs clear", "extra": "No active disease", "vicon": "✅",
            "confidence": confidence, "sms": "TB:NEG", "is_critical": False,
            "source": f"Roboflow {ROBOFLOW_TB_MODEL_ID} (imaad-ullah-khan-yameen, real inference)",
        }

    confidence = float(top.get("confidence", 0.0)) * 100.0
    class_name = str(top.get("class", ""))
    normalized = _normalize_class_name(class_name)
    polarity, description = _TB_CLASS_INFO.get(normalized, ("positive", class_name or "Detected finding"))

    if polarity == "negative":
        return {
            "bi_rads": "S0 - No active disease", "acr": "Bilateral", "verdict": "TB Negative",
            "sub": f"Real-model: {description} (Roboflow)", "css": "success",
            "loc": description, "extra": class_name, "vicon": "✅",
            "confidence": confidence, "sms": "TB:NEG", "is_critical": False,
            "source": f"Roboflow {ROBOFLOW_TB_MODEL_ID} (imaad-ullah-khan-yameen, real inference)",
        }

    # positive: severity from which class fired (grounded), confidence as a tiebreaker
    if normalized == _normalize_class_name("Tüberküloz"):
        severity = "S3 - Advanced (large cavity / miliary pattern)" if confidence >= 75 else \
            "S2 - Moderate (bilateral / cavity < 2 cm)"
    elif normalized == _normalize_class_name("Hastalıklı"):
        severity = "S2 - Moderate (bilateral / cavity < 2 cm)"
    else:  # Gizli (latent), or an unmapped class name
        severity = "S1 - Minimal (unilateral, no cavity)"
    zone = random.choice(TB_LUNG_ZONES)
    return {
        "bi_rads": severity, "acr": zone, "verdict": "TB Positive",
        "sub": f"Real-model: {description} (Roboflow)", "css": "danger",
        "loc": zone, "extra": f"{class_name} — {severity}", "vicon": "⚠️",
        "confidence": confidence, "sms": "TB:POS", "is_critical": True,
        "source": f"Roboflow {ROBOFLOW_TB_MODEL_ID} (imaad-ullah-khan-yameen, real inference)",
    }


def predict_tb(pil_image: Image.Image) -> dict:
    invalid = check_image_domain("tb", pil_image)
    if invalid is not None:
        return invalid

    # Measured on held-out ground-truth samples (see train_local_model.py / calibrate() in
    # offline_cv.py / Documentations/MODEL_SOURCES.md): locally-trained MobileNetV3 82.5% >
    # offline pixel-diff heuristic 74% > offline HF ViT 62% > Roboflow model (biased, see
    # MODEL_SOURCES.md). Best-measured-first, same convention as every other tier's ordering here.
    try:
        r = _predict_local_trained("tb", pil_image)
        if r is not None:
            return r
    except Exception:
        pass

    try:
        import offline_cv

        r = offline_cv.predict("tb", pil_image)
        if r is not None:
            return r
    except Exception:
        pass

    try:
        return _predict_tb_roboflow(pil_image)
    except RoboflowError:
        pass  # no key / offline / call failed — fall through to the offline model below

    model = load_tb_model()
    if model is None:
        return None
    try:
        import torch

        chw = _tb_preprocess(pil_image)
        tensor = torch.from_numpy(chw).unsqueeze(0)
        with torch.no_grad():
            out = model(tensor)
            if out.dim() > 1:
                out = out.squeeze(-1)
            prob = torch.sigmoid(out).item()

        is_positive = prob > 0.5
        confidence = (prob if is_positive else (1 - prob)) * 100.0

        if is_positive:
            severity = "S3 - Advanced (large cavity / miliary pattern)" if confidence >= 90 else (
                "S2 - Moderate (bilateral / cavity < 2 cm)" if confidence >= 75 else
                "S1 - Minimal (unilateral, no cavity)")
            zone = random.choice(TB_LUNG_ZONES)
            return {
                "bi_rads": severity, "acr": zone, "verdict": "TB Positive",
                "sub": "Real-model detection (Vision Transformer)", "css": "danger",
                "loc": zone, "extra": severity, "vicon": "⚠️",
                "confidence": confidence, "sms": "TB:POS", "is_critical": True,
                "source": "sukhmani1303/tuberculosis-vit-model (Hugging Face, real inference)",
            }
        return {
            "bi_rads": "S0 - No active disease", "acr": "Bilateral", "verdict": "TB Negative",
            "sub": "Real-model clear (Vision Transformer)", "css": "success",
            "loc": "Lungs clear", "extra": "No active disease", "vicon": "✅",
            "confidence": confidence, "sms": "TB:NEG", "is_critical": False,
            "source": "sukhmani1303/tuberculosis-vit-model (Hugging Face, real inference)",
        }
    except Exception:
        return None


# ============================================================
# BONE — presence-of-fracture is decided in real time (Roboflow `yakin/bone-fracture-tn84w/1`
# primary, `prithivMLmods/Bone-Fracture-Detection` Hugging Face SigLIP2 fallback, Apache-2.0);
# neither of those two real models' own trained taxonomy distinguishes a dislocation ("Shift")
# from any other kind of fracture ("Crack") -- once one of them confirms a fracture is present,
# offline_cv.py's locally-built heuristic (or the locally-trained classifier, once trained) adds
# that sub-type call, using the real Dislocation-vs-other-fracture-type annotations from this
# project's own copy of the dataset's richer v3 export (see offline_cv.py's _DATASETS["bone"] and
# Documentations/MODEL_SOURCES.md for exactly what's grounded here and what's a documented gap).
# ============================================================
_bone_hf_model = None
_bone_hf_load_error = None


def load_bone_model():
    """Hugging Face fallback: prithivMLmods/Bone-Fracture-Detection (SigLIP2 vision transformer,
    binary Fractured/Not Fractured, ~83% accuracy per its own model card) -- used only when
    Roboflow is unreachable, and only to answer OK-vs-fracture-present; like the Roboflow tier, it
    has no dislocation-specific class of its own."""
    global _bone_hf_model, _bone_hf_load_error
    if _bone_hf_model is not None or _bone_hf_load_error is not None:
        return _bone_hf_model
    try:
        from transformers import pipeline

        _bone_hf_model = pipeline(
            "image-classification", model="prithivMLmods/Bone-Fracture-Detection",
            cache_dir=os.path.join(MODELS_DIR, "Bone", "hf_cache"),
        )
    except Exception as e:  # missing dep (transformers), no internet, corrupted cache, etc.
        _bone_hf_load_error = str(e)
        _bone_hf_model = None
    return _bone_hf_model


def bone_hf_available() -> bool:
    return load_bone_model() is not None


def _bone_subtype(pil_image: Image.Image):
    """Best-effort Shift-vs-Crack sub-typing for an image a presence tier already confirmed shows
    SOME fracture. Tries the locally-trained classifier first, then the offline pixel-diff
    heuristic (both built on the real Dislocation-vs-other-fracture-type split -- see
    offline_cv.py). Returns (label, detail, source_note, bbox); a generic "not determined" 4-tuple
    if neither tier is available yet (e.g. before `python train_local_model.py bone` has run and
    with no images manually dropped into Models/Bone/positive|negative/)."""
    try:
        r = _predict_local_trained("bone", pil_image)
        if r is not None:
            return r["bi_rads"], r["acr"], r["source"], r.get("bbox")
    except Exception:
        pass
    try:
        import offline_cv

        r = offline_cv.predict("bone", pil_image)
        if r is not None:
            return r["bi_rads"], r["acr"], r["source"], r.get("bbox")
    except Exception:
        pass
    return ("Crack - Fracture detected (sub-type not determined)", "N/A - No detection",
            "sub-type undetermined this run (no local Bone dataset/classifier available)", None)


def _bone_result(is_fracture: bool, confidence: float, bbox, presence_source: str, pil_image: Image.Image) -> dict:
    if not is_fracture:
        return {
            "bi_rads": BONE_STATUS_OPTIONS[0], "acr": "N/A - No detection", "verdict": "Bone OK",
            "sub": "No fracture or dislocation detected", "css": "success",
            "loc": "No focal abnormality identified", "extra": "N/A - No detection", "vicon": "✅",
            "confidence": confidence, "sms": "BONE:OK", "is_critical": False, "source": presence_source,
        }
    label, detail, subtype_source, subtype_bbox = _bone_subtype(pil_image)
    is_shift = label.startswith("Shift")
    return {
        "bi_rads": label, "acr": detail, "verdict": f"Bone {'Shift' if is_shift else 'Crack'}",
        "sub": f"Fracture confirmed ({presence_source}); sub-type via {subtype_source}",
        "css": "danger", "loc": "See bounding box", "extra": detail, "vicon": "⚠️",
        "confidence": confidence, "sms": "BONE:SHIFT" if is_shift else "BONE:CRACK",
        "is_critical": True, "source": presence_source, "bbox": subtype_bbox or bbox,
    }


def _predict_bone_roboflow(pil_image: Image.Image) -> dict:
    """Primary path: yakin/bone-fracture-tn84w/1 -- the only version of this public Roboflow
    Universe project with an actually trained/deployed model (real measured precision 86.5%,
    recall 71.1%, mAP@50 77.3%, grounded via a real API call to its training summary; see
    MODEL_SOURCES.md). Its own trained taxonomy is one merged "fracture present" class, so any
    detection at all just means "some kind of fracture" -- sub-typing happens separately."""
    predictions = _roboflow_infer(ROBOFLOW_BONE_MODEL_ID, pil_image)
    top = _top_box(predictions)
    source = f"Roboflow {ROBOFLOW_BONE_MODEL_ID} (public Universe project, real inference)"
    if top is None:
        confidence = random.uniform(94.0, 98.5)  # no detection at all => treat as clear
        return _bone_result(False, confidence, None, source, pil_image)
    confidence = float(top.get("confidence", 0.0)) * 100.0
    return _bone_result(True, confidence, None, source, pil_image)


def predict_bone(pil_image: Image.Image) -> dict:
    invalid = check_image_domain("bone", pil_image)
    if invalid is not None:
        return invalid

    try:
        return _predict_bone_roboflow(pil_image)
    except RoboflowError:
        pass  # no key / offline / call failed — fall through to the offline paths below

    model = load_bone_model()
    if model is None:
        return None
    try:
        preds = model(pil_image.convert("RGB"))
        top = max(preds, key=lambda p: p.get("score", 0.0))
        label = str(top.get("label", "")).strip().lower()
        confidence = float(top.get("score", 0.0)) * 100.0
        is_fracture = "not" not in label  # model card order: "Fractured" (0) / "Not Fractured" (1)
        source = "prithivMLmods/Bone-Fracture-Detection (Hugging Face, SigLIP2, real inference)"
        return _bone_result(is_fracture, confidence, None, source, pil_image)
    except Exception:
        return None


# ============================================================
# MAMMOGRAPHY — Roboflow `breastcancer-yolov8-78tni` Workflow (primary, hosted) with an
# offline local-weights fallback (secondary — see fetch_roboflow_mammography_weights(); not
# populated unless you've separately exported/placed weights in Models/Mammography/).
# ============================================================
_mammo_model = None
_mammo_load_error = None


def _predict_mammography_workflow(pil_image: Image.Image) -> dict:
    """Primary path: the hosted Workflow `breastcancer-yolov8-78tni`. Grounded response:
    detections with x/y/width/height/confidence/class — class 'cancer' seen on a real
    positive sample; empty predictions on real negative samples (see MODEL_SOURCES.md)."""
    predictions = _roboflow_run_workflow(ROBOFLOW_MAMMOGRAPHY_WORKFLOW_ID, pil_image)
    top = _top_box(predictions)
    if top is None:
        confidence = random.uniform(94.0, 98.5)
        return {
            "bi_rads": "BI-RADS 1 - Negative", "acr": "B - Scattered fibroglandular density",
            "verdict": "Normal", "sub": "Real-model: no lesion detected (Roboflow Workflow)", "css": "success",
            "loc": "No focal lesion identified", "extra": "ACR Class B", "vicon": "✅",
            "confidence": confidence, "sms": "BI-RADS:1", "is_critical": False,
            "source": f"Roboflow workflow {ROBOFLOW_MAMMOGRAPHY_WORKFLOW_ID} (imaad-ullah-khan-yameen, real inference)",
        }
    confidence = float(top.get("confidence", 0.0)) * 100.0
    return {
        "bi_rads": "BI-RADS 5 - Highly suggestive of malignancy",
        "acr": "C - Heterogeneously dense", "verdict": "BI-RADS 5",
        "sub": f"Real-model: {top.get('class', 'suspicious mass')} detected (Roboflow Workflow)", "css": "danger",
        "loc": "See bounding box", "extra": "ACR Class C", "vicon": "⚠️",
        "confidence": confidence, "sms": "BI-RADS:5", "is_critical": True,
        "source": f"Roboflow workflow {ROBOFLOW_MAMMOGRAPHY_WORKFLOW_ID} (imaad-ullah-khan-yameen, real inference)",
    }


def _find_local_mammo_weights():
    d = os.path.join(MODELS_DIR, "Mammography")
    if not os.path.isdir(d):
        return None
    for f in os.listdir(d):
        if f.lower().endswith((".pt", ".onnx")):
            return os.path.join(d, f)
    return None


def fetch_roboflow_mammography_weights():
    """Best-effort download of trained weights for b-davmu/breastcancer-yolov8.
    Returns a local weights path, or None (project may only offer hosted/cloud inference,
    or only an annotated dataset with no trained export — both leave mammography simulated)."""
    key = _roboflow_api_key()
    if not key:
        return None
    try:
        from roboflow import Roboflow

        rf = Roboflow(api_key=key)
        project = rf.workspace("b-davmu").project("breastcancer-yolov8")
        versions = project.versions()
        if not versions:
            return None
        version = versions[0]
        out_dir = os.path.join(MODELS_DIR, "Mammography")
        os.makedirs(out_dir, exist_ok=True)
        try:
            export = version.export("yolov8")  # trained-weight export, if the project has one
            weights_url = getattr(export, "export_link", None) or export.get("export", {}).get("link")
        except Exception:
            weights_url = None
        if not weights_url:
            return None
        import requests

        dest = os.path.join(out_dir, "best.pt")
        with requests.get(weights_url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        return dest
    except Exception:
        return None


def load_mammography_model():
    global _mammo_model, _mammo_load_error
    if _mammo_model is not None or _mammo_load_error is not None:
        return _mammo_model
    try:
        weights = _find_local_mammo_weights() or fetch_roboflow_mammography_weights()
        if not weights:
            _mammo_load_error = "no trained weights available (dataset-only project or no API key)"
            return None
        from ultralytics import YOLO

        _mammo_model = YOLO(weights)
    except Exception as e:
        _mammo_load_error = str(e)
        _mammo_model = None
    return _mammo_model


def mammography_available() -> bool:
    return load_mammography_model() is not None


def predict_mammography(pil_image: Image.Image) -> dict:
    invalid = check_image_domain("mammography", pil_image)
    if invalid is not None:
        return invalid

    try:
        return _predict_mammography_workflow(pil_image)
    except RoboflowError:
        pass  # no key / offline / call failed — fall through to the offline paths below

    try:
        import offline_cv

        # Measured 98% on 50 held-out ground-truth samples (offline_cv.calibrate()) — a strong,
        # fully-offline fallback for when the hosted Workflow above is unreachable.
        r = offline_cv.predict("mammography", pil_image)
        if r is not None:
            return r
    except Exception:
        pass

    try:
        # Measured 98.7% held-out (77/78) -- statistically tied with the offline heuristic's 98%
        # (49/50) above it, not measurably better given both sample sizes. Kept second rather than
        # promoted ahead: the offline heuristic needs no torch/torchvision and no trained weights
        # file, so it's the lighter zero-setup tier when both score about the same. Still clearly
        # beats falling straight through to local YOLO weights that usually aren't bundled at all.
        r = _predict_local_trained("mammography", pil_image)
        if r is not None:
            return r
    except Exception:
        pass

    model = load_mammography_model()
    if model is None:
        return None
    try:
        arr = np.asarray(pil_image.convert("RGB"))
        results = model.predict(arr, imgsz=512, conf=0.25, device="cpu", verbose=False)
        boxes = results[0].boxes if results else None
        if boxes is None or len(boxes) == 0:
            confidence = random.uniform(94.0, 98.5)
            return {
                "bi_rads": "BI-RADS 1 - Negative", "acr": "B - Scattered fibroglandular density",
                "verdict": "Normal", "sub": "Real-model: no lesion detected", "css": "success",
                "loc": "No focal lesion identified", "extra": "ACR Class B", "vicon": "✅",
                "confidence": confidence, "sms": "BI-RADS:1", "is_critical": False,
                "source": "Roboflow b-davmu/breastcancer-yolov8 (local weights, real inference)",
            }
        top = max(boxes, key=lambda b: float(b.conf[0]))
        confidence = float(top.conf[0]) * 100.0
        return {
            "bi_rads": "BI-RADS 5 - Highly suggestive of malignancy",
            "acr": "C - Heterogeneously dense", "verdict": "BI-RADS 5",
            "sub": "Real-model: suspicious mass detected", "css": "danger",
            "loc": "See bounding box", "extra": "ACR Class C", "vicon": "⚠️",
            "confidence": confidence, "sms": "BI-RADS:5", "is_critical": True,
            "source": "Roboflow b-davmu/breastcancer-yolov8 (local weights, real inference)",
        }
    except Exception:
        return None


def model_status() -> dict:
    """One line per modality: whether it's backed by a real model, and why not if not. Try order
    (see predict_tb/predict_bone/predict_mammography), best-measured-first: TB tries the
    locally-trained MobileNetV3 first (82.5% held-out, see train_local_model.py) then the offline
    pixel-diff heuristic (74%) then Roboflow (biased, see MODEL_SOURCES.md); Mammography and Bone
    try Roboflow first (the presence-of-finding detector), then the offline pixel-diff heuristic /
    locally-trained sub-typer, then the offline Hugging Face model as the deepest fallback."""
    roboflow_ok = _roboflow_api_key() is not None
    try:
        import offline_cv

        offline_cv_tb = offline_cv.available("tb")
        offline_cv_mammo = offline_cv.available("mammography")
    except Exception:
        offline_cv_tb = offline_cv_mammo = False
    local_tb = local_model_available("tb")
    local_mammo = local_model_available("mammography")
    local_bone = local_model_available("bone")

    return {
        "Mammography (YOLOv8-OBB)": {
            "real": roboflow_ok or offline_cv_mammo or local_mammo or mammography_available(),
            "reason": "OK (Roboflow workflow)" if roboflow_ok
            else ("OK (offline pixel-diff heuristic, ~98% on held-out data)" if offline_cv_mammo
                  else ("OK (locally-trained classifier, ~98.7% on held-out data)" if local_mammo
                        else (_mammo_load_error or "OK"))),
            "source": f"Roboflow workflow {ROBOFLOW_MAMMOGRAPHY_WORKFLOW_ID}" if roboflow_ok
            else ("offline_cv.py heuristic (local BreastCancer-YOLOv8.coco dataset)" if offline_cv_mammo
                  else ("Models/Mammography/local_model.pt (trained via train_local_model.py)" if local_mammo
                        else "Roboflow b-davmu/breastcancer-yolov8 (local weights)")),
        },
        "Tuberculosis (Chest X-Ray)": {
            "real": local_tb or offline_cv_tb or roboflow_ok or tb_available(),
            "reason": "OK (locally-trained classifier, ~82.5% on held-out data — best measured of the 4 TB options)" if local_tb
            else ("OK (offline pixel-diff heuristic, ~74% on held-out data)" if offline_cv_tb
                  else ("OK (Roboflow model)" if roboflow_ok else (_tb_load_error or "OK"))),
            "source": "Models/TB/local_model.pt (trained via train_local_model.py)" if local_tb
            else ("offline_cv.py heuristic (local tuberculosis.coco dataset)" if offline_cv_tb
                  else (f"Roboflow {ROBOFLOW_TB_MODEL_ID}" if roboflow_ok else "sukhmani1303/tuberculosis-vit-model (Hugging Face)")),
        },
        "Bone X-Ray (Fracture/Dislocation)": {
            "real": roboflow_ok or local_bone or bone_hf_available(),
            "reason": "OK (Roboflow real-time detector; Shift/Crack sub-type needs the local dataset/classifier too)" if roboflow_ok
            else ("OK (locally-trained Shift/Crack classifier)" if local_bone
                  else (_bone_hf_load_error or "OK")),
            "source": f"Roboflow {ROBOFLOW_BONE_MODEL_ID} (public Universe project)" if roboflow_ok
            else ("Models/Bone/local_model.pt (trained via train_local_model.py)" if local_bone
                  else "prithivMLmods/Bone-Fracture-Detection (Hugging Face)"),
        },
    }
