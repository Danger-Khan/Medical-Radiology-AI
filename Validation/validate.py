#!/usr/bin/env python3
"""
Pink Edge AI (Desktop) — validation suite.
============================================
Self-contained smoke/validation test for GUI.py + inference.py + streamlit_app.py: imports,
imaging, simulated scenario generators, SQLite cache round-trip, text/PDF report generation, real
model inference for all three modalities (each tries its Roboflow-hosted model first, then an
offline Hugging Face/local-weights fallback, then SIMULATED — see inference.py), a ground-truth
cross-check against a COCO-annotated mammography sample, the run_triage() dispatcher, and that
both the Tkinter UI (hidden window, no mainloop) and the Streamlit UI (AppTest, no browser)
actually build and can run one triage cycle.

Run with:  python Validation/validate.py   (from anywhere — paths below are anchored to the
repo root, not the current working directory)
Exits 0 if every check passes, 1 otherwise. Uses a throwaway DB file under Validation/ (never
touches the app's real pink_edge_cache.db at the project root) and cleans up after itself.
"""
import os
import sys
import time
import traceback
from glob import glob

VALIDATION_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(VALIDATION_DIR)
TEST_DATA_DIR = os.path.join(ROOT_DIR, "Test Data")
sys.path.insert(0, ROOT_DIR)  # so `import GUI` / `import inference` resolve from Validation/

RESULTS = []  # (name, ok, detail)


def check(name):
    """Decorator: run fn(), record PASS/FAIL, never let one failure kill the run."""
    def wrap(fn):
        t0 = time.time()
        try:
            fn()
            RESULTS.append((name, True, f"{time.time() - t0:.2f}s"))
        except Exception as e:
            RESULTS.append((name, False, f"{type(e).__name__}: {e}"))
            traceback.print_exc()
        return fn
    return wrap


def require(cond, msg="assertion failed"):
    if not cond:
        raise AssertionError(msg)


RESULT_KEYS = {"bi_rads", "acr", "verdict", "sub", "loc", "extra", "vicon", "confidence", "sms", "is_critical"}


def validate_result_shape(r, where):
    require(isinstance(r, dict), f"{where}: result is not a dict")
    missing = RESULT_KEYS - set(r.keys())
    require(not missing, f"{where}: missing keys {missing}")
    require(0.0 <= r["confidence"] <= 100.0, f"{where}: confidence out of range: {r['confidence']}")
    require(isinstance(r["is_critical"], bool), f"{where}: is_critical not bool")


# ============================================================
print("=" * 70)
print("PINK EDGE AI (DESKTOP) — VALIDATION SUITE")
print("=" * 70)

# ---- 1. imports ----
@check("import GUI and inference modules")
def _():
    global GUI, inf
    import GUI as GUI  # noqa
    import inference as inf  # noqa


# ---- 2. imaging ----
@check("placeholder image generation (mammogram/xray/ultrasound)")
def _():
    for model in GUI.MODELS:
        img = GUI.load_placeholder(model, seed=1)
        require(img.size == (512, 512), f"{model}: wrong size {img.size}")
        require(img.mode == "L", f"{model}: wrong mode {img.mode}")


@check("draw_bbox overlay (all 3 modalities, critical + non-critical)")
def _():
    img = GUI.gen_xray(seed=2)
    for model in GUI.MODELS:
        for crit in (True, False):
            fake = {"confidence": 88.5, "is_critical": crit}
            out = GUI.draw_bbox(img, model, fake)
            require(out.size == (512, 512) and out.mode == "RGB", f"{model} crit={crit}: bad overlay output")


# ---- 3. simulated scenario generators ----
@check("simulated scenario generators (mammography/tb/maternal)")
def _():
    for fn, where in [(GUI.sim_mammography, "mammography"), (GUI.sim_tb, "tb"), (GUI.sim_maternal, "maternal")]:
        for _i in range(5):
            r = fn()
            validate_result_shape(r, f"sim_{where}")
            require("source" in r, f"sim_{where}: no source field")


# ---- 4. SQLite cache round-trip (throwaway DB, never the real one) ----
TEST_DB = os.path.join(VALIDATION_DIR, "validation_test_cache.db")


@check("SQLite cache round-trip (init/save/get/sync/counts) on throwaway DB")
def _():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    orig_path = GUI.DB_PATH
    GUI.DB_PATH = TEST_DB
    try:
        GUI.init_db()
        t0, u0 = GUI.counts()
        require(t0 == 0 and u0 == 0, "fresh DB should be empty")

        data = {
            "patient_id": 12345678, "patient_age": 34, "modality": "DX", "model_used": "Tuberculosis",
            "bi_rads": "S1 - Minimal (unilateral, no cavity)", "acr_density": "Upper Zone",
            "verdict": "TB Positive", "localization": "Right Upper Lobe", "confidence": 91.2,
            "inference_time": 1.4, "timestamp": "2026-09-12 20:00:00", "network_mode": "Fully Offline",
            "model_source": "unit-test", "synced": 0,
        }
        GUI.save_to_cache(data)
        total, unsynced = GUI.counts()
        require(total == 1 and unsynced == 1, f"expected 1/1 after save, got {total}/{unsynced}")

        rows = GUI.get_cached_reports()
        require(len(rows) == 1, "get_cached_reports should return the saved row")
        # patient_id column has TEXT affinity (schema ported as-is from the original app), so
        # SQLite stores the inserted int as "12345678" — compare as str, not equality-of-types.
        require(str(rows[0][1]) == "12345678", f"patient_id mismatch in saved row: {rows[0][1]!r}")

        unsynced_rows = GUI.get_unsynced_reports()
        require(len(unsynced_rows) == 1, "should have 1 unsynced row")
        GUI.mark_as_synced([unsynced_rows[0][0]])
        total2, unsynced2 = GUI.counts()
        require(total2 == 1 and unsynced2 == 0, f"expected 1/0 after sync, got {total2}/{unsynced2}")

        iot = GUI.simulate_iot_sync(rows)
        require(len(iot) == 1 and "iot_id" in iot[0], "simulate_iot_sync malformed output")
        oss = GUI.simulate_oss_upload(rows[0])
        require("high" in oss, "simulate_oss_upload malformed output")
        acr = GUI.simulate_acr_check()
        require("current" in acr and "available" in acr, "simulate_acr_check malformed output")
    finally:
        GUI.DB_PATH = orig_path
        if os.path.exists(TEST_DB):
            os.remove(TEST_DB)


# ---- 5. reports ----
@check("text + PDF report generation")
def _():
    d = {
        "patient_id": 87654321, "patient_age": 29, "modality": "MG (Mammography)",
        "model_used": "Mammography", "bi_rads": "BI-RADS 5 - Highly suggestive of malignancy",
        "acr_density": "C - Heterogeneously dense", "verdict": "BI-RADS 5",
        "localization": "Upper Outer Quadrant", "confidence": 94.3, "inference_time": 8.1,
        "timestamp": "2026-09-12 20:00:00", "network_mode": "Fully Offline",
        "model_source": "unit-test", "synced": 0,
    }
    txt = GUI.generate_text_report(d)
    require("PINK EDGE AI" in txt and str(d["patient_id"]) in txt, "text report missing expected content")
    require("URGENT" in txt, "BI-RADS 5 should trigger an URGENT referral line")

    pdf_bytes = GUI.generate_pdf_bytes(d)
    require(pdf_bytes is not None, "PDF generation returned None (fpdf2 missing?)")
    require(isinstance(pdf_bytes, bytes) and len(pdf_bytes) > 500, "PDF output looks too small/invalid")
    require(pdf_bytes[:4] == b"%PDF", "PDF output does not start with a %PDF header")


# ---- 6. real model inference (TB + Maternal) ----
@check("TB real-model inference on a synthetic chest X-ray")
def _():
    require(inf.tb_available(), f"TB model failed to load: {inf._tb_load_error}")
    img = GUI.gen_xray(seed=3)
    r = inf.predict_tb(img)
    require(r is not None, "predict_tb returned None despite model being available")
    validate_result_shape(r, "predict_tb")
    require("real inference" in r["source"], "TB result should be tagged as real inference")


@check("Maternal real-model inference on a synthetic ultrasound")
def _():
    require(inf.maternal_available(), f"Maternal model failed to load: {inf._maternal_load_error}")
    img = GUI.gen_ultrasound(seed=4)
    r = inf.predict_maternal(img)
    require(r is not None, "predict_maternal returned None despite model being available")
    validate_result_shape(r, "predict_maternal")
    require("real inference" in r["source"], "Maternal result should be tagged as real inference")


@check("Mammography: real Roboflow workflow if a key is configured, else SIMULATED")
def _():
    has_key = inf._roboflow_api_key() is not None
    img = GUI.gen_mammogram(seed=5)
    direct = inf.predict_mammography(img)
    if has_key:
        # A key is configured (this project's roboflow_key.txt) -> the hosted Workflow should
        # answer for real, not fall through to local weights or return None.
        require(direct is not None, "predict_mammography returned None despite a Roboflow key being configured")
        validate_result_shape(direct, "predict_mammography(roboflow)")
        require("real inference" in direct["source"], "expected a real-inference source with a key configured")
    elif not inf._find_local_mammo_weights():
        require(direct is None, "predict_mammography should return None with no key and no local weights")


# ---- 6b. real model inference on the real sample images in Test Data/ ----
@check("TB real-model inference on real sample X-rays (Test Data/Tuberculosis)")
def _():
    from PIL import Image

    files = sorted(glob(os.path.join(TEST_DATA_DIR, "Tuberculosis", "*")))
    require(len(files) > 0, f"no sample files found under {TEST_DATA_DIR}\\Tuberculosis")
    for f in files:
        img = Image.open(f)
        r = inf.predict_tb(img)
        require(r is not None, f"predict_tb returned None on real sample: {os.path.basename(f)}")
        validate_result_shape(r, f"predict_tb({os.path.basename(f)})")
        print(f"    {os.path.basename(f):40s} -> {r['verdict']:15s} ({r['confidence']:.1f}%)")


@check("Mammography pathway on real samples (Test Data/Breast Cancer)")
def _():
    from PIL import Image

    files = sorted(glob(os.path.join(TEST_DATA_DIR, "Breast Cancer", "*")))
    require(len(files) > 0, f"no sample files found under {TEST_DATA_DIR}\\Breast Cancer")
    for f in files[:3]:  # a few, not all 40+ — keep this check fast
        img = Image.open(f)
        r = GUI.run_triage("Mammography (YOLOv8-OBB)", img)
        validate_result_shape(r, f"run_triage(mammography, {os.path.basename(f)})")
        print(f"    {os.path.basename(f):40s} -> {r['verdict']:15s} ({r['source']})")


@check("Mammography Roboflow workflow matches ground truth on an annotated sample")
def _():
    """Cross-checks against the COCO annotation, not just 'did it run' — a stronger signal
    than the other checks that the real model is actually doing something sensible."""
    from PIL import Image

    if not inf._roboflow_api_key():
        return  # no key configured in this environment — nothing to cross-check
    ds_dir = os.path.join(ROOT_DIR, "Models", "Mammography", "Data Set", "BreastCancer-YOLOv8.coco", "test")
    ann_path = os.path.join(ds_dir, "_annotations.coco.json")
    require(os.path.isfile(ann_path), f"no COCO annotations found at {ann_path}")
    import json

    with open(ann_path, "r", encoding="utf-8") as fh:
        coco = json.load(fh)
    require(len(coco["annotations"]) > 0, "COCO file has no annotations to cross-check against")
    img_id = coco["annotations"][0]["image_id"]
    file_name = next(im["file_name"] for im in coco["images"] if im["id"] == img_id)
    img = Image.open(os.path.join(ds_dir, file_name))

    r = inf.predict_mammography(img)
    require(r is not None, "predict_mammography returned None on a known-positive annotated sample")
    require(r["is_critical"], f"ground truth has a cancer annotation but the model said {r['verdict']!r}")
    print(f"    {file_name} (ground truth: cancer) -> {r['verdict']} ({r['confidence']:.1f}%)")


def _images_for_category(coco_path, ds_dir, category_name, limit=3):
    """Loads a COCO annotation file and returns up to `limit` (PIL Image, file_name) pairs for
    images annotated with the given category name. Matches ALL category ids with that name —
    some Roboflow COCO exports have more than one id sharing a name (e.g. an unused id 0
    placeholder alongside the real one) — a naive "first id with this name" lookup can silently
    match zero real annotations."""
    import json

    from PIL import Image

    with open(coco_path, "r", encoding="utf-8") as fh:
        coco = json.load(fh)
    cat_ids = {c["id"] for c in coco["categories"] if c["name"] == category_name}
    if not cat_ids:
        return []
    anns = [a for a in coco["annotations"] if a["category_id"] in cat_ids]
    out = []
    seen_images = set()
    for ann in anns:
        if ann["image_id"] in seen_images:
            continue
        seen_images.add(ann["image_id"])
        file_name = next(im["file_name"] for im in coco["images"] if im["id"] == ann["image_id"])
        out.append((Image.open(os.path.join(ds_dir, file_name)), file_name))
        if len(out) >= limit:
            break
    return out


def _first_image_for_category(coco_path, ds_dir, category_name):
    imgs = _images_for_category(coco_path, ds_dir, category_name, limit=1)
    return imgs[0] if imgs else None


@check("TB Roboflow model matches ground truth (positive + negative annotated samples)")
def _():
    if not inf._roboflow_api_key():
        return
    ds_dir = os.path.join(ROOT_DIR, "Models", "TB", "Data Set", "tuberculosis.coco", "test")
    ann_path = os.path.join(ds_dir, "_annotations.coco.json")
    require(os.path.isfile(ann_path), f"no COCO annotations found at {ann_path}")

    pos = _first_image_for_category(ann_path, ds_dir, "Tüberküloz")
    require(pos is not None, "no 'Tüberküloz' (active TB) annotated sample found to cross-check against")
    img, name = pos
    r = inf.predict_tb(img)
    require(r is not None, "predict_tb returned None on a known-positive annotated sample")
    require(r["is_critical"], f"ground truth is active TB but the model said {r['verdict']!r}")
    print(f"    {name} (ground truth: Tüberküloz) -> {r['verdict']} ({r['confidence']:.1f}%)")

    # Majority vote over a few samples, not a single one: a real model calling one borderline
    # image wrong is normal (this one has an rfdetr-small model with modest class balance —
    # 159 'Sağlıklı' vs 432 'Tüberküloz' annotations); the integration is only broken if it's
    # wrong most of the time, not if it's wrong once.
    negatives = _images_for_category(ann_path, ds_dir, "Sağlıklı", limit=3)
    require(len(negatives) > 0, "no 'Sağlıklı' (healthy) annotated sample found to cross-check against")
    correct = 0
    for img2, name2 in negatives:
        r2 = inf.predict_tb(img2)
        require(r2 is not None, f"predict_tb returned None on known-negative sample {name2}")
        ok = not r2["is_critical"]
        correct += ok
        print(f"    {name2} (ground truth: Sağlıklı) -> {r2['verdict']} ({r2['confidence']:.1f}%) {'OK' if ok else 'MISS'}")
    require(correct * 2 >= len(negatives),
            f"model called {len(negatives) - correct}/{len(negatives)} known-healthy samples TB-positive — "
            f"worse than a coin flip, likely a real integration bug, not just model noise")


@check("Maternal Health Roboflow model matches ground truth on an annotated sample")
def _():
    if not inf._roboflow_api_key():
        return
    ds_dir = os.path.join(ROOT_DIR, "Models", "Maternal", "Data Set", "HASH Maternal Health.coco", "test")
    ann_path = os.path.join(ds_dir, "_annotations.coco.json")
    require(os.path.isfile(ann_path), f"no COCO annotations found at {ann_path}")

    pos = _first_image_for_category(ann_path, ds_dir, "abnormal")
    require(pos is not None, "no 'abnormal' annotated sample found to cross-check against")
    img, name = pos
    r = inf.predict_maternal(img)
    require(r is not None, "predict_maternal returned None on a known-abnormal annotated sample")
    require(r["is_critical"], f"ground truth is abnormal but the model said {r['verdict']!r}")
    print(f"    {name} (ground truth: abnormal) -> {r['verdict']} ({r['confidence']:.1f}%)")


# ---- 7. run_triage() dispatcher (what the UI actually calls) ----
@check("run_triage() dispatcher end-to-end for all 3 modalities")
def _():
    for model in GUI.MODELS:
        img = GUI.load_placeholder(model, seed=6)
        r = GUI.run_triage(model, img)
        validate_result_shape(r, f"run_triage({model})")
        require("source" in r, f"run_triage({model}): no source field")


# ---- 8. Tkinter UI builds without error (hidden window, no mainloop) ----
@check("Tkinter UI construction (hidden root, no mainloop)")
def _():
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    try:
        app = GUI.PinkEdgeApp(root)
        root.update()
        require(app.notebook.index("end") == 3, "expected 3 notebook tabs")
        # exercise the model dropdown + triage path through the real widget code
        app.model_var.set("Tuberculosis (Chest X-Ray)")
        app._on_model_change()
        app._run_triage()
        root.update()
        require(app.inference_done, "UI-driven _run_triage() did not complete")
        require(app.current_result is not None, "UI-driven _run_triage() produced no result")
    finally:
        root.destroy()


# ---- 9. Streamlit UI builds and runs one triage cycle (AppTest, no browser) ----
@check("Streamlit UI construction + one triage cycle (AppTest, headless)")
def _():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(ROOT_DIR, "streamlit_app.py"))
    at.run(timeout=60)
    require(not at.exception, f"initial script run raised: {at.exception}")

    # sidebar buttons in creation order: EN, Urdu, Run Triage, ...
    at.sidebar.button[2].click().run(timeout=90)
    require(not at.exception, f"Run Triage click raised: {at.exception}")
    require(at.session_state["inference_done"], "Run Triage did not complete in the Streamlit app")
    r = at.session_state["current_result"]
    validate_result_shape(r, "streamlit run_triage")


# ============================================================
print()
width = max(len(n) for n, _, _ in RESULTS) + 2
n_pass = sum(1 for _, ok, _ in RESULTS if ok)
for name, ok, detail in RESULTS:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name.ljust(width)} {detail}")

print()
print(f"{n_pass}/{len(RESULTS)} checks passed.")
if n_pass < len(RESULTS):
    print("\nModel status (inference.py):")
    try:
        for modality, info in inf.model_status().items():
            print(f"  - {modality}: real={info['real']}  reason={info['reason']}")
    except Exception:
        pass
    sys.exit(1)
sys.exit(0)
