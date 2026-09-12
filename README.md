# Pink Edge AI — Offline Desktop + Responsive Web Editions

Two sibling UIs over the same shared logic, ported from the `Pink_Edge_AI-main` Streamlit hackathon
demo (see `App/Misc/`): a Tkinter **desktop** app (`GUI.py`) and a responsive **Streamlit web** app
(`streamlit_app.py`). Both share the same SQLite report cache and the same model backend
(`inference.py`). Everything the desktop/web editions need lives in **`App/`**; two standalone,
optional pieces live at the repo root next to it instead — a local Android build (`Apk/`) and RK3588
edge-node deployment code (`RK3588 SBC/`) — since neither is part of running the desktop/web app
itself. See `## Project layout`.

## What this is

Clinical-triage UI for three modalities — Mammography, Tuberculosis (chest X-ray), Maternal Health
(ultrasound) — matching the original app's dashboard, hospital-hub alert feed, and simulated
Alibaba-Cloud-sync panel. Each modality tries multiple methods in an order set by **measured
accuracy against real ground truth**, not by "online first":

| Modality | Try order | Measured accuracy |
|---|---|---|
| Mammography | Roboflow Workflow `breastcancer-yolov8-78tni` → offline pixel-diff heuristic → locally-trained classifier → Simulated | Roboflow: correctly flagged one ground-truth sample (90.9% confidence). Offline heuristic: **98%** (49/50). Locally-trained classifier: **98.7%** (77/78, tied with the heuristic within sample-size noise) |
| Tuberculosis | **Locally-trained classifier** → offline pixel-diff heuristic → Roboflow model → offline HF ViT | Locally-trained classifier: **82.5%** (best of the four — Roboflow model has a ⚠️ known issue, see MODEL_SOURCES.md) |
| Maternal Health | Roboflow model `hash-maternal-health/1` → offline HF CNN | Roboflow: correctly flagged ground truth (88.3%). (No locally-trained classifier yet — dataset has zero negative images) |

The **offline pixel-diff heuristic** (`App/offline_cv.py`) needs no model weights and no internet,
ever — it builds "typical positive" / "typical negative" reference images by averaging your own
local labeled datasets (`App/Models/*/Data Set/`) and compares new images against both, drawing a
real bounding box around whatever region actually differs. Run `python offline_cv.py` (from `App/`)
to see its measured accuracy against held-out data for yourself. The **locally-trained classifier**
(`App/train_local_model.py`) is a genuine trained model (MobileNetV3, transfer-learned) fine-tuned
directly on this project's own datasets — run `python train_local_model.py` (from `App/`) to (re)train
it and print its own held-out accuracy; a modality only gets it ahead of another tier once it
measurably beats that tier on the same held-out split. The Roboflow calls need a key + internet
(cloud-dependent at inference time — a deliberate tradeoff for real predictions when available).
Before any of that runs, every upload is also checked for **out-of-domain input** — if you upload,
say, an ultrasound into the Mammography tab (or anything that just doesn't look like the right kind
of scan), it says so ("Wrong Image Type") instead of forcing a triage verdict onto the wrong image
type; see MODEL_SOURCES.md for the measured catch rate per modality. Full detail (grounding, exact
preprocessing, license, the TB accuracy finding) is in
[App/Documentations/MODEL_SOURCES.md](App/Documentations/MODEL_SOURCES.md). This is a hackathon-grade
demo, not a validated medical device — confidence numbers and severity mappings are illustrative.

## Run it

**Desktop (Tkinter):**
```
Start.bat
```
or manually: `cd App`, `pip install -r requirements.txt`, `python GUI.py`

**Web (Streamlit, responsive — resizes down to phone/tablet widths):**
```
Start_Web.bat
```
or manually: `cd App`, `pip install -r requirements.txt`, `streamlit run streamlit_app.py`
(opens `http://localhost:8501` in your browser; `--server.address 0.0.0.0` if you want it reachable
from another device on your LAN)

Both launchers `cd` into `App/` for you, then share `App/requirements.txt`. First run downloads
~1-2 GB of Python deps (PyTorch/Ultralytics) plus the offline fallback model weights (needs
internet once). The offline fallbacks then run without internet on every later run; the
Roboflow-hosted primaries need internet + a key every time (see above) — weights/datasets are
cached under `App/Models/`, and the report cache (`App/pink_edge_cache.db`, SQLite) is local and
shared by both editions.

## Enabling the Roboflow-hosted models (real, purpose-trained)

All three modalities check for a Roboflow API key and use it if present:

1. Get a key from [roboflow.com](https://roboflow.com) → your workspace → Settings → API.
2. Save it to a file named `roboflow_key.txt` inside `App/`, next to `GUI.py` (just the key,
   nothing else), set the `ROBOFLOW_API_KEY` environment variable, or (on Streamlit Community
   Cloud) add it under *App settings → Secrets*.
3. Restart the app (either edition). No key = each modality falls back to its offline model
   (Mammography: simulated, since no local weights are bundled).

## Project layout

```
README.md                 — this file
Start.bat / Start_Web.bat   — launchers: cd into App/, install deps, run
Apk/                       — optional: local Android build (Kivy + Buildozer), see Apk/README.md
RK3588 SBC/                — optional: RK3588 edge-node deployment code, see its own README.md

App/                     — everything else: the app, its models, docs, tests, and reference material
  GUI.py                    — Tkinter desktop app: UI + local SQLite cache + reports + fallbacks
  streamlit_app.py           — Streamlit web app (responsive) — same logic, imported from GUI.py
  inference.py                — model loading + prediction dispatch for all three modalities
  offline_cv.py                — the offline pixel-diff heuristic (no model, no internet, ever);
                                    run directly (`python offline_cv.py`) to see its measured accuracy
  train_local_model.py          — trains a real MobileNetV3 classifier per modality on this
                                    project's own dataset; run directly to (re)train + measure it
  requirements.txt            — Python dependencies (shared by both editions)
  pink_edge_cache.db           — local report cache (SQLite; created on first "Save to Cache")
  roboflow_key.txt             — your Roboflow key, if you added one (gitignored)
  .streamlit/config.toml       — theme config (Streamlit Cloud reads this relative to the app entrypoint)

  Models/                  — downloaded weight cache + local datasets, one folder per modality
    TB/model.pt                          — sukhmani1303/tuberculosis-vit-model (TorchScript)
    Maternal/FINAL-test-evaluation.pt    — shr3m/fetal-brain-plane-cnn
    Mammography/                         — populated only if you add a Roboflow key (see above)
    */local_model.pt, local_model_metadata.json — trained by train_local_model.py (gitignored,
                                              reproducible — re-run the script to regenerate)
    */Data Set/                          — local labeled datasets offline_cv.py builds templates from
    */positive/, */negative/             — optional: drop extra images in directly, no annotation
                                              needed — folded into the template AND next training run
    */validate/                          — optional: drop images in for a no-ground-truth spot-check
                                              (`python offline_cv.py` prints a prediction for each)
    */templates/                         — offline_cv.py's generated reference images (gitignored, auto-rebuilt)

  Validation/
    validate.py             — validation suite, run with `python Validation/validate.py` (from App/)

  Test Data/                — real sample images validate.py runs through the real models
    Tuberculosis/            — sample chest X-rays
    Breast Cancer/           — sample mammogram
    Maternal/                — sample ultrasound

  Documentations/           — reference docs
    MODEL_SOURCES.md         — exactly which model backs which modality, and why
    (+ the original project's own docs: API/BACKEND/FRONTEND/MODEL/PROJECT_ARCHITECTURE, USER_GUIDE)

  Hardware/                 — hardware plan for a real edge-node build (RK3588/Pi/ESP32/Mobile) —
                                see Hardware/README.md

  Assets/
    Changes/Changes.md       — what changed from the original Streamlit demo

  Misc/                     — the original hackathon submission this was built from
    Pink_Edge_AI-main/       — original Streamlit app (pink_edge.py), notebook, requirements.txt
```

## Validating a change

```
python Validation/validate.py
```
(run from inside `App/` — or `python App/Validation/validate.py` from the repo root; paths inside
the script are anchored to `App/`, not the caller's CWD, so both work)

21 checks covering: module imports, placeholder image synthesis, detection overlay drawing, the
simulated scenario generators, a full SQLite cache round-trip (on a throwaway DB under `Validation/`
— never touches the real `pink_edge_cache.db`), text/PDF report generation, real-model inference for
all three modalities on both synthetic images *and* the real samples in `Test Data/`, ground-truth
cross-checks against each modality's real COCO-annotated dataset (including an accuracy floor for
both the offline pixel-diff heuristic and the locally-trained classifier), the out-of-domain gate
(a wrong-type image is rejected, real scans still pass through), the `run_triage()`
dispatcher, and two full feature sweeps — every
modality, save-to-cache, report downloads, language toggle, network mode, cloud sync — for both the
**Tkinter UI** (hidden window, no mainloop) and the **Streamlit UI** (`streamlit.testing.v1.AppTest`,
no browser). Exits non-zero (and prints `inference.py`'s per-modality model status) if anything fails.

## Deploy the web edition to Streamlit Community Cloud

1. **Push to GitHub** (once git is installed):
   ```
   git init
   git add .
   git commit -m "Pink Edge AI desktop + Streamlit editions"
   ```
   Create an empty repo at github.com/new (no README/.gitignore/license — this repo already has
   them), then:
   ```
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git branch -M main
   git push -u origin main
   ```
2. **Deploy**: go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub →
   *New app* → pick your repo/branch, set **Main file path** to **`App/streamlit_app.py`** (not
   `streamlit_app.py` — it's inside `App/` now) → *Deploy*. Streamlit Cloud looks for
   `requirements.txt` next to the main file, so `App/requirements.txt` should be picked up
   automatically; this exact layout (code in a subfolder, not repo root) hasn't been re-verified
   on Streamlit Cloud since the `App/` move — if the deploy can't find requirements or the theme,
   check *Advanced settings* for a way to point at `App/requirements.txt` explicitly.
3. **Optional — real mammography model**: in the deploy dialog's *Advanced settings* (or later via
   *App settings → Secrets*), add:
   ```toml
   ROBOFLOW_API_KEY = "your-key-here"
   ```
   `inference.py` checks `st.secrets` for this automatically — no code changes needed.

**Resource caveat, honestly stated:** Streamlit Community Cloud's free tier gives each app ~1 CPU
core and ~1 GB RAM. This app's dependency stack (PyTorch, Ultralytics/OpenCV, two real model
checkpoints downloaded at first run) is heavier than a typical Streamlit demo — expect a slow first
boot (installing torch + downloading ~80 MB of weights) and keep an eye out for memory-related
crashes on that tier. If it struggles, the fixes in order of effort are: pin lighter dependency
versions, or deploy on a paid tier / your own server (`streamlit run streamlit_app.py --server.port
80 --server.address 0.0.0.0`, run from `App/`) instead.

## Optional: local Android APK and RK3588 edge-node deployment

Two standalone pieces at the repo root, neither needed to run the desktop/web app above:

- **`Apk/`** — a local Android build (Kivy + Buildozer, not a port of `GUI.py`/`streamlit_app.py` —
  see `Apk/README.md` for why torch/opencv don't cross-compile for Android and what runs instead).
- **`RK3588 SBC/`** — deployment code for running this app as a real edge node on an RK3588 board
  (kiosk autostart, the UART bridge to the ESP32/SIM800L GSM companion from `App/Hardware/WIRING.md`,
  and NPU-accelerated inference via `rknn-toolkit-lite2`) — see `RK3588 SBC/README.md`.

Neither has been built/run against real hardware in this environment (no Android SDK/NDK, no RK3588
board) — both READMEs say so plainly; treat them as a verified-on-real-hardware starting point.

## Relationship to the original project

`App/Misc/Pink_Edge_AI-main` is the original Streamlit/Alibaba-Cloud-Hackathon submission both
editions here are based on — same clinical vocabulary (BI-RADS, ACR density, TB severity/zone),
same SQLite schema, same report layout, same simulated cloud-sync panel (no real Alibaba
credentials are used here either). `streamlit_app.py` is a fresh, responsive rebuild (not the
original `pink_edge.py`) that reuses `GUI.py`'s shared logic and the real model backends instead of
the original's all-simulated scenario pickers. An Android build was discussed but deferred in favor
of these desktop/web builds.
