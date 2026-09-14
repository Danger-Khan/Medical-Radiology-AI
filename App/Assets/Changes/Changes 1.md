# Changes: Pink Edge AI → Medical Radiology AI

This file picks up where `Changes.md` (the "Pink Edge AI" era) leaves off. That file documents
everything that changed building Pink Edge AI from the original hackathon submission; this one
documents the rebuild into **Medical Radiology AI** — a new identity, theme, and layout on the
same proven triage engine.

## Why this rebuild happened

Requested directly: rebrand the app from Pink Edge AI to Medical Radiology AI, with a genuinely
distinct identity (not a find-and-replace job), a light blue / light gray visual theme in place of
the earlier dark navy/teal/pink one, and — done first, ahead of the rest — a reworked UI layout
(control menu moved to the right, more breathing room throughout, a couple of new options). A
fourth "Bone" modality (fracture/dislocation detection) was also requested but is **not** part of
this pass — deferred per "first of all work on ui," to be picked up next.

## Archive: the complete Pink Edge AI build preserved in Misc/

Per explicit instruction, everything that made up the Pink Edge AI build — not just its code, but
its datasets, trained-model cache config, validation suite, docs, and hardware plan — was moved
into `App/Misc/Pink_Edge_AI/` as a full, untouched snapshot, sibling to the already-archived
`App/Misc/Pink_Edge_AI-main/` (the original hackathon submission Pink Edge AI itself was built
from). Moved: `GUI.py`, `streamlit_app.py`, `inference.py`, `offline_cv.py`,
`train_local_model.py`, `requirements.txt`, `pink_edge_cache.db`, `.python-version`,
`roboflow_key.txt`, `Models/`, `Test Data/`, `Validation/`, `Documentations/`, `Hardware/`,
`Assets/`. Nothing in the archive was altered — it's the exact working state Pink Edge AI was in
before this rebuild started.

The new build then got **working copies** of the reusable infrastructure (`Models/`, `Test Data/`,
`Validation/`, `Hardware/`, `requirements.txt`, `.python-version`, `roboflow_key.txt`) copied back
out of the archive into fresh `App/` locations — datasets and trained-model configuration aren't
"Pink Edge" branding, they're just data, so re-downloading or re-sourcing them from scratch would
have been pure waste. This build has no runtime dependency on `Misc/` staying in place.

## New identity throughout

| Old (Pink Edge AI, archived) | New (Medical Radiology AI, active) |
|---|---|
| `GUI.py` | `radiology_console.py` |
| `streamlit_app.py` | `radiology_web.py` |
| `PinkEdgeApp` (class) | `RadiologyConsoleApp` |
| `pink_edge_cache.db` | `radiology_cache.db` |
| "Pink Edge AI" (title, headers, PDF/text report headers) | "Medical Radiology AI" |
| 🩸 (blood-drop icon) | 🩻 (X-ray icon) |

`inference.py`, `offline_cv.py`, and `train_local_model.py` kept their file names — they're
generic/functional names, not brand-specific — but had their "Pink Edge AI" docstring headers
updated. `Validation/validate.py` was updated throughout: its module docstring, print banner, and
content assertions now reference the new names; it still imports the desktop module under the
`GUI` alias internally (`import radiology_console as GUI`) so the ~50 `GUI.xxx` call sites
elsewhere in the file needed no individual changes — only the import line, the `PinkEdgeApp` class
reference, and the `streamlit_app.py` → `radiology_web.py` `AppTest.from_file()` path needed
touching. `Start.bat` / `Start_Web.bat` and the root `README.md` were rewritten for the new file
names and layout. `.gitignore` gained `radiology_cache.db` alongside the still-present
`pink_edge_cache.db` pattern (the archived copy still needs it ignored).

## New color theme: light blue / light gray

Replaced the earlier dark navy/teal/pink palette (`radiology_console.py`'s `C` dict, reused
verbatim by `radiology_web.py`'s injected CSS) with a bright clinical-console look:

```
bg:            #eef2f7   (page background)      was #0a0e1a
surface:       #ffffff   (cards)                 was #111827
surface_alt:   #f1f5f9   (nested panels)         was #1e293b
text:          #1e293b   (main text)             was #f1f5f9
primary:       #2f6fed   (blue, was teal)        was #14b8a6
accent:        #0ea5e9   (sky blue, was pink)    was #ec4899
console_bg/fg: #e7ecf3 / #334155  (new tokens — a light "terminal" panel for the telemetry/
               hardware-diagnostics readouts, replacing a hardcoded near-black #060a13)
```

`success` / `warning` / `danger` were left as their original semantic traffic-light values —
they're universal, not brand colors. Every place that previously hardcoded the old dark console
color (`#060a13`) now uses the new `console_bg`/`console_fg` tokens instead, so there's no
leftover dark panel anywhere in the light theme.

## Layout: menu moved to the right, more relaxed spacing, new options

Requested changes, done as a deliberately visible restructuring rather than a cosmetic tweak:

- **Desktop (`radiology_console.py`)**: added a full-width top header band (branding + tagline)
  that didn't exist before; the control sidebar now packs on the **right** (`side="right"`) instead
  of the left, with the main notebook content taking the left. Padding was widened throughout
  (14px → 18px in the sidebar, 14px → 18px in the metadata/verdict panel, more `pady` between
  sections) for a less cramped, more "relaxed" feel. The dashboard's image card grew a visible
  border/card treatment, metric tiles got more internal padding, and the redundant in-tab header
  (which repeated the app name already shown in the new top band) was replaced with a plainer
  "Medical Image Analysis" section heading.
- **Web (`radiology_web.py`)**: Streamlit has no official right-sidebar API, so the menu move is a
  CSS hack — `div[data-testid="stAppViewContainer"] { flex-direction: row-reverse; }` — flagged in
  a code comment as depending on Streamlit's current internal DOM structure, since it isn't a
  documented API and could need revisiting on a future Streamlit upgrade. Card/verdict-box/
  metric-tile border-radius increased (10–14px → 14–20px) and padding widened to match the
  desktop's relaxed spacing.
- **New sidebar options** (both editions): a **Clinician Notes** free-text field — its content, if
  non-empty, now flows into both the text and PDF reports as a "CLINICIAN NOTES" section (threaded
  through `_current_report_dict()`/`_save_cache()`'s data dict on desktop and the equivalent
  `report_data`/`save_cache_action()` dict on web, then read by `generate_text_report()`/
  `generate_pdf_bytes()`) — and an **About** panel with a short app description and a pointer to
  `Documentations/MODEL_SOURCES.md`.

Verified after every change: `python Validation/validate.py` — **21/21** — and both `Start.bat`
and `Start_Web.bat` actually launched (window title confirmed as "Medical Radiology AI — Clinical
Intelligence Platform (Desktop)"; the Streamlit server confirmed serving HTTP 200).

## Not done in this pass

- The original hackathon-era reference docs carried into `Documentations/` (`API_DOCUMENTATION.md`,
  `BACKEND_DOCUMENTATION (2).md`, `FRONTEND_DOCUMENTATION (1).md`, `MODEL_DOCUMENTATION.md`,
  `PROJECT_ARCHITECTURE.md`, `USER_GUIDE.md`) were left as-is except where they actively described
  the current build (`MODEL_SOURCES.md`, `Documentations/README.md`'s title/overview) — they're
  historical/reference material describing the original hackathon pitch, not "UI."
- `Apk/` and `RK3588 SBC/` (repo root) still reference "Pink Edge AI" in places — not touched this
  pass, since neither is part of the desktop/web UI this request was about.

---

## Batch 2: Bone X-Ray modality (replaces Maternal Health) + a stray doc fix

Requested directly: swap the third modality from Maternal Health to **Bone X-Ray** (fracture /
dislocation triage), and fix the one remaining stale "main file" reference left over from the
previous pass (`Documentations/README.md`'s install steps still named the *original* hackathon's
single-file app and a dead clone URL — not `radiology_console.py`/`radiology_web.py`).

### Doc fix: `Documentations/README.md`
Its "Installation" section still said `git clone .../Zobia-Irshad/Pink_Edge_AI.git`, `cd
pink-edge-ai`, and `streamlit run pink_edge.py` — all three predate even the "Pink Edge AI"
desktop/web build (they're artifacts of the *original* single-file hackathon submission). Fixed to
point at this repo (`Danger-Khan/Medical-Radiology-AI`) and `streamlit run radiology_web.py` /
`python radiology_console.py`. Its modality list, "Multi-Modal" feature bullet, and "Solution"
section bullet were updated from Maternal Health to Bone X-Ray alongside the modality swap below.

### Bone X-Ray: what backs it, honestly

Bone is structured differently from the other three modalities because no single public model
answers "is this OK, a Crack, or a Shift" in one shot — the design and every number below came from
actually probing real Roboflow projects and Hugging Face models (via the same API key already
configured), not from assuming a suitable dataset existed:

- **Presence of a fracture at all** (OK vs. not) — real-time: Roboflow's public
  `yakin/bone-fracture-tn84w` project (workspace `yakin`, not this project's own workspace — same
  pattern as Mammography's `b-davmu/breastcancer-yolov8` placeholder), specifically **version 1**,
  the only one of its 3 versions with an actually trained/deployed model (grounded via a real API
  call to its training summary: precision 86.5%, recall 71.1%, mAP@50 77.3%). Offline fallback:
  `prithivMLmods/Bone-Fracture-Detection` (Hugging Face, SigLIP2, Apache-2.0, ~83% accuracy per its
  model card) — added a `transformers` dependency to `requirements.txt` specifically for this, since
  TB/Mammography's HF integrations were deliberately hand-built to avoid it.
- **Sub-type, Crack vs. Shift** (only once "fracture" is confirmed) — the locally-trained classifier,
  then the offline pixel-diff heuristic, both trained on a **different, richer** export of the same
  dataset (version 3 — 2147 images, real per-type category annotations including `Dislocation`,
  CC BY 4.0), downloaded and placed at `Models/Bone/Data Set/Bone-Fracture.coco/` — this version has
  no Roboflow-hosted model of its own, but `offline_cv.py`/`train_local_model.py` don't need one,
  they train directly on the raw annotated images. `offline_cv.py`'s positive/negative template
  slots are reused for Shift/Crack here instead of disease/healthy — documented explicitly in
  `Models/Bone/positive|negative/README.md` and `Documentations/MODEL_SOURCES.md` so it isn't a
  silent semantic swap.
- **Measured, and honestly weak**: the sub-typer is barely better than chance (offline heuristic
  56%, local classifier 58.9%, both on real held-out data) — visually telling a dislocation from
  another fracture type by pixel pattern alone turned out to be genuinely hard with this method.
  The presence detector fares worse than its official numbers suggest when checked against this
  project's own re-annotated ground truth (2/16 known-Dislocation samples actually detected) — a
  real train/test mismatch between the deployed v1 model and the v3 annotations used to grade it,
  not a bug. Both are written up under new "⚠️ Known issue" / accuracy-table entries in
  `MODEL_SOURCES.md`, the same honesty convention already used for TB's Roboflow bias — **Bone is
  currently the weakest-measured modality in this app**, and that's stated plainly rather than
  glossed over.
- Out-of-domain gating for Bone is similarly weak and documented as such: real bone X-rays vary too
  much in framing (wrist vs. skull vs. shoulder) for the generic pixel-correlation reference to
  separate them from other radiograph types — threshold set low (like Mammography) to protect real
  scans (100% pass) rather than overstate its ~7% wrong-upload catch rate.

### Everywhere the swap touched
`inference.py` (new `predict_bone()`/`load_bone_model()`/`BONE_STATUS_OPTIONS`/
`BONE_TYPE_OPTIONS`, `predict_maternal()` and its Hugging Face CNN loader removed entirely),
`offline_cv.py` (`_DATASETS["bone"]`, a bone-specific `_to_bone_result()`, a new domain threshold),
`radiology_console.py` and `radiology_web.py` (model list, simulated scenarios, placeholder image
generator, bounding-box overlay, the "Confirm Assessment" combo-box pair, every `mod_map`/About-text
mention), `Validation/validate.py` (every Maternal-specific check rewritten for Bone, including a
new ground-truth cross-check calibrated against the measured 2/16 real hit rate so it's a genuine
regression guard rather than a flaky or rubber-stamped assertion), `train_local_model.py` needed
**no changes at all** — it's already generic over `offline_cv._DATASETS`. `Models/Maternal/` (its
dataset, positive/negative/validate folders) is left in place, now unreferenced by any code —
already fully preserved, unedited, at `App/Misc/Pink_Edge_AI/Models/Maternal/` from the earlier
archive move, so nothing here risks losing it; removing the now-dead active copy wasn't requested,
so it was left rather than guessed at.

Verified: `python Validation/validate.py` — **21/21** (this actually caught a real bug the first time
through: the Bone domain-gate threshold was initially set to protect 100% of real scans, which
silently let random noise pass too — see `offline_cv.py`'s `DOMAIN_THRESHOLDS["bone"]` comment for
the fix and why no threshold can fully solve both at once for this modality) — and both `Start.bat`
and `Start_Web.bat` confirmed launching live with "Bone X-Ray (Fracture/Dislocation)" selectable in
place of Maternal Health (window title and HTTP 200 both confirmed, same as every prior pass).

---

## Batch 3: real downloaded icons in the desktop GUI (+ an attempted, blocked root folder rename)

### Icons

Requested directly: stop relying on emoji glyphs and use real icon files in "the main GUI"
(`radiology_console.py`, the Tkinter desktop console — the literal `GUI.py` descendant; the
Streamlit web edition renders emoji fine already, via the browser, so it wasn't in scope here).

Downloaded [Twemoji](https://github.com/twitter/twemoji) PNGs (CC-BY 4.0, attribution in
`Assets/Icons/README.md`) matching every emoji glyph already used in the desktop console — 🩻 🩺 🏥
☁️ ⚙️ ℹ️ ✅ ⚠️ 🚫 📋 📝 📄 — to `App/Assets/Icons/`, plus a locally-generated (PIL, not downloaded)
multi-resolution `app_icon.ico` built from the 🩻 icon for the Windows title-bar/taskbar icon (the
window previously used Tk's default feather icon).

Wired into `radiology_console.py` via a new `_load_icon(name, size)` helper (loads + resizes +
caches a `PhotoImage`, since Tk silently drops an image with no persistent reference) and a
`_set_verdict_icon()` helper for the one place an icon changes dynamically per triage result (the
big verdict icon — success/warning/invalid/awaiting): the app window/taskbar icon, the topbar
brand mark, all 3 notebook tab icons, the sidebar's "Console Menu"/"About" header icons, the
"Sync to Cloud" button, the "Text Report"/"PDF Report" buttons, and the Hospital Hub/Cloud Sync tab
header bars. Every call site keeps its original emoji text as a fallback if the icon file is
missing, so a partial/deleted `Assets/Icons/` folder degrades gracefully rather than breaking
anything. Confirmed visually via a live screenshot, not just "it compiles" — real colorful icons
render correctly in the title bar, topbar, tabs, buttons, and verdict panel.

Not converted to icons: inline emoji inside longer sentences (the risk banner, hospital-hub alert
list, cache-sync tree checkmark/hourglass column) — those are single glyphs inside a bigger string,
not a standalone label/button, so swapping them to images would need a bigger per-string widget
refactor for comparatively little visual payback; left as text for this pass.

### Root folder rename — attempted, blocked, not done

Also requested: rename the project's root folder from `Pink Edge AI` to `Medical Radiology AI`
(confirmed with the user first, since it's disruptive to any open editor/terminal). Checked first for
anything that would break from a pure folder rename (`.devcontainer/`, `.vscode/`) — nothing
hardcodes the outer folder name, so the rename itself should have been safe content-wise. Attempted
via `Rename-Item` from the parent directory (not from inside the target, to avoid a self-referential
cwd lock) after confirming no stray app processes were running — both attempts failed with "The
process cannot access the file because it is being used by another process." The lock is VS Code's
own file watcher on this exact open workspace folder — the very editor this session runs inside as
an extension, so forcibly closing it to clear the lock isn't something this session can safely do to
itself. **Not done this pass** — closing/reloading the folder in VS Code first (or renaming it from
outside VS Code entirely, then reopening VS Code at the new path) would clear the lock; flagged back
to the user rather than worked around riskily.
