# Models/Bone/positive/

Drop extra bone X-ray images here that show a **dislocation / displaced fracture ("Shift")** — no
annotation file needed. `offline_cv.py` picks these up automatically the next time it (re)builds its
Bone template, alongside the labeled `Data Set/Bone-Fracture.coco` images, and auto-detects when
you've added/removed files here (no manual cache-clear needed).

Note this folder's "positive" means **Shift**, not "positive = presence of any fracture" — Bone is
the one modality where the offline heuristic and locally-trained classifier distinguish a *sub-type*
(Shift vs. Crack) rather than disease-vs-healthy, because the source dataset has zero normal/healthy
bone X-rays to build an OK reference from (see the "Bone" section of `Documentations/MODEL_SOURCES.md`).
Presence-of-fracture-at-all (OK vs. not) is determined separately, by the real-time Roboflow/Hugging
Face detector tiers in `inference.py`.

Accepted formats: `.jpg`, `.jpeg`, `.png`, `.bmp`. See `../negative/README.md` for the Crack
counterpart and `../validate/README.md` for a no-annotation spot-check folder instead.
