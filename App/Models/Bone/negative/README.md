# Models/Bone/negative/

Drop extra bone X-ray images here that show a **fracture that is NOT a dislocation ("Crack")** — a
hairline crack, a comminuted/greenstick/spiral/oblique break, an avulsion, etc. — no annotation file
needed. `offline_cv.py` picks these up automatically the next time it (re)builds its Bone template,
alongside the labeled `Data Set/Bone-Fracture.coco` images.

Note this folder's "negative" means **Crack** (a non-dislocation fracture), not "negative = healthy
bone" — see `../positive/README.md` for why Bone reuses the positive/negative slots for a sub-type
distinction instead of disease-vs-healthy. A genuinely normal/healthy bone X-ray doesn't belong in
either folder — the source dataset (and this heuristic) has no way to represent "OK" at all; that
state comes only from the real-time detector tiers in `inference.py` finding no fracture region.

Accepted formats: `.jpg`, `.jpeg`, `.png`, `.bmp`. See `../positive/README.md` for the Shift
counterpart and `../validate/README.md` for a no-annotation spot-check folder instead.
