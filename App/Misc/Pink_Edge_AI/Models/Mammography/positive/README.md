# Models/Mammography/positive/

Drop extra mammogram images here that show **cancer** — no annotation file needed. `offline_cv.py`
picks these up automatically the next time it (re)builds its Mammography template, alongside the
labeled `Data Set/BreastCancer-YOLOv8.coco` images, and auto-detects when you've added/removed files
here (no manual cache-clear needed).

Accepted formats: `.jpg`, `.jpeg`, `.png`, `.bmp`. See `../negative/README.md` for the normal
counterpart and `../validate/README.md` for a no-annotation spot-check folder instead.
