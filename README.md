# Smart Parking System

Edge-based smart parking system using ACPDS quadrilateral pooling, `YOLOv8-cls`, FastAPI, and `Find My Car` product flows.
The canonical project definition is [docs/prd.md](docs/prd.md). This README summarizes the current `v6` direction.

---

## Architecture

Current pipeline:

```
parking-space quadrilaterals → perspective warp → YOLOv8-cls → temporal smoothing → JSON → FastAPI
```

Key points:

- Stage 1 uses parking-space quadrilaterals from `ACPDS` annotations or future `SfM` layout generation
- Stage 2 classifies each perspective-corrected `128×128` patch as `occupied` or `free`
- only compact occupancy JSON is sent to the backend; raw video stays on the edge device
- the broader product scope includes owner setup, occupancy map views, and `Find My Car`

---

## Team & Task Assignments

### @thebkht — ML lead

> Owns the critical path: dataset, patch extraction, training, and SfM layout AI.

**Week 5**

- [ ] Download ACPDS via `github.com/martin-marek/parking-space-occupancy`
- [ ] Write `ml/extract_patches.py` — `order_corners()` + `warpPerspective` to `acpds_stage2/`
- [ ] Run `validate_patches()` on 20 samples before training — confirm no twisted warps
- [ ] Train `YOLOv8n-cls` on `acpds_stage2/` — target ≥98% accuracy on unseen test split
- [ ] Implement SfM pipeline (COLMAP or OpenCV) → generate BEV map sample for owner setup
- [ ] **Deliver** `acpds_cls/weights/best.pt` + validated sample patches to @abdusattormv
- [ ] **Deliver** SfM script + BEV map image to @mirzayv

**Week 6**

- [ ] Train `YOLOv8s-cls` and `YOLOv8m-cls` on ACPDS
- [ ] Run INT8 quantization on `YOLOv8n-cls`
- [ ] Implement SIFT car localization (`cv2.SIFT_create()` + FLANN matcher)
- [ ] Write Dataset section of report

**Week 7**

- [ ] Analyze val accuracy vs test accuracy gap
- [ ] Per-weather accuracy breakdown (sunny / overcast / low-light subsets)
- [ ] Run pooling method (a) vs (b) comparison — bonus result replicating ACPDS Table 2
- [ ] Write Stage 1 and Layout AI sections of report

**Week 8**

- [ ] Finalize all accuracy tables and figures
- [ ] Present ACPDS justification, quad pooling, and ML pipeline in class

---

### @OtabekSadriddinov — ML / research

> Owns evaluation depth, model comparison, and literature context.

**Week 5**

- [ ] Write Related Work section — ACPDS paper (`arXiv:2107.12207`), PKLot, YOLOv8, SfM / visual localization

**Week 6**

- [ ] Build ResNet50 vs YOLOv8 comparison table (accuracy + parameter count + FPS)
- [ ] Run confidence threshold sweep on trained `YOLOv8n-cls`
- [ ] Test SIFT localization accuracy on 10+ sample ACPDS photos
- [ ] Write Stage 2 section of report — architecture, training config, training curves

**Week 7**

- [ ] Confusion matrix + PR curve for best model
- [ ] Full model comparison table (`n` / `s` / `m` / INT8 vs ResNet50 baseline)
- [ ] Localization accuracy table: top-1 / top-3 on 20+ real driver photos
- [ ] Write Find My Car and Evaluation sections of report

**Week 8**

- [ ] Write Discussion section
- [ ] Review full report for consistency across all sections
- [ ] Present Related Work and Stage 2 model findings in class

---

### @abdusattormv — Edge / backend

> Owns `detect.py`, FastAPI, all benchmarks, and the report's system sections.

**Week 5**

- [ ] Refactor `detect.py` to load quad polygon ROIs from `GET /map` at startup — no hardcoded `FIXED_ROIS`
- [ ] Add temporal smoothing (majority vote over N frames)
- [ ] Implement all 7 FastAPI endpoints + full SQLite schema (`log`, `layout`, `spot_references`, `park_sessions`)
- [ ] FPS benchmark: pre-trained classifier at `128×128` on MPS and CPU

**Week 6**

- [ ] Integrate `acpds_cls/weights/best.pt` from @thebkht into `detect.py`
- [ ] Full pipeline end-to-end: camera → Stage 1 → warp → Stage 2 → JSON → API
- [ ] FPS benchmark across all backends: MPS / CPU / ONNX FP32 / ONNX INT8
- [ ] Bandwidth measurement and comparison vs H.264 streaming
- [ ] Write System Architecture and Inference Pipeline sections of report

**Week 7**

- [ ] Final FPS + latency table (all backends)
- [ ] Bandwidth savings analysis (expected >99% vs raw video)
- [ ] System stability test: 30-minute continuous run with no crashes
- [ ] Write Edge Benchmarks section of report

**Week 8**

- [ ] Compile full report PDF — merge all sections
- [ ] Run live occupancy detection demo in class
- [ ] Present pipeline architecture and benchmark results

---

### @mirzayv — App / frontend

> Owns all three app screens, React Native mobile, and final report submission.

**Week 5**

- [ ] Scaffold React app (Vite + React): 3 screens with routing
- [ ] Build Leaflet.js map component: render 2D layout + quadrilateral polygon overlays per spot

**Week 6**

- [ ] Owner setup screen: photo upload → `POST /layout` → spinner → display BEV map with polygon overlays
- [ ] Live occupancy map screen: poll `GET /status` every 2–5 s → color spots green / red in real time
- [ ] Integrate BEV map image from @thebkht into the owner setup flow

**Week 7**

- [ ] Find My Car screen: camera → `POST /park` → store `session_id` → `GET /find/{id}` → amber spot highlight
- [ ] React Native (Expo) wrapper: native camera access for mobile demo
- [ ] Write App section of report — 3 screens, tech stack, Leaflet integration

**Week 8**

- [ ] Write Abstract, Conclusion, and References
- [ ] Submit technical report via email before deadline
- [ ] Run live Find My Car demo in class — present app screens

---

## Handoff Points

| When          | From               | To            | Deliverable                                            |
| ------------- | ------------------ | ------------- | ------------------------------------------------------ |
| End of Week 5 | @thebkht           | @abdusattormv | `acpds_cls/weights/best.pt` + validated sample patches |
| End of Week 5 | @thebkht           | @mirzayv      | SfM pipeline script + BEV map image                    |
| End of Week 6 | @OtabekSadriddinov | @thebkht      | Localization accuracy results (feeds Week 7 report)    |
| End of Week 7 | All                | @mirzayv      | All report sections → compile + submit                 |

---

## Repo Layout

- [`docs/prd.md`](docs/prd.md) — canonical PRD (v6)
- [`docs/prd-diagrams.md`](docs/prd-diagrams.md) — architecture and comparison diagrams
- [`edge/detect.py`](edge/detect.py) — edge inference pipeline
- [`backend/main.py`](backend/main.py) — FastAPI backend
- [`ml/`](ml) — training, evaluation, export, and benchmarking scripts
- [`samples/`](samples) — sample images and videos

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Cross-platform with `make`:

```bash
make install
```

Quick environment check:

```bash
python -c "import cv2, ultralytics, yaml; print('env ok')"
```

---

## Common Commands

Run the backend:

```bash
make backend
```

Run edge inference on an image:

```bash
make edge EDGE_ARGS="--image samples/photo_2026-04-23 21.29.16.jpeg"
```

Run live camera inference:

```bash
make edge EDGE_ARGS="--camera 0"
```

Train Stage 2 classifier variants:

```bash
make train-stage2 STAGE2_VARIANT=n
make train-stage2 STAGE2_VARIANT=s
make train-stage2 STAGE2_VARIANT=m
```

Evaluate a trained classifier:

```bash
make evaluate-stage2 STAGE2_VARIANT=n
```

Run tests:

```bash
make test
```

---

## Dataset Direction

The current PRD centers the project on `ACPDS`:

- 293 full parking-lot images captured at ~12 m height (GoPro on telescoping pole)
- 11,236 parking-space annotations as quadrilateral polygons
- unique parking lots for train / val / test splits — true generalization by design
- 48% occupied / 52% free — near-balanced, no class weighting needed
- MIT license — dataset, code, and pretrained weights

This replaces the older repo story that emphasized PKLot/CNRPark and fixed ROI demos.
When there is a mismatch between older docs and [`docs/prd.md`](docs/prd.md), follow the PRD.

---

## Payload Contract

```json
{
  "spots": {
    "spot_1": "free",
    "spot_2": "occupied"
  },
  "confidence": {
    "spot_1": 0.91,
    "spot_2": 0.84
  },
  "timestamp": "2026-04-21T00:00:00Z"
}
```

---

## Public Release Policy

- code, configs, metrics, and reproducible commands stay in the repo
- trained weights do not get committed into git history
- dataset archives, extracted datasets, runtime databases, and generated logs stay out of git

Model publishing guidance is in [`MODEL_LICENSE.md`](MODEL_LICENSE.md).

---

## Docs

- [`docs/README.md`](docs/README.md)
- [`docs/prd.md`](docs/prd.md)
- [`docs/prd-diagrams.md`](docs/prd-diagrams.md)
- [`edge/README.md`](edge/README.md)
- [`backend/README.md`](backend/README.md)
