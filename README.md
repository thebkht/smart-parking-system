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

### [@thebkht](https://github.com/thebkht) — ML lead

> Owns the critical path: dataset, patch extraction, training, and SfM layout AI.

**Week 5**

- [x] Download ACPDS via `github.com/martin-marek/parking-space-occupancy`
- [x] Write `ml/extract_patches.py` — `order_corners()` + `warpPerspective` to `datasets/acpds_stage2/`
- [x] Run `validate_patches()` on 20 samples before training — save `validation_report.json` + `validation_samples/`
- [x] Train `YOLOv8n-cls` on `datasets/acpds_stage2/` with validation gate + promoted checkpoint at `acpds_cls/weights/best.pt`
- [x] Implement SfM pipeline (`ml/sfm_layout.py`) → generate `artifacts/layout_sample/bev_map.png` + `layout.json`
- [x] **Deliver** `acpds_cls/weights/best.pt` + validated sample patches + `map_sample.json` to [@abdusattormv](https://github.com/abdusattormv)
- [x] **Deliver** SfM script + BEV map image + layout JSON to [@mirzayv](https://github.com/mirzayv)

**Week 6**

- [x] Train `YOLOv8s-cls` and `YOLOv8m-cls` on ACPDS — best observed test accuracy was `0.8768` with `YOLOv8s-cls`, below the original `>=98%` target
- [x] Run INT8 quantization on `YOLOv8n-cls` — generated `artifacts/models/best_int8.onnx` and `artifacts/models/best.mlpackage`
- [x] Implement SIFT car localization (`cv2.SIFT_create()` + FLANN matcher)
- [x] Write Dataset section of report

**Week 7**

- [ ] Analyze val accuracy vs test accuracy gap
- [ ] Per-weather accuracy breakdown (sunny / overcast / low-light subsets)
- [ ] Run pooling method (a) vs (b) comparison — bonus result replicating ACPDS Table 2
- [ ] Write Stage 1 and Layout AI sections of report

**Week 8**

- [ ] Finalize all accuracy tables and figures
- [ ] Present ACPDS justification, quad pooling, and ML pipeline in class

---

### [@OtabekSadriddinov](https://github.com/OtabekSadriddinov) — ML / research

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

### [@abdusattormv](https://github.com/abdusattormv) — Edge / backend

> Owns `detect.py`, FastAPI, all benchmarks, and the report's system sections.

**Week 5**

- [ ] Refactor `detect.py` to load quad polygon ROIs from `GET /map` at startup — no hardcoded `FIXED_ROIS`
- [ ] Add temporal smoothing (majority vote over N frames)
- [ ] Implement all 7 FastAPI endpoints + full SQLite schema (`log`, `layout`, `spot_references`, `park_sessions`)
- [ ] FPS benchmark: pre-trained classifier at `128×128` on MPS and CPU

**Week 6**

- [ ] Integrate `acpds_cls/weights/best.pt` from [@thebkht](https://github.com/thebkht) into `detect.py`
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

### [@mirzayv](https://github.com/mirzayv) — App / frontend

> Owns all three app screens, React Native mobile, and final report submission.

**Week 5**

- [x] Scaffold React app (Vite + React): 3 screens with routing
- [x] Build Leaflet.js map component: render 2D layout + quadrilateral polygon overlays per spot

**Week 6**

- [x] Owner setup screen: photo upload → `POST /layout` → spinner → display BEV map with polygon overlays
- [x] Live occupancy map screen: poll `GET /status` every 2–5 s → color spots green / red in real time
- [x] Integrate BEV map image from [@thebkht](https://github.com/thebkht) into the owner setup flow

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

| When          | From                                                       | To                                               | Deliverable                                            |
| ------------- | ---------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------------ |
| End of Week 5 | [@thebkht](https://github.com/thebkht)                     | [@abdusattormv](https://github.com/abdusattormv) | `acpds_cls/weights/best.pt` + validated sample patches |
| End of Week 5 | [@thebkht](https://github.com/thebkht)                     | [@mirzayv](https://github.com/mirzayv)           | SfM pipeline script + BEV map image                    |
| End of Week 6 | [@OtabekSadriddinov](https://github.com/OtabekSadriddinov) | [@thebkht](https://github.com/thebkht)           | Localization accuracy results (feeds Week 7 report)    |
| End of Week 7 | All                                                        | [@mirzayv](https://github.com/mirzayv)           | All report sections → compile + submit                 |

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

Prepare, validate, and train the Week 5 ACPDS classifier:

```bash
make prepare-stage2 ACPDS_ROOT=/path/to/acpds PREP_STAGE2_ARGS="--run-validation --validation-status passed"
make validate-stage2 ACPDS_ROOT=/path/to/acpds VALIDATE_STAGE2_ARGS="--validation-status passed"
make train-stage2 STAGE2_VARIANT=n
```

Evaluate a trained classifier:

```bash
make evaluate-stage2
```

Generate a Week 5 BEV layout sample for App handoff:

```bash
make layout-sample
```

Run tests:

```bash
make test
```

---

## Week 5 Process

Use this sequence for `[@thebkht](https://github.com/thebkht)`'s ACPDS milestone.

1. Extract ACPDS patches into `datasets/acpds_stage2/`.
2. Review 20 validation samples and mark the validation report as `passed`.
3. Train `YOLOv8n-cls` and promote the selected checkpoint to `acpds_cls/weights/best.pt`.
4. Evaluate the promoted checkpoint on the ACPDS split.
5. Generate the BEV sample package for App.

Recommended commands:

```bash
make prepare-stage2 ACPDS_ROOT=/path/to/acpds PREP_STAGE2_ARGS="--run-validation"
make validate-stage2 ACPDS_ROOT=/path/to/acpds VALIDATE_STAGE2_ARGS="--validation-status passed"
make train-stage2 STAGE2_VARIANT=n
make evaluate-stage2
make layout-sample
```

Direct CLI equivalents:

```bash
python ml/extract_patches.py --dataset-root /path/to/acpds --output datasets/acpds_stage2 --run-validation
python ml/extract_patches.py --dataset-root /path/to/acpds --output datasets/acpds_stage2 --validate-only --validation-status passed
python ml/train.py --stage2 --variant n --data datasets/acpds_stage2 --device mps
python ml/evaluate.py --stage2 --weights acpds_cls/weights/best.pt --data datasets/acpds_stage2 --split test --device mps
python ml/sfm_layout.py --images samples --output artifacts/layout_sample
```

Expected outputs:

- `datasets/acpds_stage2/dataset_report.json`
- `datasets/acpds_stage2/patch_index.json`
- `datasets/acpds_stage2/validation_report.json`
- `datasets/acpds_stage2/validation_samples/`
- `datasets/acpds_stage2/map_sample.json`
- `acpds_cls/weights/best.pt`
- `artifacts/layout_sample/bev_map.png`
- `artifacts/layout_sample/layout.json`

Week 5 handoff package:

- Edge: `acpds_cls/weights/best.pt`, `datasets/acpds_stage2/validation_samples/`, `datasets/acpds_stage2/validation_report.json`, `datasets/acpds_stage2/map_sample.json`
- App: `ml/sfm_layout.py`, `artifacts/layout_sample/bev_map.png`, `artifacts/layout_sample/layout.json`

Notes:

- Stage 2 training is gated by `datasets/acpds_stage2/validation_report.json` with `status: "passed"`.
- `acpds_cls/weights/best.pt` is promoted automatically only for the Week 5 `YOLOv8n-cls` handoff path; `s` and `m` stay as comparison runs unless `--promote-stage2` is passed explicitly.
- The ACPDS split from the manifest is authoritative; this workflow does not rebuild train/val/test randomly.

## Week 6 Process

Use this sequence for the Week 6 Stage 2 comparison and export milestone.

1. Run the `s` and `m` comparison workflow against the existing ACPDS Stage 2 dataset.
2. Review `val` and `test` outputs for `n`, `s`, and `m`; do not overwrite `acpds_cls/weights/best.pt`.
3. Export the promoted Week 5 `n` checkpoint into ONNX FP32, ONNX INT8, and Core ML INT8 artifacts.
4. Evaluate the exported ONNX artifact on the ACPDS `val` and `test` splits.
5. Use the combined results to update the report and decide whether the bottleneck is model capacity or patch quality.

Recommended commands:

```bash
make week6-stage2 DEVICE=mps
make week6-export
```

Direct CLI equivalents:

```bash
python ml/train.py --stage2 --variant s --device mps
python ml/evaluate.py --stage2 --weights runs/acpds_cls/yolov8s_stage2/weights/best.pt --split val --device mps --output-json logs/week6/stage2_s_val.json
python ml/evaluate.py --stage2 --weights runs/acpds_cls/yolov8s_stage2/weights/best.pt --split test --device mps --output-json logs/week6/stage2_s_test.json
python ml/train.py --stage2 --variant m --device mps
python ml/evaluate.py --stage2 --weights runs/acpds_cls/yolov8m_stage2/weights/best.pt --split val --device mps --output-json logs/week6/stage2_m_val.json
python ml/evaluate.py --stage2 --weights runs/acpds_cls/yolov8m_stage2/weights/best.pt --split test --device mps --output-json logs/week6/stage2_m_test.json
python ml/evaluate.py --stage2 --split val --device mps --output-json logs/week6/stage2_compare_val.json --compare acpds_cls/weights/best.pt runs/acpds_cls/yolov8s_stage2/weights/best.pt runs/acpds_cls/yolov8m_stage2/weights/best.pt
python ml/evaluate.py --stage2 --split test --device mps --output-json logs/week6/stage2_compare_test.json --compare acpds_cls/weights/best.pt runs/acpds_cls/yolov8s_stage2/weights/best.pt runs/acpds_cls/yolov8m_stage2/weights/best.pt
python ml/export.py --weights acpds_cls/weights/best.pt --imgsz 128 --summary-json artifacts/models/export_summary.json
python ml/evaluate.py --stage2 --weights artifacts/models/best.onnx --split val --device cpu --output-json logs/week6/stage2_export_onnx_val.json
python ml/evaluate.py --stage2 --weights artifacts/models/best.onnx --split test --device cpu --output-json logs/week6/stage2_export_onnx_test.json
```

Expected outputs:

- `runs/acpds_cls/yolov8s_stage2/weights/best.pt`
- `runs/acpds_cls/yolov8m_stage2/weights/best.pt`
- `models/stage2_s_report.json`
- `models/stage2_m_report.json`
- `logs/week6/stage2_s_val.json`
- `logs/week6/stage2_s_test.json`
- `logs/week6/stage2_m_val.json`
- `logs/week6/stage2_m_test.json`
- `logs/week6/stage2_compare_val.json`
- `logs/week6/stage2_compare_test.json`
- `artifacts/models/best.pt`
- `artifacts/models/best.onnx`
- `artifacts/models/best_int8.onnx`
- `artifacts/models/best.mlpackage`
- `artifacts/models/export_summary.json`
- `logs/week6/stage2_export_onnx_val.json`
- `logs/week6/stage2_export_onnx_test.json`

Observed Week 6 comparison result:

- `yolov8n-cls`: test accuracy `0.8678`, F1 `0.8110`, size `2.83 MB`
- `yolov8s-cls`: test accuracy `0.8768`, F1 `0.8222`, size `9.78 MB`
- `yolov8m-cls`: test accuracy `0.8742`, F1 `0.8225`, size `30.22 MB`

This means Week 6 improved comparison coverage, but not the original `>=98%` accuracy target. The current evidence points more toward Stage 2 patch/data quality as the bottleneck than raw model size.

Notes:

- `acpds_cls/weights/best.pt` remains the Week 5 handoff checkpoint unless `--promote-stage2` is passed explicitly.
- Exported classifier artifacts are evaluated with batch size `1` in `ml/evaluate.py`.
- Exported ONNX weights must be loaded with `task="classify"` during evaluation to avoid detection-style NMS postprocessing.

Run SIFT localization against either a manifest JSON or a per-spot reference directory:

```bash
make localize-car LOCALIZE_ARGS="--query samples/query.jpg --references samples/localization_refs --output logs/localize_result.json"
```

Supported reference layouts:

```text
samples/localization_refs/
  spot_1/
    a.jpg
    b.jpg
  spot_2/
    a.jpg
```

or:

```json
{
  "spot_1": ["refs/spot_1/a.jpg", "refs/spot_1/b.jpg"],
  "spot_2": "refs/spot_2/a.jpg"
}
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
