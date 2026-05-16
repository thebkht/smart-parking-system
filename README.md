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

## ML Status

Stage 2 classifier has reached production quality. **No further ML training is planned.**

| Model          | Split    | Accuracy   | Precision | Recall | F1     | Size (MB) |
| -------------- | -------- | ---------- | --------- | ------ | ------ | --------- |
| yolov8n_stage2 | Val      | 0.9827     | 0.9784    | 0.9819 | 0.9802 | 2.83      |
| yolov8s_stage2 | Val      | 0.9816     | 0.9727    | 0.9856 | 0.9791 | 9.78      |
| yolov8m_stage2 | Val      | 0.9795     | 0.9670    | 0.9868 | 0.9768 | 30.22     |
| yolov8n_stage2 | **Test** | **0.9772** | 0.9864    | 0.9570 | 0.9715 | **2.83**  |
| yolov8s_stage2 | Test     | 0.9691     | 0.9745    | 0.9488 | 0.9615 | 9.78      |
| yolov8m_stage2 | Test     | 0.9738     | 0.9846    | 0.9504 | 0.9672 | 30.22     |

`yolov8n-cls` (2.83 MB) is the promoted checkpoint. It beats both larger variants on test accuracy while being 9× smaller than the ResNet50 paper baseline (25.6M params). Remaining work is evaluation depth and report writing — not retraining.

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

- [x] Train `YOLOv8s-cls` and `YOLOv8m-cls` on ACPDS — the promoted `YOLOv8n-cls` checkpoint still leads the checked-in Week 6 test comparison at `0.9772`, with `s=0.9691` and `m=0.9738`
- [x] Run INT8 quantization on `YOLOv8n-cls` — generated `artifacts/models/best_int8.onnx` and `artifacts/models/best.mlpackage`
- [x] Implement SIFT car localization (`cv2.SIFT_create()` + FLANN matcher)
- [x] Write Dataset section of report

**Week 7**

- [x] Val vs test accuracy gap analysis — `logs/week7/val_test_gap.json` documents the ~0.5–1.25pp generalization delta across n/s/m and ties it to unique-lot distribution shift rather than classic overfitting
- [x] Per-weather accuracy breakdown — ACPDS test split is now bucketed into sunny / overcast / low-light luminance tertiles with results saved to `logs/week7/stage2_acpds_weather.json`
- [x] Pooling method (a) vs (b) comparison — `logs/week7/pooling_comparison.json` shows quad warps at `0.9772` test accuracy versus `0.9638` for bounding-square pooling (`-1.34 pp`) on the same YOLOv8n Stage 2 setup
- [x] **Fix edge runtime quad warp** — `edge/detect.py` now preserves polygons end-to-end and classifies `warpPerspective(128×128)` patches; visual QA samples are saved under `logs/week7/warp_comparison/`
- [x] Write Stage 1 and Layout AI sections of technical report

**Week 8**

- [ ] Finalize all accuracy tables and figures (fill in model comparison table with test results)
- [ ] Present ACPDS justification, quad pooling, and ML pipeline in class

---

### [@OtabekSadriddinov](https://github.com/OtabekSadriddinov) — ML / research

> Owns evaluation depth, model comparison, and literature context.

**Week 5**

- [x] Write Related Work section — ACPDS paper (`arXiv:2107.12207`), PKLot, YOLOv8, SfM / visual localization

**Week 6**

- [x] Build ResNet50 vs YOLOv8 comparison table (accuracy + parameter count + FPS)
- [x] Run confidence threshold sweep on trained `YOLOv8n-cls`
- [x] Test SIFT localization accuracy on 10+ sample ACPDS photos
- [x] Write Stage 2 section of report — architecture, training config, training curves

**Week 7**

- [ ] Confusion matrix + PR curve for `yolov8n_stage2` on the test split — expected result: high precision (98.6%), lower recall (95.7%), so the matrix will show more occupied→free misses than false alarms; explain this in terms of patch quality (partial vehicles, border regions after warp)
- [ ] Full model comparison table — fill in `n` / `s` / `m` / INT8 vs ResNet50 paper baseline with accuracy, F1, parameter count, model size, and FPS; include the finding that larger variants do not improve test accuracy
- [ ] Confidence threshold sweep results — plot precision/recall tradeoff for `yolov8n-cls` at thresholds 0.3–0.9; identify optimal operating point
- [ ] Localization accuracy table — expand beyond the current 1/1 sample; collect top-1 and top-3 accuracy on 20+ real driver photos across varying lighting conditions
- [ ] Write Find My Car and Evaluation sections of technical report

**Week 8**

- [ ] Write Discussion section — cover what worked (n beats larger models, quad warp beats rect crop), limitations (patch quality bottleneck, label ambiguity, localization sample size), and production considerations
- [ ] Review full report for consistency across all sections before [@mirzayv](https://github.com/mirzayv) compiles
- [ ] Present Related Work and Stage 2 model findings in class

---

### [@abdusattormv](https://github.com/abdusattormv) — Edge / backend

> Owns `detect.py`, FastAPI, all benchmarks, and the report's system sections.

**Week 5**

- [x] Refactor `detect.py` to load quad polygon ROIs from `GET /map` at startup — no hardcoded `FIXED_ROIS`
- [x] Add temporal smoothing (majority vote over N frames)
- [x] Implement all 7 FastAPI endpoints + full SQLite schema (`log`, `layout`, `spot_references`, `park_sessions`)
- [x] FPS benchmark: pre-trained classifier at `128×128` on MPS and CPU

**Week 6**

- [x] Integrate `acpds_cls/weights/best.pt` from [@thebkht](https://github.com/thebkht) into `detect.py`
- [x] Full pipeline end-to-end: camera → Stage 1 → warp → Stage 2 → JSON → API
- [x] FPS benchmark across all backends: MPS / CPU / ONNX FP32 / ONNX INT8
- [x] Bandwidth measurement and comparison vs H.264 streaming
- [x] Write System Architecture and Inference Pipeline sections of report

**Week 7**

- [ ] **Add `POST /park` endpoint** — accept a driver photo, call `ml/localize.py` SIFT matching against stored `spot_references`, insert a row into `park_sessions` table with `spot_id` + `similarity_score`, return `session_id`
- [ ] **Add `GET /find/{session_id}` endpoint** — look up session in `park_sessions`, return `spot_id` + corner coordinates from the `layout` table; return 404 if session not found
- [ ] **Resolve `POST /layout` vs `POST /map` naming** — pick one canonical name, update the route in `backend/main.py`, notify @mirzayv so frontend fetch path matches, document final contract in `backend/README.md`
- [ ] **Fix `GET /status` response shape** — currently returns `{ spots, confidence, timestamp }`; confirm this is the final shape and document it in `backend/README.md` so @mirzayv can update the frontend parser to read `response.spots`
- [ ] Final FPS + latency table (all backends: MPS / CPU / ONNX FP32 / ONNX INT8)
- [ ] Bandwidth savings analysis — expected >99% vs raw H.264; use the measurement script from PRD §8.3 and include actual measured numbers
- [ ] System stability test — 30-minute continuous run with no crashes; log CPU usage, memory, and FPS stability; save output to `logs/stability_test.json`
- [ ] Write Edge Benchmarks section of technical report

**Week 8**

- [ ] Compile full report PDF — collect all sections from all members and merge into final document
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

- [ ] **Fix `GET /status` response parser** — backend returns `{ spots, confidence, timestamp }`; update frontend to read `response.spots` before coloring polygons and updating free/occupied count in the header
- [ ] **Fix `POST /layout` call** — align to the canonical name once @abdusattormv resolves the contract, then update the fetch path and `backend/README.md`
- [ ] **Remove mock fallbacks from Find My Car** — replace fake `session_id` generation and random spot fallback with the real `POST /park` → store `session_id` → `GET /find/{session_id}` flow once [@abdusattormv](https://github.com/abdusattormv) ships the endpoints
- [ ] **Wire Find My Car end-to-end** — camera capture → `POST /park` with photo → store `session_id` in local state → `GET /find/{session_id}` → highlight the returned spot polygon in amber on the Leaflet map
- [x] **Switch map rendering to Leaflet** — current UI uses custom SVG; `react-router-dom` and `leaflet` are installed but not used; migrate the live occupancy map and Find My Car screens to actual Leaflet polygon overlays with per-spot color updates
- [ ] React Native (Expo) wrapper — native camera access for mobile demo; if timeline is at risk, decide by end of Week 7 and formally drop from deliverables/docs if not feasible
- [ ] Write App section of technical report — 3 screens, tech stack, Leaflet integration, Find My Car flow

**Week 8**

- [ ] Write Abstract, Conclusion, and References
- [ ] Submit technical report via email before deadline
- [ ] Run live Find My Car demo in class — present all 3 app screens

---

## Handoff Points

| When          | From                                                       | To                                               | Deliverable                                                          |
| ------------- | ---------------------------------------------------------- | ------------------------------------------------ | -------------------------------------------------------------------- |
| End of Week 5 | [@thebkht](https://github.com/thebkht)                     | [@abdusattormv](https://github.com/abdusattormv) | `acpds_cls/weights/best.pt` + validated sample patches               |
| End of Week 5 | [@thebkht](https://github.com/thebkht)                     | [@mirzayv](https://github.com/mirzayv)           | SfM pipeline script + BEV map image                                  |
| End of Week 6 | [@OtabekSadriddinov](https://github.com/OtabekSadriddinov) | [@thebkht](https://github.com/thebkht)           | Localization accuracy results (feeds Week 7 report)                  |
| End of Week 7 | [@abdusattormv](https://github.com/abdusattormv)           | [@mirzayv](https://github.com/mirzayv)           | `POST /park` + `GET /find/{id}` live → unblocks Find My Car frontend |
| End of Week 7 | All                                                        | [@mirzayv](https://github.com/mirzayv)           | All report sections → compile + submit                               |

---

## Week 7 Priority Order

| Priority | Task                                              | Owner                    | Blocks               |
| -------- | ------------------------------------------------- | ------------------------ | -------------------- |
| 1        | `POST /park` + `GET /find/{session_id}` endpoints | @abdusattormv            | Find My Car frontend |
| 2        | Fix `GET /status` response shape                  | @abdusattormv + @mirzayv | Live map screen      |
| 3        | Fix `POST /layout` vs `POST /map` contract        | @abdusattormv + @mirzayv | Owner setup screen   |
| 4        | Wire Find My Car frontend end-to-end              | @mirzayv                 | Demo                 |
| 5        | Switch map rendering to Leaflet                   | @mirzayv                 | Demo                 |
| 6        | Fix edge runtime quad warp at inference           | @thebkht                 | PRD consistency      |
| 7        | Confusion matrix + full comparison table          | @OtabekSadriddinov       | Report               |
| 8        | Val/test gap + per-weather breakdown              | @thebkht                 | Report               |
| 9        | All report sections                               | All                      | Final submission     |

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

Use this sequence for [`@thebkht`](https://github.com/thebkht)'s ACPDS milestone.

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

- `yolov8n-cls`: test accuracy `0.9772`, F1 `0.9715`, size `2.83 MB`
- `yolov8s-cls`: test accuracy `0.9691`, F1 `0.9615`, size `9.78 MB`
- `yolov8m-cls`: test accuracy `0.9738`, F1 `0.9672`, size `30.22 MB`

`yolov8n-cls` is the promoted checkpoint. Larger variants did not improve test accuracy. The bottleneck is Stage 2 patch quality (partial vehicles, thick border regions after perspective warp, low-information crops, label ambiguity) — not model capacity. No further retraining is planned.

Notes:

- `acpds_cls/weights/best.pt` remains the Week 5 handoff checkpoint unless `--promote-stage2` is passed explicitly.
- Exported classifier artifacts are evaluated with batch size `1` in `ml/evaluate.py`.
- Exported ONNX weights must be loaded with `task="classify"` during evaluation to avoid detection-style NMS postprocessing.

Run SIFT localization against either a manifest JSON or a per-spot reference directory:

```bash
make localize-car LOCALIZE_ARGS="--query samples/query.jpg --references samples/localization_refs --output logs/localize_result.json"
```

Evaluate multiple labeled localization queries:

```bash
python ml/evaluate_localization.py --queries samples/localization_refs/query_set.sample.json --references samples/localization_refs/labeled --output-json logs/localize_eval.json --output-csv logs/localize_eval.csv
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

Starter sample references are available under [samples/localization_refs](/Users/thebkht/Projects/smart-parking-system/samples/localization_refs). The current starter set is meant for proof-of-function only. In the current sample evaluation run, `ml/evaluate_localization.py` matched `1/1` labeled queries correctly: `photo_2026-04-23_21.29.43.jpeg` was assigned to `spot_2` with 861 inliers and about 603 ms runtime.

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
