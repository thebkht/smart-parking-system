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

**Next week: process show**

- [x] Val vs test accuracy gap analysis — `logs/week7/val_test_gap.json` documents the ~0.5–1.25pp generalization delta across n/s/m and ties it to unique-lot distribution shift rather than classic overfitting
- [x] Per-weather accuracy breakdown — ACPDS test split is now bucketed into sunny / overcast / low-light luminance tertiles with results saved to `logs/week7/stage2_acpds_weather.json`
- [x] Pooling method (a) vs (b) comparison — `logs/week7/pooling_comparison.json` shows quad warps at `0.9772` test accuracy versus `0.9638` for bounding-square pooling (`-1.34 pp`) on the same YOLOv8n Stage 2 setup
- [x] **Fix edge runtime quad warp** — `edge/detect.py` now preserves polygons end-to-end and classifies `warpPerspective(128×128)` patches; visual QA samples are saved under `logs/week7/warp_comparison/`
- [x] Write Stage 1 and Layout AI sections of technical report

**Two weeks later: final submission**

- [ ] Finalize all accuracy tables and figures (fill in model comparison table with test results)
- [ ] Present ACPDS justification, quad pooling, and ML pipeline in class
- [x] **Update `docs/final-artifact-summary.md` and `docs/final-runbook.md`** — both now describe the canonical ACPDS quadrilateral warp + `YOLOv8n-cls` demo path and the current backend/mobile contract

---

### [@OtabekSadriddinov](https://github.com/OtabekSadriddinov) — ML / research

> Owns evaluation depth, model comparison, literature context, and the report-side interpretation of the promoted `yolov8n-cls` checkpoint. No further retraining is planned on his track; the remaining work is figures, tables, localization evaluation depth, and final write-up consistency.

**Week 5**

- [x] Write Related Work section — ACPDS paper (`arXiv:2107.12207`), PKLot, YOLOv8, SfM / visual localization

**Week 6**

- [x] Build ResNet50 vs YOLOv8 comparison table (accuracy + parameter count + FPS)
- [x] Run confidence threshold sweep on trained `YOLOv8n-cls`
- [x] Test SIFT localization accuracy on 10+ sample ACPDS photos
- [x] Write Stage 2 section of report — architecture, training config, training curves

**Next week: process show**

- [x] Write the final evaluation narrative around the saved Week 6/7 checkpoints — promoted `yolov8n-cls` at `0.9772` test accuracy / `0.9715` F1, the small `n/s/m` spread, and the conclusion that patch quality is the main bottleneck rather than model capacity
- [x] Confusion matrix + PR curve for `yolov8n_stage2` on the test split — explain the expected occupied→free miss pattern in terms of partial vehicles, border regions after warp, and low-information patches
- [x] Full model comparison table — fill in `n` / `s` / `m` / export rows vs the ResNet50 paper baselines with accuracy, F1, parameter count, model size, and backend/export notes
- [x] Localization accuracy table — `samples/localization_refs/query_set.night_overhead.json` now evaluates `21` labeled same-lot night queries, including three reference-frame sanity checks, with `1.000` top-1 / `1.000` top-3 accuracy; tracked summary in `samples/localization_refs/localize_eval.night_overhead.md`
- [x] Write Find My Car and Evaluation sections of technical report

**Two weeks later: final submission**

- [ ] Write Discussion section — cover what worked (n beats larger models, quad warp beats rect crop), limitations (patch quality bottleneck, label ambiguity, localization sample size), and production considerations
- [ ] Review full report for consistency across all sections before [@mirzayv](https://github.com/mirzayv) compiles
- [ ] Present Related Work and Stage 2 model findings in class
- [ ] **Generate confusion matrix figure and PR curve plot as image files** — use the promoted `yolov8n-cls` evaluation outputs and save report-ready figures under `artifacts/figures/`
- [x] **Assemble final model comparison table** — merge `n/s/m` test results plus export/backend notes into one report-ready table; keep the conclusion explicit that larger variants did not beat the promoted checkpoint

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

**Next week: process show**

- [x] **Add `POST /park` endpoint** — accept a driver photo, call `ml/localize.py` SIFT matching against stored `spot_references`, insert a row into `park_sessions` table with `spot_id` + `similarity_score`, return `session_id`
- [x] **Add `GET /find/{session_id}` endpoint** — look up session in `park_sessions`, return `spot_id` + corner coordinates from the `layout` table; return 404 if session not found
- [x] **Resolve `POST /layout` vs `POST /map` naming** — pick one canonical name, update the route in `backend/main.py`, notify [@mirzayv](https://github.com/mirzayv) so frontend fetch path matches, document final contract in `backend/README.md`
- [x] **Fix `GET /status` response shape** — currently returns `{ spots, confidence, timestamp }`; confirm this is the final shape and document it in `backend/README.md` so [@mirzayv](https://github.com/mirzayv) can update the frontend parser to read `response.spots`
- [x] Final FPS + latency table (all backends: MPS / CPU / ONNX FP32 / ONNX INT8)
- [x] Bandwidth savings analysis — expected >99% vs raw H.264; use the measurement script from PRD §8.3 and include actual measured numbers
- [x] System stability test — 30-minute continuous run with no crashes; log CPU usage, memory, and FPS stability; save output to `logs/stability_test.json`
- [x] Write Edge Benchmarks section of technical report

**Two weeks later: final submission**

- [x] **Fix duplicate `POST /park` handler** — there are two handlers for the same path in `backend/main.py`; the first returns `501` and sits before the real multipart handler, blocking the Find My Car flow end-to-end; remove the stub and confirm the multipart handler is the only route for that path
- [ ] **Wire real owner setup into `POST /layout`** — PRD says owner setup is `4-5 photos -> SfM -> store map + spot polygons`; current backend only aliases `/layout` to `/map` and stores precomputed layout JSON instead of invoking the SfM flow
- [ ] **Add owner-setup fallback path** — if SfM fails, support the PRD fallback where the app can continue with manual polygon submission/correction instead of a hard failure
- [ ] **Align `edge/stability_test.py` with the live quad-geometry path** — it still uses fixed ROI boxes while the runtime path loads quadrilaterals from `/map`; update the soak test so the 30-minute stability claim matches production behavior
- [x] **Add backend happy-path tests for PRD endpoints** — `tests/test_backend.py` now covers `/map`, `/layout`, `/sessions`, `/park`, and `/find/{session_id}` success paths
- [x] **Fix backend test/runtime dependency for multipart uploads** — `python-multipart` is declared in `requirements.txt`, and `.venv/bin/python -m pytest tests/test_backend.py tests/test_edge.py -q` passes
- [x] **Document or implement the `GET /layout` contract explicitly** — `backend/main.py` exposes `GET /layout` as an alias for `GET /map`, and `backend/README.md` documents `/map` as canonical
- [ ] **Decide whether to keep SIFT-only Find My Car or add the MobileNetV3 upgrade path** — PRD marks MobileNetV3 embeddings as the optional upgrade; record the final backend-side decision in docs/report and implement it only if it stays in scope
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

**Next week: process show**

- [x] **Fix `GET /status` response parser** — backend returns `{ spots, confidence, timestamp }`; update frontend to read `response.spots` before coloring polygons and updating free/occupied count in the header
- [x] **Fix `POST /layout` call** — align to the canonical name once [@abdusattormv](https://github.com/abdusattormv) resolves the contract, then update the fetch path and `backend/README.md`
- [x] **Remove mock fallbacks from Find My Car** — replace fake `session_id` generation and random spot fallback with the real `POST /park` → store `session_id` → `GET /find/{session_id}` flow once [@abdusattormv](https://github.com/abdusattormv) ships the endpoints
- [x] **Wire Find My Car end-to-end** — camera capture → `POST /park` with photo → store `session_id` in local state → `GET /find/{session_id}` → highlight the returned spot polygon in amber on the Leaflet map
- [x] **Switch map rendering to Leaflet** — current UI uses custom SVG; `react-router-dom` and `leaflet` are installed but not used; migrate the live occupancy map and Find My Car screens to actual Leaflet polygon overlays with per-spot color updates
- [x] React Native (Expo) wrapper — native camera/photo-picker access for the same Owner Setup, Live Occupancy, and Find My Car demo screens
- [ ] Write App section of technical report — 3 screens, tech stack, Leaflet integration, Find My Car flow

**Two weeks later: final submission**

- [ ] Write Abstract, Conclusion, and References
- [ ] Submit technical report via email before deadline
- [ ] Run live Find My Car demo in class — present all 3 app screens
- [x] **Decide React Native in/out and update docs accordingly** — Expo wrapper is in scope and documented for the final mobile demo
- [x] **Fix `GET /status` parser and Find My Car frontend** — web and mobile clients read `response.spots` and use the real `POST /park` -> `GET /find/{session_id}` flow
- [x] **Verify `POST /layout` fetch path** matches the canonical backend route — `/layout` is a documented compatibility alias over the `/map` layout contract

---

## Next Week: Process Show Priority Order

| Priority | Task                                              | Owner                                                                                     | Blocks               |
| -------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------- | -------------------- |
| 1        | `POST /park` + `GET /find/{session_id}` endpoints | [@abdusattormv](https://github.com/abdusattormv)                                          | Find My Car frontend |
| 2        | Fix `GET /status` response shape                  | [@abdusattormv](https://github.com/abdusattormv) + [@mirzayv](https://github.com/mirzayv) | Live map screen      |
| 3        | Fix `POST /layout` vs `POST /map` contract        | [@abdusattormv](https://github.com/abdusattormv) + [@mirzayv](https://github.com/mirzayv) | Owner setup screen   |
| 4        | Wire Find My Car frontend end-to-end              | [@mirzayv](https://github.com/mirzayv)                                                    | Demo                 |
| 5        | Switch map rendering to Leaflet                   | [@mirzayv](https://github.com/mirzayv)                                                    | Demo                 |
| 6        | Fix edge runtime quad warp at inference           | [@thebkht](https://github.com/thebkht)                                                    | PRD consistency      |
| 7        | Confusion matrix + full comparison table          | [@OtabekSadriddinov](https://github.com/OtabekSadriddinov)                                | Report               |
| 8        | Val/test gap + per-weather breakdown              | [@thebkht](https://github.com/thebkht)                                                    | Report               |
| 9        | All report sections                               | All                                                                                       | Final submission     |

---

## Two Weeks Later: Final Submission Priority Order

| Priority | Task                                                                    | Owner                                                      | Blocks                                                                     |
| -------- | ----------------------------------------------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------- |
| 1        | Fix duplicate `POST /park` handler                                      | [@abdusattormv](https://github.com/abdusattormv)           | Complete                                                                   |
| 2        | Fix `GET /status` parser in frontend                                    | [@mirzayv](https://github.com/mirzayv)                     | Complete                                                                   |
| 3        | Wire Find My Car frontend end-to-end                                    | [@mirzayv](https://github.com/mirzayv)                     | Complete                                                                   |
| 4        | Document Expo mobile demo flow                                           | [@mirzayv](https://github.com/mirzayv)                     | Complete — see `frontend/README.md`                                        |
| 5        | Generate confusion matrix + PR curve figures                            | [@OtabekSadriddinov](https://github.com/OtabekSadriddinov) | Report figures                                                             |
| 6        | Assemble final model comparison table                                   | [@OtabekSadriddinov](https://github.com/OtabekSadriddinov) | Report table                                                               |
| 7        | Write Discussion section                                                | [@OtabekSadriddinov](https://github.com/OtabekSadriddinov) | Report                                                                     |
| 8        | Localization accuracy (21 labeled night-overhead photos)                | [@OtabekSadriddinov](https://github.com/OtabekSadriddinov) | Complete — see `samples/localization_refs/localize_eval.night_overhead.md` |
| 9        | Update `docs/final-artifact-summary.md` + `docs/final-runbook.md` to v6 | [@thebkht](https://github.com/thebkht)                     | Complete                                                                   |
| 10       | Write Abstract, Conclusion, References, App section                     | [@mirzayv](https://github.com/mirzayv)                     | Report                                                                     |
| 11       | Finalize accuracy tables + figures for presentation                     | [@thebkht](https://github.com/thebkht)                     | Presentation                                                               |
| 12       | Compile full report PDF                                                 | [@abdusattormv](https://github.com/abdusattormv)           | Final submission                                                           |

---

## What’s Left According to the PRD

The core system exists in the repo, but the PRD is still not fully satisfied. The biggest remaining gaps are integration and product completeness rather than model training.

- **Owner setup is still a prototype path** — the PRD expects `4-5 photos -> SfM -> store map + spot polygons`, but the backend currently accepts precomputed layout JSON rather than running the SfM flow end-to-end — owners: [@abdusattormv](https://github.com/abdusattormv) + [@thebkht](https://github.com/thebkht)
- **Owner correction/fallback flow is incomplete** — the PRD expects polygon correction plus a fallback when SfM fails; the current app mostly previews the result and does not expose a full correction workflow — owners: [@mirzayv](https://github.com/mirzayv) + [@abdusattormv](https://github.com/abdusattormv)
- **Live map frontend is aligned with the backend contract** — web and mobile clients read `response.spots` and scope counts to the active layout's spot IDs — owner: [@mirzayv](https://github.com/mirzayv)
- **Find My Car frontend uses the real backend path** — web and mobile clients call `POST /park`, store `session_id`, then call `GET /find/{session_id}` — owner: [@mirzayv](https://github.com/mirzayv)
- **Reference-photo management is not productized** — localization currently reads from a fixed sample folder instead of a real per-spot reference management flow — owners: [@abdusattormv](https://github.com/abdusattormv) + [@thebkht](https://github.com/thebkht)
- **Stability verification does not yet match the live runtime path** — the checked-in stability harness still uses fixed ROI boxes instead of the `/map`-driven quadrilateral geometry contract used by `edge/detect.py` — owner: [@abdusattormv](https://github.com/abdusattormv)
- **Backend happy-path coverage is in place** — `/map`, `/layout`, `/sessions`, `/park`, and `/find/{session_id}` success cases are covered in `tests/test_backend.py` — owner: [@abdusattormv](https://github.com/abdusattormv)
- **Backend multipart dependency is declared and verified** — `/park` has `python-multipart` available through `requirements.txt`, and the focused backend/edge tests pass — owner: [@abdusattormv](https://github.com/abdusattormv)
- **Docs are normalized for the final demo contract** — `/map` is canonical, `/layout` is the frontend compatibility alias, and the ACPDS quadrilateral path is the report-facing architecture — owners: [@thebkht](https://github.com/thebkht) + [@abdusattormv](https://github.com/abdusattormv) + [@mirzayv](https://github.com/mirzayv)
- **React Native/mobile scope is in** — Expo wrapper is present for the same three demo screens and documented in `frontend/README.md` — owner: [@mirzayv](https://github.com/mirzayv)

### Short Priority View

**Before the process show**

1. Fix frontend `/status` parsing. Complete.
   Owner: [@mirzayv](https://github.com/mirzayv)
2. Replace Find My Car frontend mock fallbacks with the real backend flow. Complete.
   Owner: [@mirzayv](https://github.com/mirzayv)
3. Add backend happy-path tests and install `python-multipart`. Complete.
   Owner: [@abdusattormv](https://github.com/abdusattormv)
4. Align the stability test with the quadrilateral `/map` path.
   Owner: [@abdusattormv](https://github.com/abdusattormv)
5. Decide whether owner setup will be shown as a real integrated flow or as a documented prototype.
   Owners: [@abdusattormv](https://github.com/abdusattormv) + [@thebkht](https://github.com/thebkht) + [@mirzayv](https://github.com/mirzayv)

**Before final submission**

1. Wire the real owner setup flow into `/layout`, or explicitly narrow the documented scope.
   Owners: [@abdusattormv](https://github.com/abdusattormv) + [@thebkht](https://github.com/thebkht)
2. Add owner fallback/correction behavior.
   Owners: [@mirzayv](https://github.com/mirzayv) + [@abdusattormv](https://github.com/abdusattormv)
3. Keep report-facing docs aligned with the final `/map` + `/layout` compatibility contract if backend behavior changes again.
   Owners: [@thebkht](https://github.com/thebkht) + [@abdusattormv](https://github.com/abdusattormv) + [@mirzayv](https://github.com/mirzayv)
4. Decide SIFT-only vs MobileNetV3 upgrade scope for Find My Car.
   Owners: [@thebkht](https://github.com/thebkht) + [@abdusattormv](https://github.com/abdusattormv) + [@OtabekSadriddinov](https://github.com/OtabekSadriddinov)
5. Run the Expo demo on a phone using the Mac LAN IP in `frontend/mobile/api.js`.
   Owner: [@mirzayv](https://github.com/mirzayv)

---

## Remaining Gap Checklist

### 🔴 Must fix before demo

- [x] **Fix `GET /status` response parser** (`frontend/src/App.jsx`) — frontend reads the whole object as spot states instead of `response.spots`; live map is broken — [@mirzayv](https://github.com/mirzayv)
- [x] **Wire Find My Car frontend end-to-end** — replace fake session IDs and random-spot fallback with real `POST /park` → `GET /find/{session_id}` flow — [@mirzayv](https://github.com/mirzayv)
- [x] **Add backend happy-path tests for PRD endpoints** — `/park`, `/find/{session_id}`, `/sessions`, `/layout`, and `/map` success paths are covered in `tests/test_backend.py` — [@abdusattormv](https://github.com/abdusattormv)
- [x] **Fix backend multipart dependency and run tests green** — `python-multipart` is declared in `requirements.txt`, and `.venv/bin/python -m pytest tests/test_backend.py tests/test_edge.py -q` passes — [@abdusattormv](https://github.com/abdusattormv)

### 🟡 Should fix for PRD compliance

- [ ] **Wire real owner setup into `POST /layout`** — PRD says owner setup is `4-5 photos -> SfM -> store map + spot polygons`; current backend aliases `/layout` to `/map` and stores precomputed layout JSON instead of invoking the SfM flow — [@abdusattormv](https://github.com/abdusattormv) + [@thebkht](https://github.com/thebkht)
- [ ] **Add owner-setup fallback path** — if SfM fails, support the PRD fallback where the app can continue with manual polygon submission/correction instead of hard-failing the flow — [@abdusattormv](https://github.com/abdusattormv) + [@mirzayv](https://github.com/mirzayv)
- [ ] **Owner setup: polygon label-correction step** — PRD owner flow requires editing/relabelling spot polygons after layout generation; `frontend/src/App.jsx:347` only previews the map with no edit workflow — [@mirzayv](https://github.com/mirzayv)
- [ ] **Align `edge/stability_test.py` with the live quad-geometry path** — the checked-in stability harness still builds fixed ROI boxes, while the production runtime loads quadrilaterals from `/map`; the soak test should exercise the same geometry contract as `edge/detect.py` — [@abdusattormv](https://github.com/abdusattormv)
- [ ] **Reference-photo management for Find My Car** — PRD implies per-spot stored references; backend currently reads a fixed local sample folder rather than persisting uploaded references per spot in the DB — [@abdusattormv](https://github.com/abdusattormv)
- [x] **Resolve the `GET /layout` contract explicitly** — `backend/main.py` exposes `GET /layout` as an alias for `GET /map`, and `backend/README.md` documents `/map` as canonical — [@abdusattormv](https://github.com/abdusattormv)
- [ ] **Decide whether Find My Car stays SIFT-only** — PRD marks MobileNetV3 embeddings as the optional upgrade path; record the final decision in docs/report and implement it only if it remains in scope — [@abdusattormv](https://github.com/abdusattormv) + [@thebkht](https://github.com/thebkht)
- [x] **Write frontend README** — `frontend/README.md` is still the default Vite template; add setup instructions, env vars, screen descriptions, and how to connect to the backend — [@mirzayv](https://github.com/mirzayv)
- [x] **Resolve doc inconsistencies on canonical routes and architecture** — `docs/final-artifact-summary.md`, `docs/final-runbook.md`, and `backend/README.md` now agree on ACPDS v6, canonical `/map`, and `/layout` compatibility aliases — [@thebkht](https://github.com/thebkht)
- [ ] **Add frontend tests for three PRD screens and API contracts** — no frontend tests exist for owner setup, live map, or Find My Car flows — [@mirzayv](https://github.com/mirzayv)
- [x] **Decide React Native in/out and update all docs** — Expo wrapper is built and documented as part of the final demo path — [@mirzayv](https://github.com/mirzayv)

### 🟢 Nice to have

- [ ] **End-to-end smoke-test script** — a single runnable script that proves the full PRD path works: owner setup → map persistence → edge updates → live map → park/find flow; currently no such script exists
- [x] **React Native / Expo wrapper** — native camera/photo-picker access for the Owner Setup, Live Occupancy, and Find My Car demo screens — [@mirzayv](https://github.com/mirzayv)
- [ ] **Auth and session ownership model** — no auth, multi-user handling, or session ownership is present; acceptable for a class demo but absent from a fuller product interpretation
- [ ] **Run one full green local verification pass** — after fixing the multipart dependency, confirm `make test`, `make backend`, and one representative `make edge ...` path all work in the checked-in environment

---

## Handoff Points

| When                    | From                                                       | To                                               | Deliverable                                                                             |
| ----------------------- | ---------------------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------- |
| End of Week 5           | [@thebkht](https://github.com/thebkht)                     | [@abdusattormv](https://github.com/abdusattormv) | `acpds_cls/weights/best.pt` + validated sample patches                                  |
| End of Week 5           | [@thebkht](https://github.com/thebkht)                     | [@mirzayv](https://github.com/mirzayv)           | SfM pipeline script + BEV map image                                                     |
| Before the process show | [@OtabekSadriddinov](https://github.com/OtabekSadriddinov) | [@thebkht](https://github.com/thebkht)           | Localization accuracy results (feeds the process-show narrative and evaluation section) |
| End of next week        | [@abdusattormv](https://github.com/abdusattormv)           | [@mirzayv](https://github.com/mirzayv)           | `POST /park` + `GET /find/{id}` live → unblocks Find My Car frontend                    |
| Before final submission | All                                                        | [@mirzayv](https://github.com/mirzayv)           | All report sections → compile + submit                                                  |

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
python ml/evaluate_localization.py --queries samples/localization_refs/query_set.night_overhead.json --references samples/localization_refs/labeled --output-json logs/localize_eval.json --output-csv logs/localize_eval.csv
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

Starter sample references are available under [samples/localization_refs](samples/localization_refs). The current checked-in evaluation set is [query_set.night_overhead.json](samples/localization_refs/query_set.night_overhead.json): `21` labeled same-lot night-overhead queries, including the three labeled reference timestamps as sanity-check queries, scored `21/21` top-1 correct and `21/21` top-3 correct, with average runtime `536.18 ms`, summarized in [localize_eval.night_overhead.md](samples/localization_refs/localize_eval.night_overhead.md). Regenerated machine outputs are still written to `logs/localize_eval.json` and `logs/localize_eval.csv`.

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

---
