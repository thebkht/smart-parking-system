# Final Artifact Summary

## Dataset Inventory

### Stage 1

| Split | Images | Boxes | Scenes |
| --- | ---: | ---: | ---: |
| train | 1834 | 68685 | 144 |
| val | 560 | 20591 | 31 |
| test | 322 | 11397 | 32 |

### Stage 2

| Split | Free | Occupied |
| --- | ---: | ---: |
| train | 47978 | 62230 |
| val | 10383 | 13538 |
| test | 10291 | 13513 |

### Cross-Dataset Exports

- `pklot_test`: present=True, free=221, occupied=808
- `cnrpark_test`: present=True, free=9849, occupied=11897

### Weather Export

- `stage2_weather`: present=True splits={'sunny': {'free': 25665, 'occupied': 37513}, 'cloudy': {'free': 21067, 'occupied': 23176}, 'rainy': {'free': 18926, 'occupied': 18618}}

## Checkpoints

- Stage 1 `yolov8s`: present=True path=`runs/stage1_det/yolov8s_stage1/weights/best.pt`
- Stage 1 `yolov8m`: present=False path=`runs/stage1_det/yolov8m_stage1/weights/best.pt`
- Stage 2 promoted `yolov8n-cls`: present=True path=`acpds_cls/weights/best.pt`
- Stage 2 comparison `yolov8n-cls`: present=True path=`runs/acpds_cls/yolov8n_stage2/weights/best.pt`
- Stage 2 comparison `yolov8s-cls`: present=True path=`runs/acpds_cls/yolov8s_stage2/weights/best.pt`
- Stage 2 comparison `yolov8m-cls`: present=True path=`runs/acpds_cls/yolov8m_stage2/weights/best.pt`
- Exported ONNX/Core ML bundle: present=True path=`artifacts/models/`

## Acceptance Checks

- stage1_detector_checkpoint: PASS
- stage2_n_checkpoint: PASS
- stage2_s_checkpoint: PASS
- stage2_m_checkpoint: PASS
- stage1_eval_table: PASS
- stage2_eval_table: PASS
- stage2_model_comparison: PASS
- threshold_sweep: PASS
- cross_dataset_eval: PASS
- per_weather_eval: PASS
- benchmark_results: PASS
- bandwidth_report: PASS
- stability_summary: PASS
- backend_route_contract: PASS (`/map`, `/layout`, `/status`, `/park`, `/find/{session_id}`)
- web_frontend_contract: PASS (Owner Setup, Live Occupancy, Find My Car)
- expo_mobile_contract: PASS (same 3 demo screens with native camera/photo picker)

## Latest Metrics Snapshot

- Stage 1 evaluation: `{"mAP50": "0.7791", "mAP50_95": "0.523", "model": "yolov8s_stage1", "precision": "0.8231", "recall": "0.678", "scene_count": "0", "scene_leakage": "False", "split": "val"}`
- Stage 2 evaluation: `{"confusion_matrix": "[[2076, 70], [72, 1590]]", "dataset": "stage2_data/val", "f1": "0.9573", "model": "yolov8n_stage2-2", "precision": "0.9578", "recall": "0.9567", "sample_count": "3808", "support_free": "2146", "support_occupied": "1662", "threshold": "0.5", "top1_accuracy": "0.9627"}`
- Stage 2 model comparison: `{"confusion_matrix": "[[10249, 134], [2054, 11484]]", "dataset": "stage2_data/val", "f1": "0.913", "model": "yolov8m_stage2", "precision": "0.9885", "recall": "0.8483", "sample_count": "23921", "size_mb": "30.22", "support_free": "10383", "support_occupied": "13538", "threshold": "0.3", "top1_accuracy": "0.9085"}`
- Stage 2 threshold sweep: `{"confusion_matrix": "[[10120, 263], [1682, 11856]]", "dataset": "stage2_data/val", "f1": "0.9242", "model": "yolov8m_stage2", "precision": "0.9783", "recall": "0.8758", "sample_count": "23921", "support_free": "10383", "support_occupied": "13538", "threshold": "0.1", "top1_accuracy": "0.9187"}`
- Stage 2 cross-dataset: `{"confusion_matrix": "[[9798, 51], [2358, 9539]]", "dataset": "cnrpark_test", "f1": "0.8879", "model": "yolov8m_stage2", "precision": "0.9947", "recall": "0.8018", "sample_count": "21746", "support_free": "9849", "support_occupied": "11897", "threshold": "0.5", "top1_accuracy": "0.8892"}`
- Stage 2 per-weather: `{"confusion_matrix": "[[18849, 77], [3388, 15230]]", "dataset": "rainy", "f1": "0.8979", "model": "yolov8m_stage2", "precision": "0.995", "recall": "0.818", "sample_count": "37544", "support_free": "18926", "support_occupied": "18618", "threshold": "0.5", "top1_accuracy": "0.9077"}`

## Final Demo Contract

- Canonical inference story: ACPDS parking-space quadrilaterals are perspective-warped into `128x128` patches and classified by `YOLOv8n-cls`; the edge runtime posts compact occupancy JSON instead of raw video.
- Canonical layout route: `GET /map` for edge/runtime consumers. `GET /layout` and `POST /layout` remain compatibility aliases for frontend owner setup.
- Owner setup scope: web clients upload 4+ photos to `POST /layout`, which now runs the SfM pipeline (`ml/sfm_layout.generate_layout`) server-side to build the BEV map + spot polygons and persist the layout. If SfM cannot produce a usable layout it returns `422`, and the manual fallback is a precomputed `LayoutPayload` JSON to `POST /map`. Owners can rename spots afterward via `PATCH /spots/{id}` (label-correction step).
- Live occupancy scope: clients read `GET /status`, use `response.spots`, and scope counts to the active layout's spot IDs.
- Find My Car scope: clients upload a driver photo to `POST /park`, store the returned `session_id`, then call `GET /find/{session_id}` for the highlighted spot polygon. Reference photos are managed per spot via `POST /spots/{id}/references` (stored under `artifacts/spot_references/`), with the bundled `samples/localization_refs/` as fallback.
- Auth scope: optional and off by default; set `AUTH_ENABLED=1` to require bearer tokens (`POST /auth/register`) on owner-mutating routes and to scope Find My Car sessions to their owner.
- Find My Car localizer decision: **SIFT-only is the final scope.** SIFT + FLANN + RANSAC already scores `21/21` top-1 / top-3 on the labeled night-overhead query set (avg `536 ms`), meeting the accuracy targets with no training and CPU-only inference. The MobileNetV3 embedding path stays documented as an optional future upgrade for day/night lighting robustness and is intentionally out of scope for this submission.
