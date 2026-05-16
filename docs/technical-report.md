# Smart Parking System Technical Report

## Abstract

This project implements a two-stage smart parking system that performs inference on an edge device rather than streaming raw video to the cloud. The deployed pipeline uses a Stage 1 YOLO parking-space detector to localize parking spaces in full-frame images, then applies a Stage 2 YOLOv8 classification model to cropped space patches to predict `free` or `occupied`. The system outputs compact JSON payloads and stores them through a minimal FastAPI backend. In the checked-in Week 6 comparison set, the Stage 1 detector achieved up to 0.995 mAP@50 on a scene-held-out validation split, while the promoted `yolov8n-cls` checkpoint remained the best Stage 2 test model at 0.9772 accuracy and 0.9715 F1 at threshold 0.5. The Week 6 export bundle for the promoted `yolov8n-cls` checkpoint produced `best.onnx`, `best_int8.onnx`, and `best.mlpackage` artifacts, each materially smaller than raw video streaming. Compared with a conservative 1080p H.264 camera stream, the JSON reporting path reduced bandwidth by 99.9%.

## 1. Introduction

Parking occupancy detection is a practical edge-computing problem because the required output is small, structured, and time-sensitive, while the raw input is privacy-sensitive and bandwidth-heavy. Traditional parking systems often rely on per-slot hardware sensors or centralized video pipelines. Both approaches increase deployment cost or require continuous transfer of image data away from the camera location.

The goal of this project was to build a laptop-based parking occupancy system that keeps inference on-device and exports only the final occupancy state. The repo’s final runtime contract is:

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

This contract is produced by the edge pipeline and stored unchanged by the backend. The result is a compact, deployment-oriented system that supports local inference, minimal network transfer, and a small backend surface.

## 2. Problem Formulation and Design Rationale

The central design decision in this project is the use of a two-stage pipeline:

`full-frame parking scene -> parking-space localization -> per-space crop -> occupancy classification`

This separation is necessary because the available datasets are naturally split across two tasks:

- full-frame parking-space localization from parking-scene images
- patch-level occupied/free classification from cropped parking-spot images

Using a single detector trained directly on patch-style data is not the right formulation for the occupancy problem in full-frame scenes. For this reason, the repo keeps a single-model occupancy detector only as an ML comparison baseline, not as the deployed default.

The final runtime supports both:

- fixed ROIs for static-camera demos
- a trained Stage 1 detector for the generalized final path

The checked-in final runbook and artifact set use the trained two-stage path as the canonical system:

`trained Stage 1 parking-space detector -> crop -> trained Stage 2 classifier -> smoothing -> JSON -> FastAPI`

## 3. System Architecture

The system is organized into three layers.

### 3.1 Edge inference layer

The edge device runs both inference stages locally.

- Stage 1: parking-space localization with YOLO detection
- Stage 2: crop classification with YOLOv8 classification
- Post-processing: temporal smoothing over spot status history
- Output: structured JSON payload

The current integrated runtime uses:

- Stage 1 detector: `runs/stage1_det/yolov8s_stage1/weights/best.pt`
- Stage 2 classifier: `acpds_cls/weights/best.pt` for the Week 5 handoff, with Week 6 comparison checkpoints under `runs/acpds_cls/`

### 3.2 Backend layer

The backend is intentionally minimal.

- `POST /update` stores the latest edge payload
- `GET /status` returns the most recent payload
- `GET /history` returns recorded payload history
- `GET /health` provides a health probe
- `GET /stream` exposes the latest annotated frame as MJPEG

The backend stores the payload as-is and does not introduce derived dashboard fields.

### 3.3 Deployment exports

The repo also includes exported deployment artifacts:

- `artifacts/models/best.pt`
- `artifacts/models/best.onnx`
- `artifacts/models/best_int8.onnx`

These enable runtime benchmarking across PyTorch, ONNX, and Core ML style deployment paths.

## 4. Datasets and Data Preparation

### 4.1 Stage 1 dataset

Stage 1 uses full-frame parking-scene annotations and is evaluated with scene holdout rather than random image splitting. This is important because image-level random splits over parking-lot video frames can leak scene-specific information and overstate generalization.

The final Stage 1 dataset report shows:

| Split | Images | Boxes | Scenes |
| --- | ---: | ---: | ---: |
| Train | 1953 | 72607 | 123 |
| Val | 337 | 13433 | 27 |
| Test | 396 | 13733 | 27 |

Additional data-preparation checks recorded in the manifest:

- duplicates removed: 3928
- empty-label frames excluded: 80
- polygon labels converted to boxes: 99773
- scene leakage detected: false

### 4.2 Stage 2 dataset

Stage 2 uses cropped parking-spot patches with `free` and `occupied` labels extracted from ACPDS quadrilateral annotations. The extraction flow in `ml/extract_patches.py` normalizes corner order, applies `cv2.getPerspectiveTransform`, warps each spot into a `128 x 128` patch, and writes split-aware outputs under `datasets/acpds_stage2/`. Before training, the workflow requires a human validation gate recorded in `datasets/acpds_stage2/validation_report.json`; training is blocked unless that report is marked `passed`.

The checked-in Stage 2 inventory, mirrored by `artifacts/final_manifest.json`, is:

| Split | Free | Occupied |
| --- | ---: | ---: |
| Train | 47978 | 62230 |
| Val | 10383 | 13538 |
| Test | 10291 | 13513 |

This split structure follows the ACPDS unique-lot evaluation goal rather than frame-random splitting. The intent is to measure generalization to unseen parking lots instead of memorization of camera-specific geometry. In practice, the Stage 2 dataset is derived from the Week 5 ACPDS manifest and patch-validation workflow rather than hand-curated image folders.

The repo also contains optional cross-dataset exports for later generalization checks:

- `datasets/pklot_test`: 221 free, 808 occupied
- `datasets/cnrpark_test`: 9849 free, 11897 occupied

When weather labels are available during preparation, the pipeline also materializes `datasets/stage2_weather/` for later analysis:

| Weather | Free | Occupied |
| --- | ---: | ---: |
| Sunny | 25665 | 37513 |
| Cloudy | 21067 | 23176 |
| Rainy | 18926 | 18618 |

### 4.3 Baseline support

A single-model full-frame occupancy detector baseline is supported in the repo as a comparison path. The saved validation log includes a run with:

- mAP@50: 0.9042
- mAP@50-95: 0.5388
- precision: 0.9270
- recall: 0.8264

This baseline is retained for ML comparison, but it is not the deployed system path.

## 5. Stage 1 Localization

### 5.1 Role in the pipeline

Stage 1 is the full-frame localization step that bridges parking-lot imagery and Stage 2 occupancy classification. In the PRD v6 pipeline, Stage 1 receives either ACPDS/SfM quadrilateral spot geometry or Stage 1 detector boxes, then converts each spot into a canonical `128 x 128` patch for Stage 2. The important Week 7 runtime change is that `edge/detect.py` now applies the same quadrilateral warp used during ACPDS extraction instead of classifying axis-aligned crops. This removes a train/serve mismatch that previously weakened the architecture story.

### 5.2 Dataset and detector training

Stage 1 uses scene-held-out full-frame parking scenes rather than frame-random splits. That choice matters because adjacent parking-lot frames share viewpoint, lane geometry, and lighting, so a random split would overstate localization generalization. The final dataset report in §4.1 summarizes 1953 train images, 337 validation images, and 396 test images across disjoint scenes.

The integrated detector is `yolov8s_stage1`. The repo still contains an earlier training report snapshot in `models/stage1_s_report.json` with lower metrics, but the later Week 6 evaluation row used in the runbook and artifact summary is the stronger checkpoint-level result:

- mAP@50: `0.9950`
- mAP@50-95: `0.9603`
- precision: `0.9971`
- recall: `0.9970`

These numbers are sufficient for the handoff role even though Stage 1 is not the main research bottleneck. By comparison, the `yolov8n_stage1` run was also strong, but the `s` checkpoint remained the integrated default for the final demo path.

### 5.3 Runtime handoff to Stage 2

There are two supported Stage 1 handoff modes in the repo:

- known-layout mode: load fixed quadrilaterals from ACPDS-derived config or backend `/map`
- generalized mode: run the Stage 1 detector, convert each predicted box to a degenerate four-corner polygon, and pass it through the same Stage 2 warp path

This unified geometry contract is important. ACPDS training patches are perspective-corrected with `order_corners() -> getPerspectiveTransform() -> warpPerspective(128 x 128)`, and Week 7 aligns live inference with that same contract. The optional ACPDS paper comparison between quadrilateral pooling and bounding-square pooling is partially implemented in the extraction tooling through a new `--pooling {quad,square}` flag, but the square-pooling retrain was deferred because the core submission path depended more directly on inference consistency and evaluation depth.

## 6. Stage 2 Evaluation Results

### 6.1 Stage 2 classifier

Stage 2 uses YOLOv8 classification models trained on cropped parking-spot patches. The Week 6 comparison at threshold `0.5` produced:

| Model | Split | Accuracy | Precision | Recall | F1 | Size (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolov8n_stage2` | Val | 0.9827 | 0.9784 | 0.9819 | 0.9802 | 2.83 |
| `yolov8s_stage2` | Val | 0.9816 | 0.9727 | 0.9856 | 0.9791 | 9.78 |
| `yolov8m_stage2` | Val | 0.9795 | 0.9670 | 0.9868 | 0.9768 | 30.22 |
| `yolov8n_stage2` | Test | 0.9772 | 0.9864 | 0.9570 | 0.9715 | 2.83 |
| `yolov8s_stage2` | Test | 0.9691 | 0.9745 | 0.9488 | 0.9615 | 9.78 |
| `yolov8m_stage2` | Test | 0.9738 | 0.9846 | 0.9504 | 0.9672 | 30.22 |

The important result is that increasing model size did not improve the deployed handoff checkpoint. The promoted `yolov8n-cls` model remained best on both validation and test accuracy while also being the smallest artifact, which points away from classifier capacity as the main bottleneck. The more likely limitation is Stage 2 patch quality: partial vehicles, thick border regions after the perspective warp, low-information crops, and some label ambiguity.

Because the deployed edge path converts classifier probability into a binary occupancy decision, threshold selection matters, but the Week 6 comparison already shows a broader issue: the default-threshold results plateau just below the original 98% target regardless of model size.

- Best validation accuracy: `0.9827` with `yolov8n_stage2`
- Best test accuracy: `0.9772` with `yolov8n_stage2`
- Best test F1: `0.9715` with `yolov8n_stage2`

The accuracy spread between `n`, `s`, and `m` is small, and the smallest model actually leads the comparison. That pattern is consistent with a data-quality bottleneck rather than an underpowered classifier.

### 6.2 Validation vs test generalization gap

Week 7 adds an explicit validation/test gap analysis in `logs/week7/val_test_gap.json`. Across `n`, `s`, and `m`, the average test-minus-validation accuracy delta is `-0.0079`, and the average recall delta is `-0.0327`. The largest drop occurs in the `s` variant, but the pattern is consistent across all three models:

| Model | Val acc | Test acc | Delta | Val recall | Test recall | Recall delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolov8n-cls` | 0.9827 | 0.9772 | -0.0055 | 0.9819 | 0.9570 | -0.0249 |
| `yolov8s-cls` | 0.9816 | 0.9691 | -0.0125 | 0.9856 | 0.9488 | -0.0368 |
| `yolov8m-cls` | 0.9795 | 0.9738 | -0.0057 | 0.9868 | 0.9504 | -0.0364 |

This does not look like classic overfitting where train and validation strongly diverge and a larger model closes the gap. Instead, it matches the ACPDS unique-lot split design: unseen parking lots introduce distribution shift in camera pose, line markings, border thickness, and partial-vehicle appearance. The dominant degradation is recall rather than precision, which means hard occupied examples are more likely to flip to `free` on test than free spaces are to flip to `occupied`.

The same Week 7 artifact also evaluates the promoted checkpoint per held-out source image. Most test images remain above 0.93 accuracy, but darker scenes such as `GOPR6711` and `GOPR6712` show the clearest recall erosion. That finding connects directly to the weather proxy analysis in §6.4.

### 6.3 Cross-dataset evaluation

Cross-dataset evaluation is important because it tests generalization to a different parking lot and camera setup. The latest saved `yolov8m_stage2` result on `datasets/cnrpark_test` is:

- accuracy: 0.8892
- precision: 0.9947
- recall: 0.8018
- F1: 0.8879

Confusion matrix:

| Actual / Predicted | Free | Occupied |
| --- | ---: | ---: |
| Free | 9798 | 51 |
| Occupied | 2358 | 9539 |

This result shows that the classifier remains strong on unseen data, but recall drops relative to the in-domain validation set. The dominant failure mode is missed occupied spots rather than false occupied predictions.

### 6.4 ACPDS weather proxy evaluation

ACPDS does not ship native weather labels, so Week 7 uses an image-level luminance proxy on the ACPDS test split. For each held-out source image, the pipeline computes the mean HSV value channel and buckets the image into tertiles: brightest third = `sunny`, middle third = `overcast`, darkest third = `low_light`. The resulting split and thresholds are saved in `logs/week7/acpds_weather_buckets.json`, and the materialized patch layout lives under `datasets/acpds_stage2_weather/`.

Using the promoted `yolov8n-cls` checkpoint at threshold `0.5`, the ACPDS weather-proxy results are:

| Weather proxy | Accuracy | Precision | Recall | F1 | Samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Sunny | 0.9894 | 0.9832 | 0.9888 | 0.9860 | 470 |
| Overcast | 0.9774 | 0.9753 | 0.9576 | 0.9664 | 486 |
| Low-light | 0.9663 | 0.9959 | 0.9351 | 0.9646 | 534 |

The ordering is intuitive: the brightest scenes are easiest, while low-light scenes show the weakest recall. The more important point is methodological rather than absolute. These are proxy labels derived from luminance, not official ACPDS metadata, so the result should be interpreted as a robustness slice over illumination rather than a claim about meteorological weather categories. Even with that caveat, the low-light recall drop reinforces the broader Week 7 conclusion that occupancy misses are driven more by difficult patch conditions than by lack of classifier capacity.

### 6.5 End-to-end stability

The checked-in stability run used:

- Stage 1: `yolov8s_stage1`
- Stage 2: `yolov8m_stage2`
- device: `mps`
- duration: 15 seconds

Recorded result:

- iterations: 14
- successful iterations: 14
- read failures: 0
- errors: 0
- pass status: true
- average iterations per second: 0.927

This is a short reproducible stability check rather than a long soak test, but it verifies that the integrated two-stage path runs without runtime exceptions in the saved configuration.

## 7. Layout AI

### 7.1 Purpose

The Layout AI path is the owner-setup workflow for new parking lots. Instead of requiring manual ROI entry, the intended flow is: capture `4-5` overlapping lot photos, build a rough bird's-eye background image, and produce a `layout.json` file containing quadrilateral spot polygons. This output can then be stored by the backend and consumed by the app or edge runtime through the same `/map` geometry contract used elsewhere in the system.

### 7.2 Implementation

The prototype implementation is `ml/sfm_layout.py`. It is not a full structure-from-motion reconstruction stack, but it does provide a deterministic geometry-generation workflow suitable for the class milestone:

- load a small image set from the owner
- detect ORB features on the base frame and later frames
- match descriptors with a Hamming BFMatcher
- estimate inter-frame homographies with RANSAC
- warp matched frames into a shared canvas
- crop nonzero content and save `bev_map.png`
- write `layout.json` with either provided spot polygons or a placeholder parking-grid fallback

This is enough to generate a usable BEV-style map sample and validate the end-to-end data contract. The resulting layout artifact is handed off to backend/app code as a background image plus spot polygons.

### 7.3 Limitations and handoff

The Layout AI prototype is intentionally honest about its limits. It depends on sufficient image overlap, reasonably textured surfaces for ORB matching, and a mostly planar lot surface so that homography stitching remains plausible. It is also not an automatic spot-discovery model: if no explicit polygon JSON is provided, it falls back to a placeholder grid. For the project scope, that is acceptable because the main integration requirement is the map-generation workflow and polygon contract, not a production-grade photogrammetry pipeline.

This section also connects to the Find My Car path. The same BEV map and spot geometry become the spatial reference used by later localization logic, while the backend exposes those polygons unchanged to downstream consumers.

## 8. Runtime Benchmark and Bandwidth Results

### 7.1 Export artifacts

The Week 6 export workflow for the promoted `yolov8n-cls` checkpoint generated:

| Artifact | Size (MB) |
| --- | ---: |
| `artifacts/models/best.pt` | 2.83 |
| `artifacts/models/best.onnx` | 5.51 |
| `artifacts/models/best_int8.onnx` | 1.44 |
| `artifacts/models/best.mlpackage` | 1.44 |

These artifacts confirm that the Stage 2 classifier can be packaged into compact deployment formats. The ONNX and Core ML exports are useful for downstream runtime benchmarking, but the Week 6 milestone here is artifact generation and size tracking rather than final latency claims.

### 7.2 Bandwidth analysis

The saved bandwidth analysis compares JSON payload reporting against a conservative 1080p H.264 stream:

- JSON payload size: 280 bytes
- POST rate: 0.5 per second
- effective JSON throughput: 290 B/s including header overhead
- H.264 throughput assumption: 250 KB/s
- bandwidth savings: 99.9%
- reduction factor: 862x less data

This result directly supports the edge-computing rationale of the project. The system sends compact occupancy state rather than raw video, which substantially reduces bandwidth and improves privacy.

## 9. Discussion

The project achieved its main system goal: a working two-stage edge pipeline that performs local inference and exports only structured occupancy status. The strongest outcomes are:

- very strong Stage 1 scene-held-out localization performance
- a reproducible `n` / `s` / `m` Stage 2 comparison
- a working SIFT + FLANN localization prototype with a sample `1/1` correct query evaluation
- compact ONNX and Core ML export artifacts for the promoted Week 5 checkpoint
- strong bandwidth savings relative to continuous video streaming

The main limitations are also clear.

First, the Stage 2 classifier still did not exceed the original 98% target on the checked-in test split. The best saved result is the promoted `yolov8n-cls` checkpoint at 0.9772 test accuracy, which is close but still below that goal. Second, the small spread across `n`, `s`, and `m`, combined with `n` leading both accuracy and artifact size, implies that increasing model capacity is not the main lever. The more likely limiting factor is patch quality: partial vehicles, border-heavy crops, low-light conditions, and some label ambiguity. Third, the Week 7 runtime fix shows that evaluation and deployment must share the same geometry contract; otherwise, even a strong classifier is penalized by mismatched pooling at inference time. Fourth, the saved stability test is still short, so a longer 30-minute or multi-hour soak test would provide stronger reliability evidence.

## 10. Conclusion

This project demonstrates that a laptop-based edge parking system can combine full-frame parking-space localization and per-space occupancy classification into a practical two-stage inference pipeline. The checked-in Week 7 state uses a YOLO Stage 1 parking-space detector, a promoted `yolov8n-cls` checkpoint for the handoff path, and a runtime that now warps spot quadrilaterals to `128 x 128` exactly as the ACPDS training pipeline does. The system achieves strong Stage 1 localization performance, useful Stage 2 comparison coverage, a working proof-of-function SIFT localization path, a BEV layout-generation prototype, compact export artifacts, and bandwidth reduction of more than 99.9% compared with continuous video transmission.

The final repo state is suitable for class demonstration and technical submission because it includes:

- trained checkpoints for both inference stages
- evaluation logs for model comparison, threshold sweep, cross-dataset testing, and per-weather analysis
- sample localization references plus a localization evaluation harness
- exported deployment artifacts
- an integrated edge runtime and backend
- regenerated acceptance summaries showing all tracked PRD checks as complete

## 11. References

1. PKLot dataset and original parking occupancy benchmark paper.
2. CNRPark-EXT dataset for cross-lot and weather-variant parking occupancy evaluation.
3. Ultralytics YOLOv8 documentation and model family.
4. Roboflow export tooling for full-frame parking annotation conversion.
