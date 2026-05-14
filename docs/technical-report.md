# Smart Parking System Technical Report

## Abstract

This project implements a two-stage smart parking system that performs inference on an edge device rather than streaming raw video to the cloud. The deployed pipeline uses a Stage 1 YOLO parking-space detector to localize parking spaces in full-frame images, then applies a Stage 2 YOLOv8 classification model to cropped space patches to predict `free` or `occupied`. The system outputs compact JSON payloads and stores them through a minimal FastAPI backend. In the checked-in final artifact set, the Stage 1 detector achieved up to 0.995 mAP@50 on a scene-held-out validation split, while the selected Stage 2 classifier (`yolov8m-cls`) reached 0.8958 top-1 accuracy and 0.8994 F1 at the default threshold of 0.5, improving to 0.9187 accuracy and 0.9242 F1 at the best offline threshold of 0.1. The runtime benchmark reached 350.8 FPS on Apple MPS and 2290.7 FPS for the exported Core ML INT8 artifact. Compared with a conservative 1080p H.264 camera stream, the JSON reporting path reduced bandwidth by 99.9%.

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
- Stage 2 classifier: `runs/acpds_cls/yolov8m_stage2/weights/best.pt`

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

## 5. Models and Training Configuration

### 5.1 Stage 1 detector

Stage 1 uses YOLO detection models trained on full-frame spot-localization data. The strongest saved Stage 1 validation row for the deployed `yolov8s_stage1` detector is:

- mAP@50: 0.9950
- mAP@50-95: 0.9603
- precision: 0.9971
- recall: 0.9970

The repo also contains a `yolov8n_stage1` run with similarly strong results:

- mAP@50: 0.9949
- mAP@50-95: 0.8864
- precision: 0.9948
- recall: 0.9911

For integrated runtime and documentation consistency, the final runbook uses the `yolov8s_stage1` checkpoint.

### 5.2 Stage 2 classifier

Stage 2 uses YOLOv8 classification models trained on cropped parking-spot patches. Three classifier sizes were compared:

| Model | Threshold | Accuracy | Precision | Recall | F1 | Size (MB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yolov8n_stage2` | 0.5 | 0.8708 | 0.9914 | 0.7784 | 0.8721 | 2.83 |
| `yolov8s_stage2` | 0.5 | 0.8804 | 0.9897 | 0.7969 | 0.8829 | 9.78 |
| `yolov8m_stage2` | 0.5 | 0.8958 | 0.9910 | 0.8233 | 0.8994 | 30.22 |
| `yolov8m_stage2` | 0.3 | 0.9085 | 0.9885 | 0.8483 | 0.9130 | 30.22 |

The `m` model provided the best F1 and overall accuracy among the saved comparison checkpoints, so it became the final selected classifier.

## 6. Evaluation Results

### 6.1 Stage 2 threshold sweep

Because the deployed edge path converts classifier probability into a binary occupancy decision, threshold selection matters. The saved sweep over `yolov8m_stage2` found the best validation result at threshold `0.1`:

- accuracy: 0.9187
- precision: 0.9783
- recall: 0.8758
- F1: 0.9242

The current edge configuration remains at threshold `0.3`, which matches the saved comparison run and provides a more conservative deployment choice for demos while still outperforming the default 0.5 threshold.

### 6.2 Cross-dataset evaluation

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

### 6.3 Per-weather evaluation

Weather-specific performance on `yolov8m_stage2` at threshold 0.5 is:

| Weather | Accuracy | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Sunny | 0.8525 | 0.9944 | 0.7559 | 0.8589 |
| Cloudy | 0.9261 | 0.9937 | 0.8643 | 0.9245 |
| Rainy | 0.9077 | 0.9950 | 0.8180 | 0.8979 |

Cloudy conditions produced the strongest saved result, while sunny conditions were the weakest. This suggests that harsh illumination and shadows may be more disruptive than moderate rain in the current patch-classification setup.

### 6.4 End-to-end stability

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

## 7. Runtime Benchmark and Bandwidth Results

### 7.1 Inference benchmark

The checked-in benchmark results for the exported final classifier path are:

| Backend | FPS | Latency (ms) | Model Size (MB) |
| --- | ---: | ---: | ---: |
| YOLO MPS | 350.8 | 2.9 | 31.7 |
| YOLO CPU | 263.9 | 3.8 | 31.7 |
| ONNX FP32 | 337.5 | 3.0 | 63.1 |
| Core ML INT8 | 2290.7 | 0.4 | 15.9 |

These numbers show that the patch-classification stage is well-suited for edge deployment. Even the plain PyTorch MPS path is fast enough for real-time parking updates, while the INT8 export provides very large additional headroom.

### 7.2 Bandwidth analysis

The saved bandwidth analysis compares JSON payload reporting against a conservative 1080p H.264 stream:

- JSON payload size: 280 bytes
- POST rate: 0.5 per second
- effective JSON throughput: 290 B/s including header overhead
- H.264 throughput assumption: 250 KB/s
- bandwidth savings: 99.9%
- reduction factor: 862x less data

This result directly supports the edge-computing rationale of the project. The system sends compact occupancy state rather than raw video, which substantially reduces bandwidth and improves privacy.

## 8. Discussion

The project achieved its main system goal: a working two-stage edge pipeline that performs local inference and exports only structured occupancy status. The strongest outcomes are:

- very strong Stage 1 scene-held-out localization performance
- clear Stage 2 improvement from `n` to `m` classifiers
- a threshold sweep showing measurable gains from deployment-aware calibration
- high runtime throughput on MPS and extremely high throughput for the INT8 export
- strong bandwidth savings relative to continuous video streaming

The main limitations are also clear.

First, Stage 2 cross-dataset and weather-specific results are lower than the in-domain validation results. The largest weakness is recall under domain shift, especially for occupied spots. Second, the saved stability test is short; a longer 30-minute or multi-hour soak test would provide stronger reliability evidence. Third, the current final artifact summary records the latest per-weather row rather than a full aggregated weather table, so the report must read the CSV itself to describe all weather conditions.

Another important detail is threshold policy. The best offline threshold for the final classifier is `0.1`, but the deployed config remains at `0.3`. This is reasonable for a demo-oriented edge system, but a production system would likely require threshold tuning using a deployment-specific validation set and cost-sensitive error analysis.

## 9. Conclusion

This project demonstrates that a laptop-based edge parking system can combine full-frame parking-space localization and per-space occupancy classification into a practical two-stage inference pipeline. The checked-in final system uses a YOLO Stage 1 parking-space detector and a `yolov8m-cls` Stage 2 classifier, achieves strong in-domain classification performance, maintains useful cross-dataset generalization, runs at real-time speeds on Apple MPS, and reduces bandwidth by more than 99.9% compared with continuous video transmission.

The final repo state is suitable for class demonstration and technical submission because it includes:

- trained checkpoints for both inference stages
- evaluation logs for model comparison, threshold sweep, cross-dataset testing, and per-weather analysis
- exported deployment artifacts
- an integrated edge runtime and backend
- regenerated acceptance summaries showing all tracked PRD checks as complete

## 10. References

1. PKLot dataset and original parking occupancy benchmark paper.
2. CNRPark-EXT dataset for cross-lot and weather-variant parking occupancy evaluation.
3. Ultralytics YOLOv8 documentation and model family.
4. Roboflow export tooling for full-frame parking annotation conversion.
