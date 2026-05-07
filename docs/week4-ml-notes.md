# Week 4 ML Notes

Week 4 should frame the two-stage full-frame pipeline as the main ML track:

`full-frame parking-space detector -> space crop -> occupancy classifier`

The repo also supports a single-model full-frame occupancy detector as an ML-only baseline, but it is not the deployed default path.

PKLot full-frame detection must be evaluated with scene holdout. Random image splits are not accepted as generalization evidence here.

PKLot full-frame detection should be treated carefully because Roboflow exports may include duplicate `.rf.*` variants and zero-label frames. Detection prep now deduplicates by normalized frame id, rebuilds train/val/test by scene group, excludes zero-label frames, and writes leakage plus annotation-audit details to the dataset report.

## Dataset Story

- primary source: PKLot Roboflow export with full-frame parking-space labels
- optional source: CNRPark-EXT patch folders merged into Stage 2 training
- canonical outputs: `stage2_data/`, `pklot_test/`, `cnrpark_test/`

## Recommended Stage 1 Training Path

```bash
make prepare-stage1 PKLOT_DIR=/path/to/pklot_roboflow
make train-stage1 STAGE1_VARIANT=s DEVICE=mps
make train-stage1 STAGE1_VARIANT=m DEVICE=mps TRAIN_STAGE1_ARGS="--imgsz 960"
make evaluate-stage1 STAGE1_VARIANT=s
```

## Stage 2 Training Path

```bash
make prepare-stage2 PKLOT_DIR=/path/to/pklot_roboflow
make train-stage2 STAGE2_VARIANT=n DEVICE=mps
make train-stage2 STAGE2_VARIANT=s DEVICE=mps
make train-stage2 STAGE2_VARIANT=m DEVICE=mps
```

## Other Supported ML Tracks

Single-model occupancy detector baseline:

```bash
make prepare-single-model PKLOT_DIR=/path/to/pklot_roboflow
```

Single-model training and evaluation do not currently have dedicated `make` targets. Keep using the direct `ml/train.py --single-model ...` and `ml/evaluate.py --single-model ...` commands for that older comparison path.

## Comparison Matrix

The main comparison is `yolov8n-cls` vs `yolov8s-cls` vs `yolov8m-cls` on:

- top-1 accuracy
- precision
- recall
- F1
- checkpoint size
- deployment latency / FPS

## Evaluation Path

```bash
make evaluate-stage1 STAGE1_VARIANT=s
make evaluate-stage2 STAGE2_VARIANT=n
```

The cross-dataset evaluation example still requires running `ml/evaluate.py` directly because the `Makefile` only covers the standard validation path.

Per-weather evaluation is only valid when the dataset is arranged as:

```text
<dataset-root>/sunny/<class>/*
<dataset-root>/cloudy/<class>/*
<dataset-root>/rainy/<class>/*
```

If that layout is missing, the repo should fail clearly instead of inventing weather labels.
