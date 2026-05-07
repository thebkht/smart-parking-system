# Final Runbook

This runbook matches the final required architecture:

`trained Stage 1 detector -> crop -> trained Stage 2 classifier -> smoothing -> JSON -> FastAPI`

Current checked-in final artifact choices:

- Stage 1 detector for integrated runtime: `runs/stage1_det/yolov8s_stage1/weights/best.pt`
- strongest saved Stage 2 classifier: `runs/stage2_cls/yolov8m_stage2/weights/best.pt`
- exported deployment bundle: `artifacts/models/best.pt`, `artifacts/models/best.onnx`, `artifacts/models/best_int8.onnx`

## 1. Prepare Datasets

```bash
make prepare-stage1 PKLOT_DIR=datasets/pklot_raw
make prepare-stage2 PKLOT_DIR=datasets/pklot_raw
```

Add `--cnrpark-dir <path>` to the Stage 2 command when the CNRPark-EXT patch folders are available.
The Stage 2 prep command accepts either the official `cnrpark.it` `PATCHES/` + `LABELS/` archive layout or pre-flattened `free/` / `occupied/` folders. When weather labels are present it also writes `datasets/stage2_weather/` for per-weather evaluation.

## 2. Train Models

Stage 1 detector:

```bash
make train-stage1 STAGE1_VARIANT=s DEVICE=mps
```

Stage 2 classifier comparison:

```bash
make train-stage2 STAGE2_VARIANT=n DEVICE=mps
make train-stage2 STAGE2_VARIANT=s DEVICE=mps
make train-stage2 STAGE2_VARIANT=m DEVICE=mps
```

## 3. Evaluate

```bash
make evaluate-stage1 STAGE1_VARIANT=s DEVICE=mps
make evaluate-stage2 STAGE2_VARIANT=m DEVICE=mps EVALUATE_STAGE2_ARGS="--batch 256"
make compare-stage2 DEVICE=mps COMPARE_STAGE2_ARGS="--batch 256"
make sweep-stage2 STAGE2_VARIANT=m DEVICE=mps SWEEP_STAGE2_ARGS="--batch 256"
```

For cross-dataset and per-weather evaluation, the current `Makefile` does not have dedicated targets yet. Use `make` for the standard validation path above, and run `ml/evaluate.py` directly only for these extra modes.

The saved sweep artifact currently identifies `0.1` as the best validation threshold for `yolov8m_stage2`.
The deployed edge config remains at `0.3`, which is the threshold used in the saved model-comparison and runtime demo path.

## 4. Export + Benchmark

```bash
make export-stage2 STAGE2_VARIANT=m
make benchmark-stage2 STAGE2_VARIANT=m BENCHMARK_IMAGE=samples/demo.jpg BENCHMARK_ROI="50 100 200 250"
make bandwidth
```

## 5. End-to-End Validation

Start backend:

```bash
make backend
```

Run integrated demo:

```bash
make edge EDGE_ARGS="--image samples/demo.jpg --stage1-detector --stage1-model runs/stage1_det/yolov8s_stage1/weights/best.pt --stage2-model runs/stage2_cls/yolov8m_stage2/weights/best.pt --post --save-annotated logs/final-demo-annotated.jpg"
```

Run the short reproducible stability check used for the checked-in summary:

```bash
make stability BENCHMARK_IMAGE=samples/demo.jpg STAGE1_VARIANT=s STAGE2_VARIANT=m STABILITY_DURATION=15 STABILITY_ARGS="--device mps --frame-interval 500"
```

For the formal Week 7 soak test, run `make stability STAGE1_VARIANT=s STAGE2_VARIANT=m STABILITY_ARGS="--device mps"` to use the default `1800` second duration.

## 6. Final Packaging

```bash
make finalize
```

This generates:

- `artifacts/final_manifest.json`
- `docs/final-artifact-summary.md`
