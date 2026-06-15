# Final Runbook

This runbook matches the final required architecture:

`parking-space quadrilateral -> perspective warp -> trained Stage 2 classifier -> smoothing -> JSON -> FastAPI`

Current checked-in final artifact choices:

- promoted Stage 2 classifier: `acpds_cls/weights/best.pt`
- comparison checkpoints: `runs/acpds_cls/yolov8n_stage2/weights/best.pt`, `runs/acpds_cls/yolov8s_stage2/weights/best.pt`, `runs/acpds_cls/yolov8m_stage2/weights/best.pt`
- exported deployment bundle: `artifacts/models/best.pt`, `artifacts/models/best.onnx`, `artifacts/models/best_int8.onnx`, `artifacts/models/best.mlpackage`
- optional generalized detector path: `runs/stage1_det/yolov8s_stage1/weights/best.pt`

## 1. Prepare Datasets

```bash
make prepare-stage2 ACPDS_ROOT=datasets/acpds_raw ACPDS_STAGE2_DIR=datasets/acpds_stage2
make validate-stage2 ACPDS_ROOT=datasets/acpds_raw ACPDS_STAGE2_DIR=datasets/acpds_stage2
```

Stage 2 prep reads ACPDS quadrilateral annotations, orders each corner set, and
writes perspective-corrected `128x128` free/occupied patches. `validate-stage2`
is the visual gate: inspect the saved patch samples before training or
presenting the final story.

The PKLot/CNRPark preparation targets are retained for older baselines and
related-work comparisons. They are not the canonical final demo path.

## 2. Train Models

Stage 2 classifier comparison:

```bash
make train-stage2 STAGE2_VARIANT=n DEVICE=mps
make train-stage2 STAGE2_VARIANT=s DEVICE=mps
make train-stage2 STAGE2_VARIANT=m DEVICE=mps
```

The promoted deployment checkpoint is the checked-in `YOLOv8n-cls` model at
`acpds_cls/weights/best.pt`. Larger `s` and `m` variants are kept for the
comparison table, but did not beat `n` on the final test split.

## 3. Evaluate

```bash
make evaluate-stage2 STAGE2_VARIANT=n STAGE2_SPLIT=test DEVICE=mps EVALUATE_STAGE2_ARGS="--batch 256"
make compare-stage2 DEVICE=mps COMPARE_STAGE2_ARGS="--batch 256"
make sweep-stage2 STAGE2_VARIANT=n DEVICE=mps SWEEP_STAGE2_ARGS="--batch 256"
make week7-eval DEVICE=mps
```

Use the saved Week 6/7 outputs for report tables unless you intentionally rerun
the full evaluation pass. The report-facing conclusion is that `YOLOv8n-cls`
reaches `0.9772` test accuracy / `0.9715` F1 while staying much smaller than
the larger variants and the ResNet50 baseline.

## 4. Export + Benchmark

```bash
make export-stage2 STAGE2_WEIGHTS=acpds_cls/weights/best.pt
make benchmark-stage2 STAGE2_WEIGHTS=acpds_cls/weights/best.pt BENCHMARK_IMAGE=samples/demo.jpg BENCHMARK_ROI="50 100 200 250"
make bandwidth
```

## 5. End-to-End Validation

Start backend:

```bash
make backend
```

The backend binds to `0.0.0.0:8000` by default. The Expo mobile app
auto-detects the host machine's IP from the Metro bundler; set
`EXPO_PUBLIC_API_BASE` in `frontend/mobile/.env.local` only if auto-detection
fails. Keep the backend terminal open; request failures from the mobile app are
logged with `[api]` and screen-specific tags such as `[OccupancyMap]`.

Run integrated demo:

```bash
make edge EDGE_ARGS="--image samples/demo.jpg --stage2-model acpds_cls/weights/best.pt --post --save-annotated logs/final-demo-annotated.jpg"
```

Run the short reproducible stability check used for the checked-in summary:

```bash
make stability BENCHMARK_IMAGE=samples/demo.jpg STAGE1_VARIANT=s STAGE2_WEIGHTS=acpds_cls/weights/best.pt STABILITY_DURATION=15 STABILITY_ARGS="--device mps --frame-interval 500"
```

For the formal Week 7 soak test, run `make stability STAGE1_VARIANT=s STAGE2_WEIGHTS=acpds_cls/weights/best.pt STABILITY_ARGS="--device mps"` to use the default `1800` second duration.

**Demo flow (platform split):**

1. **Publish layout from web** — open the web app, go to Owner Setup, upload lot photos or load the sample handoff; verify the map renders at the correct canvas scale.
2. **View live occupancy on both platforms** — start `make edge --post`; confirm spot colors update on both the web app and the phone simultaneously.
3. **Find My Car (mobile)** — on the phone, switch to the Find Car tab, take a photo near your parking spot, tap POST /park, then GET /find/{id}; verify the amber polygon appears at the correct coordinates on the map.

Mobile validation checklist:

- Launch: app opens on Live Map tab, fetches layout automatically, no Setup tab visible.
- Layout missing: shows "Lot layout not available — publish it from the web dashboard" + Retry button.
- Live Occupancy polls `GET /status` and scopes Free / Occupied counts to the active layout's spot IDs.
- Map: spot quads render at exact coordinates; pinch zooms 1×–5×; at zoom >1× one-finger pan moves the map.
- Find My Car posts a driver photo to `POST /park`, stores the returned `session_id`, then calls `GET /find/{session_id}`; amber quad highlights the found spot on the coordinate map.

## 6. Final Packaging

```bash
make finalize
```

This generates:

- `artifacts/final_manifest.json`
- `artifacts/final-artifact-summary.md`
