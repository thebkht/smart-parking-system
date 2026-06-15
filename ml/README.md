# ML Pipeline

Training, evaluation, export, and localization scripts for Smart Parking System.
The production model is the **Stage 2 classifier** (`YOLOv8-cls`) that labels a
perspective-warped `128×128` parking-space patch as `occupied` or `free`.

All commands assume the shared environment is active (`make install-dev &&
source .venv/bin/activate`) and run from the repo root. Each script also has a
`--help`. The `make` targets below wrap the scripts with sensible defaults; see
the [`Makefile`](../Makefile) for variables like `ACPDS_ROOT`, `STAGE2_VARIANT`,
and `DEVICE`.

## Canonical flow (Stage 2 classifier)

```bash
# 1. Extract perspective-warped patches from ACPDS, then validate them
make prepare-stage2 ACPDS_ROOT=/path/to/acpds PREP_STAGE2_ARGS="--run-validation"
make validate-stage2 ACPDS_ROOT=/path/to/acpds VALIDATE_STAGE2_ARGS="--validation-status passed"

# 2. Train (variant n / s / m), then evaluate on val and test splits
make train-stage2 STAGE2_VARIANT=n
make evaluate-stage2 STAGE2_SPLIT=val
make evaluate-stage2 STAGE2_SPLIT=test

# 3. Export to ONNX (FP32 / INT8) + Core ML INT8
make export-stage2

# 4. Measure the bandwidth saving vs. raw video
make bandwidth
```

ACPDS is not redistributed here — download it from the
[original project](https://github.com/martin-marek/parking-space-occupancy) and
point `ACPDS_ROOT` at it. The promoted checkpoint is published as a release
asset; fetch it with `make fetch-weights`.

## Scripts by purpose

**Dataset preparation**

- `extract_patches.py` — ACPDS quadrilaterals → warped `128×128` patches; the
  Stage 2 dataset builder (`--validate-only` for validation).
- `build_acpds_manifest.py` — build/repair the ACPDS split manifest.
- `prepare_dataset.py` — Stage 1 detection dataset and the single-model baseline
  (PKLot-based; legacy comparison paths).
- `patch_geometry.py` — `order_corners()` + perspective-warp helpers shared by
  extraction and the edge runtime.

**Training & evaluation**

- `train.py` — train Stage 1 detector, Stage 2 classifier, or the single-model
  baseline (`--stage2 --variant {n,s,m}`).
- `evaluate.py` — accuracy/precision/recall/F1 on a split; supports `--compare`,
  `--sweep`, and `--per-weather`.
- `compare_pooling.py` — quadrilateral warp vs. bounding-square crop comparison.
- `analyze_generalization.py`, `bucket_acpds_weather.py` — val/test gap and
  per-weather generalization analysis.

**Export & cost**

- `export.py` — export weights to ONNX (FP32/INT8) and Core ML INT8.
- `bandwidth.py` — compute the JSON-vs-video bandwidth reduction.

**Owner setup & Find My Car**

- `sfm_layout.py` — Structure-from-Motion → bird's-eye layout + spot polygons
  for a new camera (`make layout-sample`).
- `localize.py` — SIFT + FLANN + RANSAC photo-to-spot matching (`make
  localize-car`).
- `evaluate_localization.py` — localization accuracy evaluation.

**Reporting / misc**

- `predict.py` — run the classifier on a single image/source.
- `finalize.py` — generate the artifact manifest/summary.
- `make_report_figures.py`, `make_analysis_figures.py`, `report_style.py` —
  figure generation.

## Outputs

- Training runs land in `runs/` (gitignored).
- Exports and manifests land in `artifacts/` (gitignored).
- The promoted checkpoint lives at `acpds_cls/weights/best.pt` (fetched via
  `make fetch-weights`, gitignored).
