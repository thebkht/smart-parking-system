# Smart Parking System

Edge-based smart parking system using ACPDS quadrilateral pooling, `YOLOv8-cls`, FastAPI, and `Find My Car` product flows.

The canonical project definition is [docs/prd.md](docs/prd.md). This README summarizes the current `v6` direction.

## Architecture

Current pipeline:

`parking-space quadrilaterals -> perspective warp -> YOLOv8-cls -> temporal smoothing -> JSON -> FastAPI`

Key points:

- Stage 1 uses parking-space quadrilaterals from `ACPDS` annotations or future `SfM` layout generation
- Stage 2 classifies each perspective-corrected `128 x 128` patch as `occupied` or `free`
- only compact occupancy JSON is sent to the backend; raw video stays on the edge device
- the broader product scope includes owner setup, occupancy map views, and `Find My Car`

## Repo Layout

- [docs/prd.md](docs/prd.md): canonical PRD
- [docs/prd-diagrams.md](docs/prd-diagrams.md): architecture and comparison diagrams
- [edge/detect.py](edge/detect.py): edge inference pipeline
- [backend/main.py](backend/main.py): FastAPI backend
- [ml/](ml): training, evaluation, export, and benchmarking scripts
- [samples/](samples): sample images and videos

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

## Common Commands

Run the backend:

```bash
make backend
```

Run edge inference on an image:

```bash
make edge EDGE_ARGS="--image samples/photo_2026-04-23\ 21.29.16.jpeg"
```

Run live camera inference:

```bash
make edge EDGE_ARGS="--camera 0"
```

Train Stage 2 classifier variants:

```bash
make train-stage2 STAGE2_VARIANT=n
make train-stage2 STAGE2_VARIANT=s
make train-stage2 STAGE2_VARIANT=m
```

Evaluate a trained classifier:

```bash
make evaluate-stage2 STAGE2_VARIANT=n
```

Run tests:

```bash
make test
```

## Dataset Direction

The current PRD centers the project on `ACPDS`:

- 293 full parking-lot images
- 11,236 parking-space annotations
- quadrilateral spot geometry
- unseen-lot validation and test splits

This replaces the older repo story that emphasized PKLot/CNRPark and fixed ROI demos.

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

## Public Release Policy

- code, configs, metrics, and reproducible commands stay in the repo
- trained weights do not get committed into git history
- dataset archives, extracted datasets, runtime databases, and generated logs stay out of git

Model publishing guidance is in [MODEL_LICENSE.md](MODEL_LICENSE.md).

## Docs

- [docs/README.md](docs/README.md)
- [docs/prd.md](docs/prd.md)
- [docs/prd-diagrams.md](docs/prd-diagrams.md)
- [edge/README.md](edge/README.md)
- [backend/README.md](backend/README.md)

## Note on Older Docs

Some implementation docs and commands in the repo still reflect earlier `v3` assumptions such as fixed ROIs or PKLot/CNRPark-heavy workflows. When there is a mismatch, follow [docs/prd.md](docs/prd.md).
