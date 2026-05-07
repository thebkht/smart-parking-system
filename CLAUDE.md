# CLAUDE.md

This file provides repository-specific guidance for coding agents working in this project.

## Canonical Source

Use [docs/prd.md](docs/prd.md) as the canonical project definition.

Current product direction is `v6`, not the older `v3` summary:

- primary dataset: `ACPDS`
- Stage 1 extraction path: quadrilateral parking-space polygons
- Stage 2 classifier: `YOLOv8-cls`
- pooling method: `order_corners()` + `getPerspectiveTransform()` + `warpPerspective()`
- product scope: edge occupancy detection + backend + app + `Find My Car`

If another doc conflicts with `docs/prd.md`, treat the PRD as authoritative.

## Current Repo Reality

Some implementation docs still reflect the earlier `v3` architecture:

- [edge/README.md](edge/README.md)
- [backend/README.md](backend/README.md)

When editing code or docs:

- prefer the current PRD direction over older ROI-first descriptions
- do not assume fixed ROIs are the long-term architecture just because older docs mention them
- preserve backward-compatible runtime behavior unless the task explicitly changes it
- call out mismatches between implementation and PRD instead of silently rewriting scope

## Project Overview

Smart Parking System is an edge-based parking occupancy project built around a two-stage pipeline:

1. load or generate parking-space quadrilaterals
2. perspective-warp each spot into a clean `128 x 128` patch
3. classify each patch as `occupied` or `free` with `YOLOv8-cls`
4. temporally smooth status outputs
5. send compact JSON to the backend instead of raw video

The broader v6 product also includes:

- owner setup flow for lot layout creation
- backend persistence and map/status APIs
- web/mobile app flows
- `Find My Car` based on photo-to-spot localization

## Environment

Single shared `.venv` at the repo root:

```bash
make install-dev
source .venv/bin/activate
```

Quick verification:

```bash
python -c "import cv2, ultralytics, yaml; print('env ok')"
```

## Common Commands

Run the backend:

```bash
uvicorn backend.main:app --reload
```

Run tests:

```bash
pytest
```

Lint / format:

```bash
ruff check .
black .
```

Representative edge commands:

```bash
python edge/detect.py --image /path/to/parking.jpg
python edge/detect.py --image /path/to/parking.jpg --post
python edge/detect.py --image /path/to/parking.jpg --save-annotated logs/out.jpg
python edge/detect.py --image /path/to/parking.jpg --device cpu
python edge/detect.py --camera 0
```

Representative ML commands:

```bash
python ml/train.py --stage2 --variant n
python ml/train.py --stage2 --variant s
python ml/train.py --stage2 --variant m
python ml/evaluate.py --weights runs/stage2_cls/yolov8n_stage2/weights/best.pt --full
python ml/bandwidth.py
```

Note: command examples in the repo may still mention earlier PKLot/CNRPark or fixed-ROI workflows. Treat them as implementation history unless the current PRD or task says otherwise.

## Architecture Notes

Key files:

- [edge/detect.py](edge/detect.py): edge inference pipeline
- [backend/main.py](backend/main.py): FastAPI backend
- [docs/prd.md](docs/prd.md): canonical requirements
- [docs/prd-diagrams.md](docs/prd-diagrams.md): architecture diagrams

Preferred v6 architecture story:

`parking-space quadrilaterals -> perspective warp -> YOLOv8-cls -> temporal smoothing -> JSON -> FastAPI`

Preferred payload contract:

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

## Conventions

- keep project-level agent guidance in this file tracked in git
- do not put secrets, personal notes, or machine-specific private data here
- prefer updating shared docs over creating parallel one-off guidance
- model artifacts belong outside git-tracked source unless explicitly needed
- logs and local runtime outputs should remain untracked
