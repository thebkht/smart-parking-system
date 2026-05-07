# Week 4 Demo Runbook

The Week 4 demo should already reflect the v3 architecture, even if it uses a placeholder classifier checkpoint:

`static image -> fixed ROIs -> per-spot crop -> classifier -> JSON -> backend`

## Deliverables

- `edge/detect.py` running the two-stage path with fixed ROIs by default
- `backend/main.py` accepting and storing the v3 payload
- one known-good static parking image
- optional annotated output image saved before class

## Start The Backend

```bash
make backend
```

## Run The Static-Image Demo

```bash
make edge EDGE_ARGS="--image samples/demo.jpg --stage2-model yolov8n-cls.pt --post --save-annotated logs/week4-demo-annotated.jpg"
```

## What To Show

1. the source parking image
2. the ROI boxes and per-spot predictions on the annotated output
3. terminal JSON in the v3 payload shape
4. optional backend `/status` response showing the same payload

## Demo Payload Example

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
