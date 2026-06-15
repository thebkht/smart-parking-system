# Architecture

This is the canonical architecture reference for Smart Parking System. If
another doc conflicts with this one, treat this file as authoritative.

## Pipeline

```
parking-space quadrilaterals → perspective warp → YOLOv8-cls → temporal smoothing → JSON → FastAPI
```

1. **Load spot quadrilaterals** — parking-space polygons come from `ACPDS`
   annotations or, for a new camera, from a Structure-from-Motion (SfM) layout
   generated during owner setup.
2. **Perspective warp** — each quadrilateral is ordered with `order_corners()`
   and warped to a clean `128×128` patch via `getPerspectiveTransform()` +
   `warpPerspective()`.
3. **Classify** — `YOLOv8-cls` labels each patch `occupied` or `free` with a
   confidence score.
4. **Temporal smoothing** — per-spot status is smoothed across frames to
   suppress flicker.
5. **Publish** — the edge node sends a compact JSON payload (see below) to the
   FastAPI backend instead of raw video, a ~99.9% bandwidth reduction versus
   H.264 streaming.

The same published layout drives the edge runtime, the web map, and the mobile
map, so all three stay consistent by construction.

See [prd-diagrams.md](prd-diagrams.md) for the Mermaid pipeline, system-overview,
and approach-evolution diagrams.

## Components

| Area | Path | Responsibility |
| --- | --- | --- |
| Edge inference | [`edge/detect.py`](../edge/detect.py) | Load polygons, warp, classify, smooth, POST |
| Backend | [`backend/main.py`](../backend/main.py) | FastAPI app, SQLite persistence, map/status/park APIs |
| Web app | [`frontend/src/`](../frontend/src) | Owner setup + live occupancy map (Leaflet) |
| Mobile app | [`frontend/mobile/`](../frontend/mobile) | Live occupancy map + Find My Car (SVG) |
| ML | [`ml/`](../ml) | Dataset extraction, training, evaluation, export, localization |

## Platform split

| Platform | Owner Setup | Live Occupancy Map | Find My Car |
| -------- | ----------- | ------------------ | ----------- |
| Web      | ✅          | ✅                 | —           |
| Mobile   | —           | ✅                 | ✅          |

- Owner Setup is web-only (`frontend/src/App.jsx`).
- Find My Car is mobile-only (`frontend/mobile/screens/FindMyCarScreen.js`).
- The web map renders spot quadrilaterals at exact image coordinates with
  Leaflet; the mobile map uses coordinate-accurate SVG polygons with
  pinch-to-zoom and pan (`react-native-svg` + `react-native-gesture-handler`).
  The `LotMap` component lives at `frontend/mobile/components/LotMap.js`.

## Occupancy payload contract

```json
{
  "spots": { "spot_1": "free", "spot_2": "occupied" },
  "confidence": { "spot_1": 0.91, "spot_2": 0.84 },
  "timestamp": "2026-04-21T00:00:00Z"
}
```

## Design notes

- The fixed-ROI path (rectangular crops from `config.yaml`) is a legacy fallback,
  not the canonical architecture. Prefer the quadrilateral-pooling path; do not
  assume fixed ROIs are the long-term direction just because older examples
  mention them.
- Preserve backward-compatible runtime behavior unless a change explicitly
  targets it.
- Trained weights are not committed to git; see the repository `README.md`
  "Model weights" section and [`../MODEL_LICENSE.md`](../MODEL_LICENSE.md).
