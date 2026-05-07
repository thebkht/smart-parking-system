# Edge Track

The edge pipeline follows the v3 architecture:

`static camera -> fixed ROIs -> crop each spot -> YOLOv8*-cls -> status smoothing -> JSON -> FastAPI`

## Default Behavior

- fixed ROI boxes are the default static-camera localization path
- Stage 2 classification is the required model path for normal use
- backend POSTs are enabled by default, so backend `/status` and `/history` update automatically unless `--no-post` is used
- smoothing stabilizes only the final occupied/free status
- confidence values are reported directly from the classifier and are not smoothed
- image mode and camera mode share the same inference and postprocess logic
- camera mode updates `logs/latest_frame.jpg` on every processed frame for backend MJPEG streaming
- camera mode tolerates short read glitches before stopping, which helps Continuity Camera sessions recover

## Stage 1 Parking-Space Detection

`edge/detect.py` supports a YOLO Stage 1 parking-space detector behind `--stage1-detector`. This is the repo's ML localization path for full-frame scenes. For a single fixed deployment camera, configured ROIs can still be simpler and more stable.

## Config

Copy [edge/config.example.yaml](config.example.yaml) to `edge/config.yaml` and adjust:

- `model.stage2_path` for the classifier checkpoint
- `stage1.detector_path` for the parking-space detector checkpoint
- `stage1.min_box_area` to reject tiny false-positive parking-space boxes such as road markings
- `stage1.filter_mode` to choose detector spatial filtering:
  `bounds` for general scenes, `roi_center` for a fixed camera with matched slot ROIs
- `input.frame_interval_ms` to control default camera processing cadence
- `backend.timeout_s` and `backend.retry_delay_s` to keep camera mode responsive when the backend is slow or unavailable
- `stream.max_width` and `stream.jpeg_quality` to control default MJPEG cost
- `postprocess.classifier_threshold` for deployed occupied/free thresholding
- `postprocess.smoothing_window` for temporal status smoothing
- `preprocess.perspective` to rectify an oblique camera view before ROI matching
- `rois` for camera-specific parking-space boxes

When `--stage1-detector` is enabled, the parking-space detector output is filtered before Stage 2:

- boxes smaller than `stage1.min_box_area` are discarded
- in `bounds` mode, boxes outside the union of the configured parking `rois` are discarded
- in `roi_center` mode, a box must also have its center inside one configured slot ROI

For angled cameras, configure `preprocess.perspective.source_points` and either
`destination_points` or `output_size` in [edge/config.example.yaml](config.example.yaml). The warp runs before Stage 1/Stage 2, so the configured `rois` should match the rectified image, not the raw camera frame.

## Commands

Static image:

```bash
python edge/detect.py \
  --image samples/demo.jpg \
  --stage2-model runs/stage2_cls/yolov8m_stage2/weights/best.pt \
  --save-annotated logs/annotated.jpg
```

Live camera:

```bash
python edge/detect.py \
  --camera 0 \
  --stage2-model runs/stage2_cls/yolov8m_stage2/weights/best.pt \
```

The default camera profile is intentionally reduced now: `frame_interval_ms: 500`, `stream.max_width: 640`, `stream.jpeg_quality: 55`, `backend.timeout_s: 0.75`, and `backend.retry_delay_s: 10.0`.

Override the streamed frame target if needed:

```bash
python edge/detect.py \
  --camera 0 \
  --stage2-model runs/stage2_cls/yolov8m_stage2/weights/best.pt \
  --latest-frame-path logs/latest_frame.jpg
```

If the live stream feels laggy, lower the stream cost without changing inference:

```bash
python edge/detect.py \
  --camera 0 \
  --stage2-model runs/stage2_cls/yolov8m_stage2/weights/best.pt \
  --stream-max-width 640 \
  --stream-jpeg-quality 55
```

macOS iPhone camera:

```bash
python edge/detect.py \
  --camera iphone \
  --stage2-model runs/stage2_cls/yolov8m_stage2/weights/best.pt \
```

`--camera iphone` is macOS-only and expects an available Continuity Camera / iPhone camera device.

Stage 1 parking-space detector:

```bash
python edge/detect.py \
  --image samples/demo.jpg \
  --stage1-detector \
  --stage1-model runs/stage1_det/yolov8s_stage1/weights/best.pt \
  --stage2-model runs/stage2_cls/yolov8m_stage2/weights/best.pt
```

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

Run without backend updates only when needed:

```bash
python edge/detect.py \
  --image samples/demo.jpg \
  --stage2-model runs/stage2_cls/yolov8m_stage2/weights/best.pt \
  --no-post
```
