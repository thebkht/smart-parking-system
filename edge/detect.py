#!/usr/bin/env python3
"""Two-stage edge inference for the v3 smart parking pipeline.

Default path:
  static camera -> fixed ROIs -> crop each spot -> YOLOv8-cls -> smoothing -> JSON

ML Stage 1 path:
  enable --stage1-detector to run a full-frame parking-space detector before
  Stage 2 occupancy classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import cv2
import numpy as np
import requests
import yaml
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.patch_geometry import box_to_corners, order_corners, warp_patch

STAGE2_MODEL_DEFAULT = "acpds_cls/weights/best.pt"
DEFAULT_STAGE1_LOCAL_CHECKPOINT = "runs/stage1_det/yolov8s_stage1/weights/best.pt"
STAGE1_MODEL_DEFAULT = "yolov8n.pt"
DEFAULT_BACKEND_URL = "http://127.0.0.1:8000/update"
DEFAULT_BACKEND_TIMEOUT_S = 0.75
DEFAULT_BACKEND_RETRY_DELAY_S = 10.0
DEFAULT_SMOOTH_N = 5
DEFAULT_FRAME_INTERVAL_MS = 500
DEFAULT_POST_INTERVAL_S = 2.0
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_LATEST_FRAME_PATH = DEFAULT_LOG_DIR / "latest_frame.jpg"
DEFAULT_LOG_FORMAT = "json"
DEFAULT_STREAM_JPEG_QUALITY = 55
DEFAULT_STREAM_MAX_WIDTH = 640
DEFAULT_STAGE2_THRESHOLD = 0.5
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
DEFAULT_CAMERA_PROBE_LIMIT = 10
DEFAULT_STAGE2_LOCAL_CHECKPOINT = "acpds_cls/weights/best.pt"
DEFAULT_STAGE1_MIN_BOX_AREA = 1500
DEFAULT_STAGE1_FILTER_MODE = "bounds"
DEFAULT_STAGE1_CONFIDENCE = 0.4
DEFAULT_STAGE1_POSTPROCESS_TYPE = "nmm"
DEFAULT_STAGE1_MATCH_THRESHOLD = 0.3
DEFAULT_CAMERA_READ_RETRY_LIMIT = 30        # was 10 — give ~6s before giving up
DEFAULT_CAMERA_REOPEN_LIMIT = 5             # was 3 — more reopen attempts before fallback
DEFAULT_BUILTIN_CAMERA_LABEL = "built-in"
DEFAULT_IPHONE_CAMERA_LABEL = "iphone"
DEFAULT_CAMERA_PROBE_MISS_STREAK_LIMIT = 2
STAGE1_OCCUPY_THRESHOLD = 0.45

DEFAULT_ROIS: Dict[str, Tuple[int, int, int, int]] = {
    "spot_1": (50, 100, 200, 250),
    "spot_2": (210, 100, 360, 250),
    "spot_3": (370, 100, 520, 250),
    "spot_4": (530, 100, 680, 250),
}

PerspectiveTransform = Tuple[np.ndarray, Tuple[int, int]]
SpotBox = Tuple[int, int, int, int]
SpotGeometry = np.ndarray
SpotGeometries = Dict[str, SpotGeometry]


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v3 smart parking edge pipeline. Fixed ROIs remain the "
            "static-camera path; --stage1-detector enables a parking-space "
            "detector for full-frame localization."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", help="Path to a parking image (static mode).")
    group.add_argument("--video", help="Path to a parking video file.")
    group.add_argument(
        "--camera",
        metavar="SOURCE",
        help="Camera source: numeric index like 0 or the macOS-only value 'iphone'.",
    )

    parser.add_argument(
        "--stage2-model",
        default=None,
        help="Primary Stage 2 classifier checkpoint. Defaults to config or yolov8n-cls.pt.",
    )
    parser.add_argument(
        "--stage1-model",
        default=None,
        help="Stage 1 parking-space detector checkpoint used with --stage1-detector.",
    )
    parser.add_argument(
        "--stage1-detector",
        action="store_true",
        help="Use the Stage 1 parking-space detector instead of fixed ROIs.",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help=(
            "Skip Stage 2 classifier entirely. A Stage 1 detection box is "
            "treated as occupied; undetected spots are free. "
            "Useful when Stage 2 domain-shifts badly (e.g. steep overhead angle)."
        ),
    )
    parser.add_argument("--device", default=None, help="Inference device. Default from config.")
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument(
        "--backend-timeout",
        type=float,
        default=DEFAULT_BACKEND_TIMEOUT_S,
        help="HTTP timeout in seconds for backend POSTs.",
    )
    parser.add_argument(
        "--backend-retry-delay",
        type=float,
        default=DEFAULT_BACKEND_RETRY_DELAY_S,
        help="Seconds to wait before retrying backend POST after a failure.",
    )
    parser.add_argument(
        "--post",
        action="store_true",
        default=None,
        help="POST payloads to the backend. Enabled by default for the integrated pipeline.",
    )
    parser.add_argument(
        "--no-post",
        action="store_false",
        dest="post",
        help="Disable backend POSTs and run detect in local-only mode.",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        default=False,
        help="Open an OpenCV window showing the annotated frame (demo mode). Press q or close the window to quit.",
    )
    parser.add_argument(
        "--save-annotated",
        help=(
            "Path to save annotated output. In image mode this saves an image; "
            "in video/camera mode this saves an annotated video."
        ),
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--smooth-n",
        type=int,
        default=None,
        help="Temporal smoothing window in frames.",
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=None,
        metavar="MS",
        help="Target milliseconds between frames in camera mode.",
    )
    parser.add_argument(
        "--post-interval",
        type=float,
        default=None,
        metavar="SEC",
        help="Seconds between backend POSTs in camera mode.",
    )
    parser.add_argument(
        "--loop-video",
        action="store_true",
        help="Loop the input video file when it reaches the end.",
    )
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--log-format", choices=["csv", "json"], default=None)
    parser.add_argument(
        "--latest-frame-path",
        default=str(DEFAULT_LATEST_FRAME_PATH),
        help="Path to the latest annotated JPEG written for backend MJPEG streaming.",
    )
    parser.add_argument(
        "--stream-jpeg-quality",
        type=int,
        default=DEFAULT_STREAM_JPEG_QUALITY,
        help="JPEG quality for the MJPEG latest-frame file (1-100). Lower is faster.",
    )
    parser.add_argument(
        "--stream-max-width",
        type=int,
        default=DEFAULT_STREAM_MAX_WIDTH,
        help="Resize streamed frames down to this width before JPEG encoding. Use 0 to disable.",
    )
    parser.add_argument(
        "--stage2-threshold",
        type=float,
        default=None,
        help="Occupied confidence threshold used by the edge classifier.",
    )
    parser.add_argument(
        "--stage1-imgsz",
        type=int,
        default=None,
        help="Inference image size for stage1 detector (default: 1280).",
    )
    parser.add_argument(
        "--stage1-sahi",
        action="store_true",
        default=True,
        help="Use SAHI sliced inference for stage1 (default: enabled).",
    )
    parser.add_argument(
        "--no-stage1-sahi",
        action="store_false",
        dest="stage1_sahi",
        help="Disable SAHI sliced inference for stage1.",
    )
    parser.add_argument(
        "--stage1-slice-size",
        type=int,
        default=None,
        help="SAHI slice size in pixels (default: 640).",
    )
    parser.add_argument(
        "--stage1-overlap",
        type=float,
        default=None,
        help="SAHI slice overlap ratio (default: 0.2).",
    )
    parser.add_argument(
        "--stage1-min-box-area",
        type=int,
        default=None,
        help="Minimum Stage 1 detection box area in pixels squared.",
    )
    parser.add_argument(
        "--stage1-filter-mode",
        choices=["none", "bounds", "roi_center"],
        default=None,
        help="Stage 1 spatial filtering mode: none, lot bounds, or ROI-center gating.",
    )
    parser.add_argument(
        "--dump-warp-samples",
        default=None,
        help="Optional directory to save side-by-side rectangular crops vs 128x128 quad warps for visual QA.",
    )
    parser.add_argument(
        "--dump-warp-limit",
        type=int,
        default=5,
        help="Maximum number of warped spot samples to save per input source.",
    )
    parser.add_argument(
        "--camera-read-retry-limit",
        type=int,
        default=DEFAULT_CAMERA_READ_RETRY_LIMIT,
        help="How many consecutive camera read failures to tolerate before stopping.",
    )
    parser.add_argument(
        "--camera-reopen-limit",
        type=int,
        default=DEFAULT_CAMERA_REOPEN_LIMIT,
        help="How many times camera mode may reopen the capture session after repeated read failures.",
    )
    return parser.parse_args()


def resolve_settings(args: argparse.Namespace, cfg: dict) -> argparse.Namespace:
    model_cfg = cfg.get("model", {})
    input_cfg = cfg.get("input", {})
    post_cfg = cfg.get("postprocess", {})
    log_cfg = cfg.get("logging", {})
    stream_cfg = cfg.get("stream", {})
    backend_cfg = cfg.get("backend", {})
    stage1_cfg = cfg.get("stage1", {})

    if args.stage2_model is None:
        args.stage2_model = model_cfg.get("stage2_path", STAGE2_MODEL_DEFAULT)
    if args.stage1_model is None:
        args.stage1_model = stage1_cfg.get("detector_path", STAGE1_MODEL_DEFAULT)
    if args.device is None:
        args.device = model_cfg.get("device", "mps")
    if args.smooth_n is None:
        args.smooth_n = post_cfg.get("smoothing_window", DEFAULT_SMOOTH_N)
    if args.frame_interval is None:
        args.frame_interval = input_cfg.get("frame_interval_ms", DEFAULT_FRAME_INTERVAL_MS)
    if args.post_interval is None:
        args.post_interval = input_cfg.get("post_interval_s", DEFAULT_POST_INTERVAL_S)
    if args.log_dir is None:
        args.log_dir = log_cfg.get("output_dir", str(DEFAULT_LOG_DIR))
    if args.log_format is None:
        args.log_format = log_cfg.get("format", DEFAULT_LOG_FORMAT)
    if args.backend_timeout == DEFAULT_BACKEND_TIMEOUT_S:
        args.backend_timeout = backend_cfg.get("timeout_s", DEFAULT_BACKEND_TIMEOUT_S)
    if args.backend_retry_delay == DEFAULT_BACKEND_RETRY_DELAY_S:
        args.backend_retry_delay = backend_cfg.get(
            "retry_delay_s", DEFAULT_BACKEND_RETRY_DELAY_S
        )
    if args.post is None:
        args.post = backend_cfg.get("enabled", True)
    if args.stream_jpeg_quality == DEFAULT_STREAM_JPEG_QUALITY:
        args.stream_jpeg_quality = stream_cfg.get(
            "jpeg_quality", DEFAULT_STREAM_JPEG_QUALITY
        )
    if args.stream_max_width == DEFAULT_STREAM_MAX_WIDTH:
        args.stream_max_width = stream_cfg.get(
            "max_width", DEFAULT_STREAM_MAX_WIDTH
        )
    if args.stage2_threshold is None:
        args.stage2_threshold = post_cfg.get(
            "classifier_threshold", DEFAULT_STAGE2_THRESHOLD
        )
    if args.stage1_slice_size is None:
        args.stage1_slice_size = stage1_cfg.get("slice_size", 640)
    if args.stage1_overlap is None:
        args.stage1_overlap = stage1_cfg.get("overlap", 0.2)
    if args.stage1_imgsz is None:
        args.stage1_imgsz = stage1_cfg.get("imgsz", 1280)
    if args.stage1_sahi:
        args.stage1_sahi = stage1_cfg.get("use_sahi", True)
    if args.stage1_min_box_area is None:
        args.stage1_min_box_area = stage1_cfg.get("min_box_area", DEFAULT_STAGE1_MIN_BOX_AREA)
    if args.stage1_filter_mode is None:
        args.stage1_filter_mode = stage1_cfg.get("filter_mode", DEFAULT_STAGE1_FILTER_MODE)
    args.stage1_postprocess_type = str(
        stage1_cfg.get("postprocess_type", DEFAULT_STAGE1_POSTPROCESS_TYPE)
    ).strip().lower()
    args.stage1_match_threshold = float(
        stage1_cfg.get("match_threshold", DEFAULT_STAGE1_MATCH_THRESHOLD)
    )

    configured_stage2_path = Path(str(args.stage2_model))
    if not configured_stage2_path.exists():
        local_stage2_checkpoint = Path(DEFAULT_STAGE2_LOCAL_CHECKPOINT)
        if local_stage2_checkpoint.exists():
            args.stage2_model = str(local_stage2_checkpoint)
            print(
                f"Stage 2 model not found at {configured_stage2_path}; "
                f"using {local_stage2_checkpoint}."
            )
    configured_stage1_path = Path(str(args.stage1_model))
    if args.stage1_detector and not configured_stage1_path.exists():
        local_stage1_checkpoint = Path(DEFAULT_STAGE1_LOCAL_CHECKPOINT)
        if local_stage1_checkpoint.exists():
            args.stage1_model = str(local_stage1_checkpoint)
            print(
                f"Stage 1 model not found at {configured_stage1_path}; "
                f"using {local_stage1_checkpoint}."
            )
    return args


def normalize_rois(raw_rois: dict | None) -> Dict[str, Tuple[int, int, int, int]]:
    rois = raw_rois or DEFAULT_ROIS
    normalized: Dict[str, Tuple[int, int, int, int]] = {}
    for spot_id, coords in rois.items():
        if not isinstance(coords, (list, tuple)) or len(coords) != 4:
            raise ValueError(f"ROI for {spot_id!r} must be [x1, y1, x2, y2].")
        x1, y1, x2, y2 = (int(v) for v in coords)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"ROI for {spot_id!r} has invalid bounds: {coords}")
        normalized[str(spot_id)] = (x1, y1, x2, y2)
    return normalized


def fetch_rois_from_backend(
    backend_url: str = "http://127.0.0.1:8000",
    timeout: float = 3.0,
) -> SpotGeometries:
    """Fetch spot polygons from GET /map without collapsing them to rectangles."""
    try:
        response = requests.get(f"{backend_url}/map", timeout=timeout)
        response.raise_for_status()
        data = response.json()
        rois: SpotGeometries = {}
        for spot in data.get("spots", []):
            spot_id = spot["spot_id"]
            points = spot.get("points") or spot.get("corners")
            if not points:
                continue
            rois[str(spot_id)] = order_corners(points)
        if rois:
            print(f"Loaded {len(rois)} spot geometries from backend /map.")
            return rois
        print("Backend /map returned no spots, falling back to config ROIs.")
    except Exception as exc:
        print(f"Could not fetch ROIs from backend: {exc}. Using config ROIs.")
    return {}


def normalize_spot_geometries(raw_rois: dict | None) -> SpotGeometries:
    normalized_boxes = normalize_rois(raw_rois)
    return {spot_id: box_to_corners(box) for spot_id, box in normalized_boxes.items()}


def geometry_to_box(corners: np.ndarray) -> SpotBox:
    ordered = order_corners(corners)
    xs = ordered[:, 0]
    ys = ordered[:, 1]
    return (
        int(np.floor(float(xs.min()))),
        int(np.floor(float(ys.min()))),
        int(np.ceil(float(xs.max()))),
        int(np.ceil(float(ys.max()))),
    )


def geometries_to_boxes(geometries: SpotGeometries) -> Dict[str, SpotBox]:
    return {spot_id: geometry_to_box(corners) for spot_id, corners in geometries.items()}


def load_spot_geometries(config_path: Path, backend_url: str = "http://127.0.0.1:8000") -> SpotGeometries:
    rois = fetch_rois_from_backend(backend_url)
    if rois:
        return rois
    cfg = load_config(config_path)
    return normalize_spot_geometries(cfg.get("rois"))


def load_rois(config_path: Path, backend_url: str = "http://127.0.0.1:8000") -> Dict[str, Tuple[int, int, int, int]]:
    """Legacy box view over spot geometries for tests and ROI-only code paths."""
    return geometries_to_boxes(load_spot_geometries(config_path, backend_url=backend_url))


def _normalize_points(raw_points: Any, label: str) -> np.ndarray:
    if not isinstance(raw_points, (list, tuple)) or len(raw_points) != 4:
        raise ValueError(f"{label} must contain exactly 4 [x, y] points.")
    points: list[Tuple[float, float]] = []
    for point in raw_points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError(f"{label} entries must be [x, y] pairs.")
        x, y = point
        points.append((float(x), float(y)))
    return np.array(points, dtype=np.float32)


def load_perspective_transform(cfg: dict) -> Optional[PerspectiveTransform]:
    preprocess_cfg = cfg.get("preprocess", {})
    perspective_cfg = preprocess_cfg.get("perspective")
    if not perspective_cfg:
        return None

    source_points = _normalize_points(perspective_cfg.get("source_points"), "preprocess.perspective.source_points")
    destination_points_raw = perspective_cfg.get("destination_points")
    output_size_raw = perspective_cfg.get("output_size")

    if destination_points_raw is None and output_size_raw is None:
        raise ValueError(
            "Perspective preprocessing requires either destination_points or output_size."
        )

    if destination_points_raw is not None:
        destination_points = _normalize_points(
            destination_points_raw,
            "preprocess.perspective.destination_points",
        )
        if output_size_raw is None:
            max_x = int(np.ceil(float(destination_points[:, 0].max())))
            max_y = int(np.ceil(float(destination_points[:, 1].max())))
            output_size = (max_x, max_y)
        else:
            if not isinstance(output_size_raw, (list, tuple)) or len(output_size_raw) != 2:
                raise ValueError("preprocess.perspective.output_size must be [width, height].")
            output_size = (int(output_size_raw[0]), int(output_size_raw[1]))
    else:
        if not isinstance(output_size_raw, (list, tuple)) or len(output_size_raw) != 2:
            raise ValueError("preprocess.perspective.output_size must be [width, height].")
        width, height = (int(output_size_raw[0]), int(output_size_raw[1]))
        destination_points = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype=np.float32,
        )
        output_size = (width, height)

    width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("preprocess.perspective.output_size values must be positive.")

    matrix = cv2.getPerspectiveTransform(source_points, destination_points)
    return matrix, (width, height)


def apply_perspective_transform(
    frame: np.ndarray,
    transform: Optional[PerspectiveTransform],
) -> np.ndarray:
    if transform is None:
        return frame
    matrix, (width, height) = transform
    return cv2.warpPerspective(frame, matrix, (width, height))


class SmoothingBuffer:
    """Per-spot majority-vote smoothing over string statuses."""

    def __init__(self, window: int = DEFAULT_SMOOTH_N) -> None:
        self._window = max(1, int(window))
        self._history: Dict[str, deque[str]] = {}

    def update(self, statuses: Dict[str, str]) -> None:
        for spot_id, status in statuses.items():
            history = self._history.setdefault(spot_id, deque(maxlen=self._window))
            history.append(status)

    def get_status(self) -> Dict[str, str]:
        smoothed: Dict[str, str] = {}
        for spot_id, history in self._history.items():
            occupied_count = sum(1 for status in history if status == "occupied")
            smoothed[spot_id] = (
                "occupied" if occupied_count > len(history) / 2 else "free"
            )
        return smoothed

    def reset(self) -> None:
        for history in self._history.values():
            history.clear()


# ---------------------------------------------------------------------------
# Core ML helpers
# ---------------------------------------------------------------------------

def _is_coreml(model_path: str) -> bool:
    return str(model_path).endswith(".mlpackage") or str(model_path).endswith(".mlmodel")


def _load_coreml(model_path: str) -> Any:
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError(
            "coremltools is required for Core ML inference. "
            "Install it with: pip install coremltools"
        )
    return ct.models.MLModel(model_path)


def _classify_coreml(
    model: Any,
    patch: np.ndarray,
    imgsz: int,
    threshold: float,
    class_names: Dict[int, str],
) -> Tuple[str, float]:
    """Run a single patch through a Core ML classifier. Returns (status, confidence)."""
    import PIL.Image

    # Cache input metadata once to avoid repeated spec parsing
    if not hasattr(model, "_cached_input_name"):
        _inp = model.get_spec().description.input[0]
        model._cached_input_name = _inp.name
        _img = _inp.type.imageType
        model._cached_input_w = _img.width or imgsz
        model._cached_input_h = _img.height or imgsz

    resized = cv2.resize(patch, (model._cached_input_w, model._cached_input_h))
    pil_img = PIL.Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
    output = model.predict({model._cached_input_name: pil_img})

    # Output shape: classLabel (str), classLabel_probs (dict), var_NNN (ndarray)
    # Use classLabel_probs when available; fall back to any dict in output.
    probs: Dict[str, float] = {}
    if "classLabel_probs" in output and isinstance(output["classLabel_probs"], dict):
        probs = output["classLabel_probs"]
    else:
        for v in output.values():
            if isinstance(v, dict):
                probs = v
                break

    if probs:
        label = max(probs, key=lambda k: probs[k])
        confidence = float(probs[label])
    elif "classLabel" in output:
        label = str(output["classLabel"])
        confidence = 1.0
    else:
        return "free", 0.0

    status = "occupied" if label == "occupied" and confidence >= threshold else "free"
    return status, confidence


# ---------------------------------------------------------------------------
# Stage 1 / Stage 2 model loading & inference
# ---------------------------------------------------------------------------

def get_spot_boxes(
    frame: np.ndarray,
    fixed_rois: Dict[str, Tuple[int, int, int, int]],
    stage1_model: Optional[YOLO],
    device: str,
    use_stage1_detector: bool,
    use_sahi: bool = False,
    stage1_imgsz: int = 1280,
    slice_size: int = 640,
    overlap: float = 0.2,
    min_box_area: int = DEFAULT_STAGE1_MIN_BOX_AREA,
    filter_mode: str = DEFAULT_STAGE1_FILTER_MODE,
    postprocess_type: str = DEFAULT_STAGE1_POSTPROCESS_TYPE,
    match_threshold: float = DEFAULT_STAGE1_MATCH_THRESHOLD,
) -> Dict[str, Tuple[int, int, int, int]]:
    if not use_stage1_detector:
        return fixed_rois
    if stage1_model is None:
        raise ValueError("Stage 1 parking-space detector requested but no model was loaded.")

    # Stage 1 is always a YOLO .pt — Core ML only applies to stage 2
    yolo_device = device if device != "mps" else "cpu"
    lot_mask = roi_bounds(fixed_rois) if fixed_rois and fixed_rois != DEFAULT_ROIS else None
    roi_boxes = list(fixed_rois.values()) if fixed_rois else []

    if use_sahi:
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            import tempfile, os

            sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=stage1_model.ckpt_path,
                confidence_threshold=DEFAULT_STAGE1_CONFIDENCE,
                device=yolo_device,
            )
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, frame)
            result = get_sliced_prediction(
                tmp_path,
                sahi_model,
                slice_height=slice_size,
                slice_width=slice_size,
                overlap_height_ratio=overlap,
                overlap_width_ratio=overlap,
                verbose=0,
            )
            os.unlink(tmp_path)
            candidates: list[Tuple[Tuple[int, int, int, int], float]] = []
            for pred in result.object_prediction_list:
                b = pred.bbox
                kept = filter_stage1_box(
                    frame.shape,
                    (int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)),
                    lot_mask=lot_mask,
                    roi_boxes=roi_boxes,
                    min_box_area=min_box_area,
                    filter_mode=filter_mode,
                )
                if kept is None:
                    continue
                score = float(getattr(pred.score, "value", 1.0))
                candidates.append((kept, score))
            return consolidate_stage1_boxes(
                candidates,
                postprocess_type=postprocess_type,
                match_threshold=match_threshold,
            )
        except ImportError:
            print("SAHI not installed; falling back to standard inference.")

    results = stage1_model(
        frame,
        device=yolo_device,
        verbose=False,
        imgsz=stage1_imgsz,
        conf=DEFAULT_STAGE1_CONFIDENCE,
    )[0]
    candidates: list[Tuple[Tuple[int, int, int, int], float]] = []
    raw_boxes = results.boxes.xyxy.cpu().numpy()
    box_conf = getattr(results.boxes, "conf", None)
    raw_scores = box_conf.cpu().numpy() if box_conf is not None else None
    for index, box in enumerate(raw_boxes):
        kept = filter_stage1_box(
            frame.shape,
            tuple(box.astype(int)),
            lot_mask=lot_mask,
            roi_boxes=roi_boxes,
            min_box_area=min_box_area,
            filter_mode=filter_mode,
        )
        if kept is None:
            continue
        score = float(raw_scores[index]) if raw_scores is not None else 1.0
        candidates.append((kept, score))
    return consolidate_stage1_boxes(
        candidates,
        postprocess_type=postprocess_type,
        match_threshold=match_threshold,
    )


def clip_box(
    frame_shape: Tuple[int, int, int],
    box: Tuple[int, int, int, int],
) -> Optional[Tuple[int, int, int, int]]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def roi_bounds(rois: Dict[str, Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
    if not rois:
        return None
    return (
        min(box[0] for box in rois.values()),
        min(box[1] for box in rois.values()),
        max(box[2] for box in rois.values()),
        max(box[3] for box in rois.values()),
    )


def box_area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    union = box_area(box_a) + box_area(box_b) - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _axis_overlap_ratio(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    intersection = max(0, min(end_a, end_b) - max(start_a, start_b))
    span = min(max(1, end_a - start_a), max(1, end_b - start_b))
    return intersection / span


def _box_center(box: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _merge_boxes(boxes: Iterable[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    box_list = list(boxes)
    return (
        min(box[0] for box in box_list),
        min(box[1] for box in box_list),
        max(box[2] for box in box_list),
        max(box[3] for box in box_list),
    )


def _should_merge_stage1_boxes(
    box_a: Tuple[int, int, int, int],
    box_b: Tuple[int, int, int, int],
    *,
    match_threshold: float,
    median_width: float,
    median_height: float,
) -> bool:
    iou = box_iou(box_a, box_b)
    if iou >= match_threshold:
        return True

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    horizontal_overlap = _axis_overlap_ratio(ax1, ax2, bx1, bx2)
    vertical_overlap = _axis_overlap_ratio(ay1, ay2, by1, by2)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    smaller_area = min(box_area(box_a), box_area(box_b))
    inter_smaller = (inter_area / smaller_area) if smaller_area > 0 else 0.0
    if inter_smaller >= max(0.5, match_threshold):
        return True

    acx, acy = _box_center(box_a)
    bcx, bcy = _box_center(box_b)
    center_dx = abs(acx - bcx)
    center_dy = abs(acy - bcy)
    width_gate = max(8.0, median_width * 0.35)
    height_gate = max(8.0, median_height * 0.35)
    if horizontal_overlap >= 0.55 and center_dx <= width_gate and center_dy <= height_gate:
        return True
    if vertical_overlap >= 0.8 and center_dx <= width_gate * 0.6:
        return True
    return False


def consolidate_stage1_boxes(
    candidates: Iterable[Tuple[Tuple[int, int, int, int], float]],
    *,
    postprocess_type: str,
    match_threshold: float,
) -> Dict[str, Tuple[int, int, int, int]]:
    candidate_list = list(candidates)
    if not candidate_list:
        return {}

    if postprocess_type == "none":
        return {
            f"spot_{index}": box
            for index, (box, _score) in enumerate(candidate_list, start=1)
        }

    widths = [max(1, box[2] - box[0]) for box, _score in candidate_list]
    heights = [max(1, box[3] - box[1]) for box, _score in candidate_list]
    median_width = float(np.median(widths))
    median_height = float(np.median(heights))

    ordered = sorted(candidate_list, key=lambda item: item[1], reverse=True)
    merged_groups: list[list[Tuple[Tuple[int, int, int, int], float]]] = []
    for box, score in ordered:
        assigned = False
        for group in merged_groups:
            merged_box = _merge_boxes(group_box for group_box, _group_score in group)
            if _should_merge_stage1_boxes(
                box,
                merged_box,
                match_threshold=match_threshold,
                median_width=median_width,
                median_height=median_height,
            ):
                group.append((box, score))
                assigned = True
                break
        if not assigned:
            merged_groups.append([(box, score)])

    merged_candidates: list[Tuple[Tuple[int, int, int, int], float]] = []
    for group in merged_groups:
        if postprocess_type == "nms":
            best_box, best_score = max(group, key=lambda item: item[1])
            merged_candidates.append((best_box, best_score))
            continue
        merged_box = _merge_boxes(box for box, _score in group)
        merged_score = max(score for _box, score in group)
        merged_candidates.append((merged_box, merged_score))

    merged_candidates.sort(key=lambda item: (item[0][1], item[0][0]))
    return {
        f"spot_{index}": box
        for index, (box, _score) in enumerate(merged_candidates, start=1)
    }


def filter_stage1_box(
    frame_shape: Tuple[int, int, int],
    box: Tuple[int, int, int, int],
    *,
    lot_mask: Optional[Tuple[int, int, int, int]],
    roi_boxes: Optional[Iterable[Tuple[int, int, int, int]]] = None,
    min_box_area: int,
    filter_mode: str = DEFAULT_STAGE1_FILTER_MODE,
) -> Optional[Tuple[int, int, int, int]]:
    clipped = clip_box(frame_shape, box)
    if clipped is None:
        return None
    if box_area(clipped) < max(0, int(min_box_area)):
        return None
    if filter_mode == "none":
        return clipped
    if lot_mask is not None:
        lot = clip_box(frame_shape, lot_mask)
        if lot is None:
            return None
        lot_x1, lot_y1, lot_x2, lot_y2 = lot
        x1, y1, x2, y2 = clipped
        if x1 < lot_x1 or y1 < lot_y1 or x2 > lot_x2 or y2 > lot_y2:
            return None
    if filter_mode == "roi_center" and roi_boxes:
        x1, y1, x2, y2 = clipped
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        center_in_roi = False
        for roi in roi_boxes:
            clipped_roi = clip_box(frame_shape, roi)
            if clipped_roi is None:
                continue
            rx1, ry1, rx2, ry2 = clipped_roi
            if rx1 <= cx < rx2 and ry1 <= cy < ry2:
                center_in_roi = True
                break
        if not center_in_roi:
            return None
    return clipped


def crop_patch(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    clipped = clip_box(frame.shape, box)
    if clipped is None:
        return None
    x1, y1, x2, y2 = clipped
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    return patch


def _result_label_confidence(result: Any) -> Tuple[str, float]:
    top1 = int(result.probs.top1)
    label = str(result.names[top1])
    confidence = float(result.probs.top1conf)
    return label, confidence


def classify_patch(
    frame: np.ndarray,
    corners: np.ndarray,
    model: Union[YOLO, Any],
    device: str,
    threshold: float,
) -> Tuple[str, float]:
    try:
        patch = warp_patch(frame, corners, size=128)
    except ValueError:
        return "free", 0.0

    # Core ML path — model is a ct.models.MLModel, not a YOLO instance
    if device == "coreml":
        return _classify_coreml(model, patch, imgsz=128, threshold=threshold, class_names={})

    # Stage 2 was trained on 128x128 warped patches; pass the aligned patch directly.
    result = model(patch, device=device, verbose=False)[0]
    label, confidence = _result_label_confidence(result)
    status = "occupied" if label == "occupied" and confidence >= threshold else "free"
    return status, confidence


def build_payload(
    statuses: Dict[str, str],
    confidences: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "spots": dict(sorted(statuses.items())),
        "confidence": {k: round(v, 3) for k, v in sorted(confidences.items())},
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def annotate_frame(
    frame: np.ndarray,
    statuses: Dict[str, str],
    spot_geometries: SpotGeometries,
    confidences: Dict[str, float],
) -> np.ndarray:
    annotated = frame.copy()
    for spot_id, status in statuses.items():
        corners = spot_geometries.get(spot_id)
        if corners is None:
            continue
        box = clip_box(frame.shape, geometry_to_box(corners))
        if box is None:
            continue
        x1, y1, x2, y2 = box
        color = (255, 0, 255) if status == "occupied" else (47, 255, 173)
        text_color = (255, 255, 255) if status == "occupied" else (0, 0, 0)
        confidence = confidences.get(spot_id, 0.0)
        polygon = np.round(order_corners(corners)).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [polygon], isClosed=True, color=color, thickness=1)
        label = f"{int(confidence * 100)}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        tx, ty = x1, y1 - 6
        cv2.rectangle(annotated, (tx - 1, ty - th - 1), (tx + tw + 1, ty + baseline), color, -1)
        cv2.putText(annotated, label, (tx, ty), font, font_scale, text_color, thickness, cv2.LINE_AA)
    return annotated


def dump_warp_samples(
    frame: np.ndarray,
    spot_geometries: SpotGeometries,
    output_dir: Path,
    *,
    source_name: str,
    limit: int = 5,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    dumped = 0
    for spot_id, corners in sorted(spot_geometries.items()):
        if dumped >= max(1, int(limit)):
            break
        box = clip_box(frame.shape, geometry_to_box(corners))
        if box is None:
            continue
        rect_crop = crop_patch(frame, box)
        if rect_crop is None:
            continue
        try:
            quad_warp = warp_patch(frame, corners, size=128)
        except ValueError:
            continue

        rect_vis = cv2.resize(rect_crop, (128, 128), interpolation=cv2.INTER_LINEAR)
        canvas = np.zeros((128, 256, 3), dtype=np.uint8)
        canvas[:, :128] = rect_vis
        canvas[:, 128:] = quad_warp
        cv2.putText(canvas, "rect", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, "warp", (134, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        filename = f"{Path(source_name).stem}__{spot_id}.jpg"
        cv2.imwrite(str(output_dir / filename), canvas)
        dumped += 1
    return dumped


def log_result(payload: Dict[str, Any], log_dir: Path, log_format: str) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"parking_log_{datetime.now().strftime('%Y-%m-%d')}.{log_format}"
    if log_format == "json":
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
        return

    fieldnames = ["timestamp", *payload["spots"].keys()]
    write_header = not log_path.exists()
    row = {"timestamp": payload["timestamp"], **payload["spots"]}
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _resize_for_stream(frame: np.ndarray, max_width: int) -> np.ndarray:
    if max_width <= 0:
        return frame
    height, width = frame.shape[:2]
    if width <= max_width:
        return frame
    scale = max_width / float(width)
    resized_height = max(1, int(round(height * scale)))
    return cv2.resize(frame, (max_width, resized_height), interpolation=cv2.INTER_AREA)


def write_latest_frame(
    frame: np.ndarray,
    output_path: Path,
    jpeg_quality: int = DEFAULT_STREAM_JPEG_QUALITY,
    max_width: int = DEFAULT_STREAM_MAX_WIDTH,
) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stream_frame = _resize_for_stream(frame, max_width)
    quality = max(1, min(100, int(jpeg_quality)))
    success, encoded = cv2.imencode(
        ".jpg",
        stream_frame,
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )
    if not success:
        return False
    tmp_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    tmp_path.write_bytes(encoded.tobytes())
    os.replace(tmp_path, output_path)
    return True


def _resolve_video_output_path(save_annotated: Optional[str], default_name: str) -> Optional[Path]:
    if not save_annotated:
        return None
    output_path = Path(save_annotated)
    if output_path.suffix:
        return output_path
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path / default_name


def _make_partial_video_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.partial{output_path.suffix}")


def _video_writer_candidates(output_path: Path) -> Tuple[Tuple[str, Path], ...]:
    suffix = output_path.suffix.lower()
    mp4_candidates = (
        ("avc1", output_path.with_suffix(".mp4")),
        ("mp4v", output_path.with_suffix(".mp4")),
    )
    avi_candidates = (
        ("MJPG", output_path.with_suffix(".avi")),
        ("XVID", output_path.with_suffix(".avi")),
    )
    if suffix == ".avi":
        return avi_candidates + mp4_candidates
    if suffix in {".mp4", ".m4v", ".mov"}:
        return mp4_candidates + avi_candidates
    return mp4_candidates + avi_candidates


def create_video_writer(
    output_path: Path,
    frame_size: Tuple[int, int],
    fps: float,
) -> Tuple[cv2.VideoWriter, Path, Path, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_fps = fps if fps and fps > 0 else 1.0
    writer: Optional[cv2.VideoWriter] = None
    opened_path: Optional[Path] = None
    partial_path: Optional[Path] = None
    selected_fourcc: Optional[str] = None
    for fourcc_name, candidate_output_path in _video_writer_candidates(output_path):
        candidate_partial_path = _make_partial_video_path(candidate_output_path)
        candidate = cv2.VideoWriter(
            str(candidate_partial_path),
            cv2.VideoWriter_fourcc(*fourcc_name),
            normalized_fps,
            frame_size,
        )
        if candidate.isOpened():
            writer = candidate
            opened_path = candidate_output_path
            partial_path = candidate_partial_path
            selected_fourcc = fourcc_name
            break
        candidate.release()
    if writer is None:
        raise RuntimeError(f"Could not create video writer for: {output_path}")
    if opened_path is None or partial_path is None or selected_fourcc is None:
        raise RuntimeError(f"Could not determine output path for: {output_path}")
    return writer, opened_path, partial_path, selected_fourcc


def finalize_video_output(partial_path: Optional[Path], final_path: Optional[Path]) -> None:
    if partial_path is None or final_path is None or not partial_path.exists():
        return
    final_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path.replace(final_path)


def post_payload(payload: Dict[str, Any], backend_url: str, timeout: float) -> None:
    response = requests.post(backend_url, json=payload, timeout=timeout)
    response.raise_for_status()



def get_spot_boxes_with_scores(
    frame: np.ndarray,
    stage1_model: "YOLO",
    device: str,
    use_sahi: bool = False,
    stage1_imgsz: int = 1280,
    slice_size: int = 640,
    overlap: float = 0.2,
    min_box_area: int = DEFAULT_STAGE1_MIN_BOX_AREA,
    filter_mode: str = DEFAULT_STAGE1_FILTER_MODE,
    postprocess_type: str = DEFAULT_STAGE1_POSTPROCESS_TYPE,
    match_threshold: float = DEFAULT_STAGE1_MATCH_THRESHOLD,
    fixed_rois: Optional[Dict[str, Tuple[int, int, int, int]]] = None,
) -> Tuple[Dict[str, Tuple[int, int, int, int]], Dict[str, float]]:
    """Run stage1 and return (spot_boxes, confidences_per_spot).

    Confidence per spot is the max stage1 score of detections consolidated
    into that spot.  Used by --stage1-only mode to infer occupancy directly.
    """
    yolo_device = device if device != "coreml" else "cpu"
    lot_mask = roi_bounds(fixed_rois) if fixed_rois else None
    roi_boxes_list = list(fixed_rois.values()) if fixed_rois else []

    raw_candidates: list[Tuple[Tuple[int, int, int, int], float]] = []

    if use_sahi:
        try:
            from sahi import AutoDetectionModel
            from sahi.predict import get_sliced_prediction
            import tempfile, os as _os

            sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=stage1_model.ckpt_path,
                confidence_threshold=DEFAULT_STAGE1_CONFIDENCE,
                device=yolo_device,
            )
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            import cv2 as _cv2
            _cv2.imwrite(tmp_path, frame)
            result = get_sliced_prediction(
                tmp_path, sahi_model,
                slice_height=slice_size, slice_width=slice_size,
                overlap_height_ratio=overlap, overlap_width_ratio=overlap,
                verbose=0,
            )
            _os.unlink(tmp_path)
            for pred in result.object_prediction_list:
                b = pred.bbox
                kept = filter_stage1_box(
                    frame.shape,
                    (int(b.minx), int(b.miny), int(b.maxx), int(b.maxy)),
                    lot_mask=lot_mask, roi_boxes=roi_boxes_list,
                    min_box_area=min_box_area, filter_mode=filter_mode,
                )
                if kept is not None:
                    raw_candidates.append((kept, float(getattr(pred.score, "value", 1.0))))
        except ImportError:
            print("SAHI not installed; falling back to standard inference.")
            use_sahi = False

    if not use_sahi:
        results = stage1_model(
            frame, device=yolo_device, verbose=False,
            imgsz=stage1_imgsz, conf=DEFAULT_STAGE1_CONFIDENCE,
        )[0]
        raw_boxes  = results.boxes.xyxy.cpu().numpy()
        raw_scores = results.boxes.conf.cpu().numpy() if results.boxes.conf is not None else None
        for idx, box in enumerate(raw_boxes):
            kept = filter_stage1_box(
                frame.shape, tuple(box.astype(int)),
                lot_mask=lot_mask, roi_boxes=roi_boxes_list,
                min_box_area=min_box_area, filter_mode=filter_mode,
            )
            if kept is not None:
                score = float(raw_scores[idx]) if raw_scores is not None else 1.0
                raw_candidates.append((kept, score))

    spot_boxes = consolidate_stage1_boxes(
        raw_candidates,
        postprocess_type=postprocess_type,
        match_threshold=match_threshold,
    )

    # Build per-spot confidence: max raw score whose box overlaps the consolidated box
    spot_scores: Dict[str, float] = {}
    for spot_id, sbox in spot_boxes.items():
        best = 0.0
        for raw_box, raw_score in raw_candidates:
            if box_iou(sbox, raw_box) > 0.05:
                best = max(best, raw_score)
        spot_scores[spot_id] = round(best, 3)

    return spot_boxes, spot_scores

def run_pipeline(
    frame: np.ndarray,
    fixed_geometries: SpotGeometries,
    stage1_model: Optional[YOLO],
    stage2_model: Union[YOLO, Any],
    smooth_buf: SmoothingBuffer,
    args: argparse.Namespace,
    last_confidences: Optional[Dict[str, float]] = None,  # ← add this
) -> Tuple[Dict[str, Any], SpotGeometries]:
    stage1_only = getattr(args, "stage1_only", False)
    stage1_postprocess_type = getattr(args, "stage1_postprocess_type", DEFAULT_STAGE1_POSTPROCESS_TYPE)
    stage1_match_threshold = getattr(args, "stage1_match_threshold", DEFAULT_STAGE1_MATCH_THRESHOLD)
    fixed_rois = geometries_to_boxes(fixed_geometries)

    if stage1_only and args.stage1_detector and stage1_model is not None:
        # Skip stage2: treat every stage1 detection as "occupied"
        spot_boxes, spot_scores = get_spot_boxes_with_scores(
            frame=frame,
            stage1_model=stage1_model,
            device=args.device,
            use_sahi=args.stage1_sahi,
            stage1_imgsz=args.stage1_imgsz,
            slice_size=args.stage1_slice_size,
            overlap=args.stage1_overlap,
            min_box_area=args.stage1_min_box_area,
            filter_mode=args.stage1_filter_mode,
            postprocess_type=stage1_postprocess_type,
            match_threshold=stage1_match_threshold,
            fixed_rois=fixed_rois,
        )
        raw_statuses: Dict[str, str] = {
            sid: ("occupied" if spot_scores.get(sid, 0) >= STAGE1_OCCUPY_THRESHOLD else "free")
            for sid in spot_boxes
        }
        confidences: Dict[str, float] = spot_scores
        spot_geometries = {spot_id: box_to_corners(box) for spot_id, box in spot_boxes.items()}
    else:
        if args.stage1_detector:
            spot_boxes = get_spot_boxes(
                frame=frame,
                fixed_rois=fixed_rois,
                stage1_model=stage1_model,
                device=args.device,
                use_stage1_detector=args.stage1_detector,
                use_sahi=args.stage1_sahi,
                stage1_imgsz=args.stage1_imgsz,
                slice_size=args.stage1_slice_size,
                overlap=args.stage1_overlap,
                min_box_area=args.stage1_min_box_area,
                filter_mode=args.stage1_filter_mode,
                postprocess_type=stage1_postprocess_type,
                match_threshold=stage1_match_threshold,
            )
            spot_geometries = {spot_id: box_to_corners(box) for spot_id, box in spot_boxes.items()}
        else:
            spot_geometries = fixed_geometries

        raw_statuses: Dict[str, str] = {}
        confidences: Dict[str, float] = {}
        for spot_id, corners in sorted(spot_geometries.items()):
            status, confidence = classify_patch(
                frame=frame,
                corners=corners,
                model=stage2_model,
                device=args.device,
                threshold=args.stage2_threshold,
            )
            raw_statuses[spot_id] = status
            confidences[spot_id] = confidence

    smooth_buf.update(raw_statuses)
    if last_confidences is not None:
        merged = {**last_confidences, **confidences}
        last_confidences.clear()
        last_confidences.update(merged)
        confidences = merged

    payload = build_payload(smooth_buf.get_status(), confidences)
    return payload, spot_geometries


def create_models(args: argparse.Namespace) -> Tuple[Optional[YOLO], Union[YOLO, Any]]:
    stage1_model = YOLO(args.stage1_model) if args.stage1_detector else None

    # Stage 2: load via coremltools if .mlpackage, otherwise ultralytics YOLO
    if _is_coreml(args.stage2_model):
        stage2_model = _load_coreml(args.stage2_model)
        # Force device to coreml so classify_patch takes the right branch
        args.device = "coreml"
    else:
        stage2_model = YOLO(args.stage2_model)

    return stage1_model, stage2_model


def _collect_camera_names(node: Any) -> Iterable[str]:
    if isinstance(node, dict):
        for key, value in node.items():
            lowered = str(key).lower()
            if lowered in {"_name", "name", "spcamera_name"} and isinstance(value, str):
                yield value
            yield from _collect_camera_names(value)
    elif isinstance(node, list):
        for item in node:
            yield from _collect_camera_names(item)


def _has_iphone_camera(camera_data: dict) -> bool:
    keywords = ("iphone", "continuity")
    for name in _collect_camera_names(camera_data):
        lowered = name.lower()
        if any(keyword in lowered for keyword in keywords):
            return True
    return False


def _macos_camera_backend() -> int:
    return int(getattr(cv2, "CAP_AVFOUNDATION", 0))


def _probe_camera_index(index: int, backend: Optional[int] = None) -> bool:
    if backend in (None, 0):
        capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index, backend)
    try:
        if not capture.isOpened():
            return False
        ok, _frame = capture.read()
        return bool(ok)
    finally:
        capture.release()


def open_camera_capture(index: int, backend: Optional[int] = None):
    if backend in (None, 0):
        return cv2.VideoCapture(index)
    return cv2.VideoCapture(index, backend)


def reopen_camera_capture(index: int, backend: Optional[int], current_capture) -> Any:
    try:
        current_capture.release()
    except Exception:
        pass
    return open_camera_capture(index, backend)


def _resolve_macos_iphone_camera(
    probe_limit: int = DEFAULT_CAMERA_PROBE_LIMIT,
) -> Tuple[int, Optional[int]]:
    result = subprocess.run(
        ["system_profiler", "SPCameraDataType", "-json"],
        capture_output=True,
        text=True,
        check=True,
    )
    camera_data = json.loads(result.stdout or "{}")
    if not _has_iphone_camera(camera_data):
        raise ValueError(
            "No iPhone camera detected. Connect or enable Continuity Camera in macOS and try again."
        )

    backend = _macos_camera_backend()
    candidates: list[int] = []
    miss_streak = 0
    for index in range(max(1, int(probe_limit))):
        if _probe_camera_index(index, backend):
            candidates.append(index)
            miss_streak = 0
            continue
        if candidates:
            miss_streak += 1
            if miss_streak >= DEFAULT_CAMERA_PROBE_MISS_STREAK_LIMIT:
                break
    if not candidates:
        raise RuntimeError(
            "Detected an iPhone camera but could not open any AVFoundation camera index."
        )

    # Continuity Camera commonly appears after the built-in webcam in AVFoundation.
    # Prefer the highest working index instead of the first openable camera.
    iphone_index = candidates[-1]
    builtin_index = candidates[0] if candidates[0] != iphone_index else None
    return iphone_index, builtin_index


def resolve_camera_source(source: str, probe_limit: int = DEFAULT_CAMERA_PROBE_LIMIT) -> int:
    value = str(source).strip()
    if value.isdigit():
        return int(value)
    if value.lower() != "iphone":
        raise ValueError("Invalid --camera value. Use a numeric index like '0' or 'iphone'.")
    if platform.system() != "Darwin":
        raise ValueError("The iPhone camera option is supported only on macOS.")
    iphone_index, _builtin_index = _resolve_macos_iphone_camera(probe_limit)
    return iphone_index


def resolve_camera_runtime(source: str) -> Tuple[int, Optional[int], Optional[int], Optional[int], str]:
    value = str(source).strip()
    if value.isdigit():
        return int(value), None, None, None, value
    if value.lower() != DEFAULT_IPHONE_CAMERA_LABEL:
        raise ValueError("Invalid --camera value. Use a numeric index like '0' or 'iphone'.")
    if platform.system() != "Darwin":
        raise ValueError("The iPhone camera option is supported only on macOS.")

    iphone_index, builtin_index = _resolve_macos_iphone_camera()
    backend = _macos_camera_backend()
    fallback_backend = backend if builtin_index is not None else None
    return iphone_index, backend, builtin_index, fallback_backend, DEFAULT_IPHONE_CAMERA_LABEL


def handoff_to_builtin_camera(index: Optional[int], backend: Optional[int]) -> None:
    if index is None:
        return
    capture = open_camera_capture(index, backend)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    try:
        if capture.isOpened():
            capture.read()
    finally:
        capture.release()


def run_inference(args: argparse.Namespace, fixed_geometries: SpotGeometries) -> int:
    import glob

    input_path = Path(args.image)
    if not input_path.exists():
        raise FileNotFoundError(f"Image path not found: {input_path}")

    # Collect image paths — support both a single file and a directory
    IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tiff", "*.tif")
    if input_path.is_dir():
        image_paths: list[Path] = []
        for ext in IMAGE_EXTS:
            image_paths.extend(sorted(input_path.glob(ext)))
            image_paths.extend(sorted(input_path.glob(ext.upper())))
        # Deduplicate while preserving order
        seen: set[Path] = set()
        image_paths = [p for p in image_paths if not (p in seen or seen.add(p))]
        if not image_paths:
            raise ValueError(f"No images found in directory: {input_path}")
        print(f"Found {len(image_paths)} image(s) in {input_path}")
    else:
        image_paths = [input_path]

    stage1_model, stage2_model = create_models(args)
    smooth_buf = SmoothingBuffer(window=args.smooth_n)

    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print(f"Warning: OpenCV could not read image, skipping: {image_path}")
            continue
        frame = apply_perspective_transform(frame, getattr(args, "perspective_transform", None))

        # Reset smoothing buffer per image in batch mode so frames don't bleed together
        if len(image_paths) > 1:
            smooth_buf.reset()

        payload, spot_geometries = run_pipeline(frame, fixed_geometries, stage1_model, stage2_model, smooth_buf, args)

        print(f"\n--- {image_path.name} ---")
        print(json.dumps(payload, indent=2))
        log_result(payload, Path(args.log_dir), args.log_format)

        if args.save_annotated:
            save_dir = Path(args.save_annotated)
            save_dir.mkdir(parents=True, exist_ok=True)
            # Directory input → save each file under its original name in save_dir
            # Single file input → honour the exact path the user provided
            if input_path.is_dir():
                out_file = save_dir / image_path.name
            else:
                out_file = save_dir if save_dir.suffix else save_dir / image_path.name
            cv2.imwrite(
                str(out_file),
                annotate_frame(frame, payload["spots"], spot_geometries, payload["confidence"]),
            )
            print(f"Annotated image saved to: {out_file}")
        if args.dump_warp_samples:
            dumped = dump_warp_samples(
                frame,
                spot_geometries,
                Path(args.dump_warp_samples),
                source_name=image_path.name,
                limit=args.dump_warp_limit,
            )
            print(f"Warp QA samples saved: {dumped}")

        if args.post:
            try:
                post_payload(payload, args.backend_url, args.backend_timeout)
            except requests.RequestException as exc:
                print(f"Backend POST failed: {exc}")

    return 0

import threading
import queue

def run_camera(args: argparse.Namespace, fixed_geometries: SpotGeometries) -> int:
    stage1_model, stage2_model = create_models(args)
    smooth_buf = SmoothingBuffer(window=args.smooth_n)
    latest_frame_path = Path(args.latest_frame_path)
    video_writer: Optional[cv2.VideoWriter] = None
    output_video_path = _resolve_video_output_path(args.save_annotated, "annotated_camera.mp4")
    video_partial_path: Optional[Path] = None
    video_final_path: Optional[Path] = None

    # Must open on the main thread for AVFoundation / Continuity Camera
    cap = open_camera_capture(args.camera, getattr(args, "camera_backend", None))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    last_post = time.perf_counter() - args.post_interval
    backend_retry_after = 0.0
    last_confidences: Dict[str, float] = {}
    consecutive_failures = 0
    warp_dumped = False

    if getattr(args, "display", False):
        cv2.namedWindow("Smart Parking — live", cv2.WINDOW_NORMAL)

    try:
        while True:
            # Grab without decoding to flush stale buffered frames
            for _ in range(2):
                cap.grab()

            ok, frame = cap.retrieve()
            if not ok:
                consecutive_failures += 1
                if consecutive_failures >= args.camera_read_retry_limit:
                    print("Too many consecutive failures, stopping.")
                    break
                time.sleep(0.05)
                continue
            consecutive_failures = 0
            frame = apply_perspective_transform(frame, getattr(args, "perspective_transform", None))

            started_at = time.perf_counter()
            payload, spot_geometries = run_pipeline(
                frame, fixed_geometries, stage1_model, stage2_model, smooth_buf, args, last_confidences
            )
            annotated = annotate_frame(frame, payload["spots"], spot_geometries, payload["confidence"])
            if args.dump_warp_samples and not warp_dumped:
                dump_warp_samples(
                    frame,
                    spot_geometries,
                    Path(args.dump_warp_samples),
                    source_name="camera_frame",
                    limit=args.dump_warp_limit,
                )
                warp_dumped = True
            if getattr(args, "display", False):
                cv2.imshow("Smart Parking — live", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if output_video_path is not None:
                if video_writer is None:
                    frame_height, frame_width = annotated.shape[:2]
                    fps = 1000.0 / max(1, int(args.frame_interval))
                    video_writer, video_final_path, video_partial_path, selected_fourcc = create_video_writer(
                        output_video_path,
                        (frame_width, frame_height),
                        fps,
                    )
                    print(
                        f"Annotated video will be saved to: {video_final_path} "
                        f"(codec={selected_fourcc})"
                    )
                video_writer.write(annotated)
            write_latest_frame(
                annotated,
                latest_frame_path,
                jpeg_quality=args.stream_jpeg_quality,
                max_width=args.stream_max_width,
            )

            now = time.perf_counter()
            if now - last_post >= args.post_interval:
                print(json.dumps(payload))
                log_result(payload, Path(args.log_dir), args.log_format)
                if args.post and now >= backend_retry_after:
                    try:
                        post_payload(payload, args.backend_url, args.backend_timeout)
                        backend_retry_after = 0.0
                    except requests.RequestException as exc:
                        backend_retry_after = now + args.backend_retry_delay
                        print(f"Backend POST failed: {exc}")
                last_post = now

            elapsed = time.perf_counter() - started_at
            sleep_s = max(0.0, args.frame_interval / 1000.0 - elapsed)
            if sleep_s:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
            finalize_video_output(video_partial_path, video_final_path)
        if getattr(args, "display", False):
            cv2.destroyAllWindows()
        if getattr(args, "requested_camera", None) == DEFAULT_IPHONE_CAMERA_LABEL:
            handoff_to_builtin_camera(
                getattr(args, "fallback_camera", None),
                getattr(args, "fallback_camera_backend", None),
            )
    return 0


def run_video(args: argparse.Namespace, fixed_geometries: SpotGeometries) -> int:
    video_path = Path(str(args.video))
    if not video_path.exists():
        raise FileNotFoundError(f"Video path not found: {video_path}")

    stage1_model, stage2_model = create_models(args)
    smooth_buf = SmoothingBuffer(window=args.smooth_n)
    latest_frame_path = Path(args.latest_frame_path)
    output_video_path = _resolve_video_output_path(
        args.save_annotated,
        f"{video_path.stem}_annotated.mp4",
    )
    video_writer: Optional[cv2.VideoWriter] = None
    video_partial_path: Optional[Path] = None
    video_final_path: Optional[Path] = None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    last_post = time.perf_counter() - args.post_interval
    backend_retry_after = 0.0
    last_confidences: Dict[str, float] = {}
    warp_dumped = False

    if getattr(args, "display", False):
        cv2.namedWindow("Smart Parking — live", cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    smooth_buf.reset()
                    last_confidences.clear()
                    continue
                break

            frame = apply_perspective_transform(frame, getattr(args, "perspective_transform", None))

            started_at = time.perf_counter()
            payload, spot_geometries = run_pipeline(
                frame, fixed_geometries, stage1_model, stage2_model, smooth_buf, args, last_confidences
            )
            annotated = annotate_frame(frame, payload["spots"], spot_geometries, payload["confidence"])
            if args.dump_warp_samples and not warp_dumped:
                dump_warp_samples(
                    frame,
                    spot_geometries,
                    Path(args.dump_warp_samples),
                    source_name=video_path.name,
                    limit=args.dump_warp_limit,
                )
                warp_dumped = True
            if getattr(args, "display", False):
                cv2.imshow("Smart Parking — live", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if output_video_path is not None:
                if video_writer is None:
                    frame_height, frame_width = annotated.shape[:2]
                    video_writer, video_final_path, video_partial_path, selected_fourcc = create_video_writer(
                        output_video_path,
                        (frame_width, frame_height),
                        source_fps,
                    )
                    print(
                        f"Annotated video will be saved to: {video_final_path} "
                        f"(codec={selected_fourcc})"
                    )
                video_writer.write(annotated)
            write_latest_frame(
                annotated,
                latest_frame_path,
                jpeg_quality=args.stream_jpeg_quality,
                max_width=args.stream_max_width,
            )

            now = time.perf_counter()
            if now - last_post >= args.post_interval:
                print(json.dumps(payload))
                log_result(payload, Path(args.log_dir), args.log_format)
                if args.post and now >= backend_retry_after:
                    try:
                        post_payload(payload, args.backend_url, args.backend_timeout)
                        backend_retry_after = 0.0
                    except requests.RequestException as exc:
                        backend_retry_after = now + args.backend_retry_delay
                        print(f"Backend POST failed: {exc}")
                last_post = now

            elapsed = time.perf_counter() - started_at
            sleep_s = max(0.0, args.frame_interval / 1000.0 - elapsed)
            if sleep_s:
                time.sleep(sleep_s)

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
            finalize_video_output(video_partial_path, video_final_path)
        if getattr(args, "display", False):
            cv2.destroyAllWindows()
    return 0

def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    args = resolve_settings(args, cfg)
    backend_base = args.backend_url.rsplit("/", 1)[0] if args.backend_url.endswith("/update") else args.backend_url
    fixed_geometries = load_spot_geometries(Path(args.config), backend_url=backend_base)
    if not fixed_geometries:
        fixed_geometries = normalize_spot_geometries(cfg.get("rois"))
    args.perspective_transform = load_perspective_transform(cfg)

    if args.camera is not None:
        (
            args.camera,
            args.camera_backend,
            args.fallback_camera,
            args.fallback_camera_backend,
            args.camera_source_label,
        ) = resolve_camera_runtime(args.camera)
        args.requested_camera = str(args.camera_source_label).strip().lower()
        if args.camera_backend not in (None, 0):
            print(f"Using camera index {args.camera} with AVFoundation backend.")
        else:
            print(f"Using camera index {args.camera}.")
        if args.requested_camera == DEFAULT_IPHONE_CAMERA_LABEL and args.fallback_camera is not None:
            print(
                f"Built-in camera fallback is ready on index {args.fallback_camera}."
            )
        raise SystemExit(run_camera(args, fixed_geometries))
    if args.video is not None:
        raise SystemExit(run_video(args, fixed_geometries))
    raise SystemExit(run_inference(args, fixed_geometries))


if __name__ == "__main__":
    main()
