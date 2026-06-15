#!/usr/bin/env python3
"""Run one-off prediction for smart parking models.

Primary path:
  Stage 2 patch classification with YOLOv8*-cls.

Optional paths:
  Stage 1 spot detection or the single-model occupancy detector.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

DEFAULT_STAGE2_IMGSZ = 64
DEFAULT_DETECT_IMGSZ = 640
DEFAULT_CONF = 0.25
REFERENCE_COLORS = {
    "free": (47, 255, 173),
    "occupied": (255, 0, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict with Stage 2 classification by default, or Stage 1 / single-model detection."
    )
    parser.add_argument(
        "--weights", required=True, help="Path to model checkpoint (.pt or .onnx)."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Image path, directory, or video/camera source accepted by Ultralytics.",
    )
    parser.add_argument(
        "--device", default="mps", help="Inference device, e.g. mps or cpu."
    )
    parser.add_argument(
        "--imgsz", type=int, default=None, help="Inference image size override."
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=DEFAULT_CONF,
        help="Detection confidence threshold.",
    )
    parser.add_argument(
        "--save", action="store_true", help="Save Ultralytics prediction artifacts."
    )
    parser.add_argument(
        "--save-reference-style",
        action="store_true",
        help="Save a custom annotated image with confidence-only labels styled like the reference screenshot.",
    )

    parser.add_argument(
        "--stage1", action="store_true", help="Run Stage 1 detection prediction."
    )
    parser.add_argument(
        "--stage2", action="store_true", help="Run Stage 2 classification prediction."
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Run the ML-only single-model occupancy detector prediction.",
    )
    return parser.parse_args()


def prediction_mode(args: argparse.Namespace) -> str:
    modes = [args.stage1, args.stage2, args.single_model]
    if sum(bool(value) for value in modes) > 1:
        raise SystemExit("Choose only one of --stage1, --stage2, or --single-model.")
    if args.stage1:
        return "stage1"
    if args.single_model:
        return "single_model"
    return "stage2"


def default_imgsz(mode: str, override: int | None) -> int:
    if override is not None:
        return int(override)
    return DEFAULT_STAGE2_IMGSZ if mode == "stage2" else DEFAULT_DETECT_IMGSZ


def classification_output(
    result: Any, *, source: str, weights: str
) -> dict[str, object]:
    top1 = int(result.probs.top1)
    label = str(result.names[top1])
    confidence = float(result.probs.top1conf)
    return {
        "task": "classify",
        "source": source,
        "weights": weights,
        "label": label,
        "confidence": round(confidence, 4),
        "probabilities": {
            str(name): round(float(result.probs.data[int(index)]), 4)
            for index, name in sorted(
                result.names.items(), key=lambda item: int(item[0])
            )
        },
    }


def detection_output(
    result: Any, *, source: str, weights: str, mode: str
) -> dict[str, object]:
    boxes = []
    names = {int(index): str(name) for index, name in result.names.items()}
    xyxy = result.boxes.xyxy.cpu().tolist() if result.boxes is not None else []
    confs = result.boxes.conf.cpu().tolist() if result.boxes is not None else []
    classes = result.boxes.cls.cpu().tolist() if result.boxes is not None else []
    for box, conf, cls_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [round(float(value), 2) for value in box]
        class_id = int(cls_id)
        boxes.append(
            {
                "class_id": class_id,
                "label": names.get(class_id, str(class_id)),
                "confidence": round(float(conf), 4),
                "xyxy": [x1, y1, x2, y2],
            }
        )
    return {
        "task": "detect",
        "track": mode,
        "source": source,
        "weights": weights,
        "count": len(boxes),
        "boxes": boxes,
    }


def _reference_color(label: str) -> tuple[int, int, int]:
    return REFERENCE_COLORS.get(label, (255, 255, 0))


def save_reference_style(result: Any, output_path: Path) -> None:
    annotated = result.orig_img.copy()
    boxes = result.boxes
    names = {int(index): str(name) for index, name in result.names.items()}
    xyxy = boxes.xyxy.cpu().tolist() if boxes is not None else []
    confs = boxes.conf.cpu().tolist() if boxes is not None else []
    classes = boxes.cls.cpu().tolist() if boxes is not None else []

    for box, conf, cls_id in zip(xyxy, confs, classes):
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        label = names.get(int(cls_id), str(int(cls_id)))
        color = _reference_color(label)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)

        text = f"{int(round(float(conf) * 100.0))}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.75
        thickness = 2
        (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
        ty1 = max(0, y1 - th - baseline - 4)
        ty2 = max(th + baseline + 4, y1)
        tx2 = min(annotated.shape[1], x1 + tw + 8)
        cv2.rectangle(annotated, (x1, ty1), (tx2, ty2), color, -1)
        cv2.putText(
            annotated,
            text,
            (x1 + 4, ty2 - baseline - 2),
            font,
            scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    count_text = f"{len(xyxy)} objects detected"
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.9
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(count_text, font, scale, thickness)
    pad = 14
    x2 = annotated.shape[1] - pad
    y2 = annotated.shape[0] - pad
    x1 = x2 - tw - 22
    y1 = y2 - th - baseline - 16
    cv2.rectangle(annotated, (x1, y1), (x2, y2), (85, 85, 85), -1)
    cv2.putText(
        annotated,
        count_text,
        (x1 + 10, y2 - baseline - 8),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)


def main() -> None:
    args = parse_args()
    mode = prediction_mode(args)
    imgsz = default_imgsz(mode, args.imgsz)

    source = args.source
    if not source.isdigit():
        source_path = Path(source)
        if not source_path.exists():
            raise SystemExit(f"Prediction source not found: {source_path}")

    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Weights not found: {weights}")

    model = YOLO(str(weights))
    kwargs = {
        "source": source,
        "device": args.device,
        "imgsz": imgsz,
        "verbose": False,
        "save": args.save and not args.save_reference_style,
    }
    if mode != "stage2":
        kwargs["conf"] = args.conf

    results = model.predict(**kwargs)
    normalized = []
    for result in results:
        result_source = getattr(result, "path", source)
        if mode == "stage2":
            normalized.append(
                classification_output(
                    result, source=result_source, weights=str(weights)
                )
            )
        else:
            normalized.append(
                detection_output(
                    result, source=result_source, weights=str(weights), mode=mode
                )
            )
            if args.save_reference_style:
                output_dir = Path("runs/detect/reference_style")
                output_name = Path(str(result_source)).name
                save_reference_style(result, output_dir / output_name)

    print(json.dumps({"mode": mode, "predictions": normalized}, indent=2))


if __name__ == "__main__":
    main()
