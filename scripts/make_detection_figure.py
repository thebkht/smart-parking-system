#!/usr/bin/env python3
"""Regenerate the annotated edge-inference figure with the v6 quad pipeline.

The old figure showed axis-aligned rectangles (the midterm fixed-ROI / square
crop style). This regenerates it from a real ACPDS image using the current
pipeline: each annotated parking-space *quadrilateral* is perspective-warped to
a 128x128 patch and classified by the promoted YOLOv8n-cls checkpoint, then the
quad polygons are drawn colored by prediction with per-spot confidence.

Outputs full-res to runs/output/detection.jpg and a report-sized copy to
outputs/figures/detection.jpg.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "edge"))
from detect import classify_patch, order_corners  # noqa: E402

IMAGE_NAME = "GOPR6751.JPG"
IMAGE_PATH = REPO / "datasets/acpds/images" / IMAGE_NAME
ANNOTATIONS = REPO / "datasets/acpds/annotations.json"
WEIGHTS = str(REPO / "acpds_cls/weights/best.pt")

# Match the report legend: free = green (#1a7a4a), occupied = red (#c0392b).
FREE_BGR = (74, 122, 26)
OCC_BGR = (43, 57, 192)


def load_quads(image_name: str) -> list[np.ndarray]:
    data = json.loads(ANNOTATIONS.read_text())
    for split in data.values():
        names = split["file_names"]
        if image_name in names:
            return split["rois_list"][names.index(image_name)]
    raise SystemExit(f"{image_name} not found in annotations")


def main() -> None:
    frame = cv2.imread(str(IMAGE_PATH))
    if frame is None:
        raise SystemExit(f"Could not read {IMAGE_PATH}")
    h, w = frame.shape[:2]

    quads = load_quads(IMAGE_NAME)
    model = YOLO(WEIGHTS, task="classify")

    # Scale stroke/text to image resolution so polygons read after downscaling.
    thickness = max(2, w // 450)
    font_scale = w / 2600.0
    font = cv2.FONT_HERSHEY_SIMPLEX

    annotated = frame.copy()
    free = occ = 0
    for raw in quads:
        corners = order_corners(np.array([[x * w, y * h] for x, y in raw], dtype=np.float32))
        status, conf = classify_patch(frame, corners, model, device="cpu", threshold=0.5)
        color = OCC_BGR if status == "occupied" else FREE_BGR
        if status == "occupied":
            occ += 1
        else:
            free += 1
        poly = np.round(corners).astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(annotated, [poly], isClosed=True, color=color, thickness=thickness)
        label = f"{int(round(conf * 100))}%"
        tx, ty = int(corners[:, 0].min()), int(corners[:, 1].min()) - 6
        (tw, th), base = cv2.getTextSize(label, font, font_scale, max(1, thickness // 2))
        cv2.rectangle(annotated, (tx, ty - th - base), (tx + tw, ty + base), color, -1)
        cv2.putText(annotated, label, (tx, ty), font, font_scale, (255, 255, 255),
                    max(1, thickness // 2), cv2.LINE_AA)

    full_out = REPO / "runs/output/detection.jpg"
    full_out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(full_out), annotated)

    # Report-sized copy (cap width at 1920) for a lean PDF.
    report_out = REPO / "outputs/figures/detection.jpg"
    scale = min(1.0, 1920.0 / w)
    resized = cv2.resize(annotated, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(report_out), resized, [cv2.IMWRITE_JPEG_QUALITY, 90])

    print(f"{IMAGE_NAME}: {len(quads)} spots -> {free} free / {occ} occupied")
    print(f"Saved {full_out}")
    print(f"Saved {report_out}")


if __name__ == "__main__":
    main()
