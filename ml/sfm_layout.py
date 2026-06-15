#!/usr/bin/env python3
"""Build a deterministic Week 5 BEV-style layout sample from lot photos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


class SfMError(RuntimeError):
    """Raised when SfM cannot produce a usable layout from the input photos."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Week 5 BEV-style map sample from 4-5 lot photos.")
    parser.add_argument("--images", required=True, help="Directory containing lot photos.")
    parser.add_argument("--output", required=True, help="Output directory for bev_map.png and layout.json.")
    parser.add_argument("--spots-json", default=None, help="Optional explicit spot polygons in BEV coordinates.")
    return parser.parse_args()


def image_paths(images_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(sorted(images_dir.glob(suffix)))
    if len(paths) < 1:
        raise SfMError(f"No images found in {images_dir}")
    return paths


def load_images(paths: list[Path]) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise SfMError(f"Failed to read image: {path}")
        frames.append(frame)
    return frames


def build_bev_canvas(frames: list[np.ndarray]) -> np.ndarray:
    base = frames[0]
    height, width = base.shape[:2]
    canvas = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
    offset_x = width // 2
    offset_y = height // 2
    canvas[offset_y:offset_y + height, offset_x:offset_x + width] = base

    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(base, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for frame in frames[1:]:
        kp2, des2 = orb.detectAndCompute(frame, None)
        if des1 is None or des2 is None or not kp1 or not kp2:
            continue
        matches = matcher.match(des2, des1)
        matches = sorted(matches, key=lambda m: m.distance)[:150]
        if len(matches) < 8:
            continue
        src = np.float32([kp2[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst = np.float32([kp1[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        homography, _mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
        if homography is None:
            continue
        translate = np.array([[1, 0, offset_x], [0, 1, offset_y], [0, 0, 1]], dtype=np.float64)
        warped = cv2.warpPerspective(frame, translate @ homography, (canvas.shape[1], canvas.shape[0]))
        mask = warped.any(axis=2)
        canvas[mask] = warped[mask]

    return crop_nonzero(canvas)


def crop_nonzero(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    if coords is None:
        return frame
    x, y, w, h = cv2.boundingRect(coords)
    return frame[y:y + h, x:x + w]


def build_placeholder_spots(width: int, height: int, count: int = 6) -> list[dict[str, Any]]:
    cols = 3
    rows = max(1, int(np.ceil(count / cols)))
    margin_x = max(20, width // 12)
    margin_y = max(20, height // 10)
    lane_width = max(40, (width - (2 * margin_x)) // cols)
    lane_height = max(40, (height - (2 * margin_y)) // (rows * 2))
    spots: list[dict[str, Any]] = []
    index = 1
    for row in range(rows):
        for col in range(cols):
            if index > count:
                break
            x1 = margin_x + col * lane_width
            y1 = margin_y + row * (lane_height + margin_y // 2)
            x2 = min(width - margin_x, x1 + lane_width - 12)
            y2 = min(height - margin_y, y1 + lane_height)
            corners = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            spots.append({"spot_id": f"spot_{index}", "corners": corners})
            index += 1
    return spots


def load_spots(path: str | None, width: int, height: int) -> tuple[list[dict[str, Any]], str]:
    if not path:
        return build_placeholder_spots(width, height), "placeholder_grid"
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected a JSON array in {path}")
    return data, "provided_json"


def generate_layout(
    paths: list[Path],
    output_dir: Path,
    spots_json: Path | str | None = None,
) -> dict[str, Any]:
    """Run the SfM/BEV pipeline and persist ``bev_map.png`` + ``layout.json``.

    Returns the layout dict (``canvas``, ``background_image``, ``spot_source``,
    ``source_images``, ``spots``). Raises :class:`SfMError` when the input photos
    cannot produce a usable layout, so callers (e.g. the backend) can fall back
    to manual polygon submission.
    """
    if not paths:
        raise SfMError("No images provided for SfM.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_images(paths)
    bev = build_bev_canvas(frames)
    if bev is None or bev.size == 0:
        raise SfMError("SfM produced an empty BEV canvas.")

    bev_path = output_dir / "bev_map.png"
    if not cv2.imwrite(str(bev_path), bev):
        raise SfMError(f"Failed to write {bev_path}")

    height, width = bev.shape[:2]
    spots, spot_source = load_spots(str(spots_json) if spots_json else None, width, height)
    layout = {
        "canvas": {"width": width, "height": height},
        "background_image": bev_path.name,
        "spot_source": spot_source,
        "source_images": [path.name for path in paths],
        "spots": spots,
    }
    (output_dir / "layout.json").write_text(json.dumps(layout, indent=2), encoding="utf-8")
    return layout


def main() -> None:
    args = parse_args()
    images_dir = Path(args.images)
    output_dir = Path(args.output)

    try:
        paths = image_paths(images_dir)
        layout = generate_layout(paths, output_dir, args.spots_json)
    except SfMError as exc:
        raise SystemExit(str(exc)) from exc

    print(f"[SfM] wrote {output_dir / 'bev_map.png'}")
    print(f"[SfM] wrote {output_dir / 'layout.json'} ({len(layout['spots'])} spots)")


if __name__ == "__main__":
    main()
