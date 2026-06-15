#!/usr/bin/env python3
"""ROI Picker for Smart Parking System.

Click the TOP-LEFT and BOTTOM-RIGHT corners of each parking spot.
Each pair of clicks defines one ROI. Press:
  U  - undo last ROI
  S  - save to config.yaml and quit
  Q  - quit without saving
  R  - reset all ROIs

Usage:
  python3 pick_rois.py --image "samples/your_image.jpg"
  python3 pick_rois.py --video "samples/your_video.mp4"
  python3 pick_rois.py --video "samples/your_video.mp4" --frame-number 120
  python3 pick_rois.py --video "samples/your_video.mp4" --time-sec 4.0
  python3 pick_rois.py --image "samples/your_image.jpg" --config edge/config.yaml
  python3 pick_rois.py --image "samples/your_image.jpg" --scale 0.7
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

# ── colours ────────────────────────────────────────────────────────────────
COLOR_CONFIRMED = (47, 255, 173)  # green
COLOR_PENDING = (0, 200, 255)  # yellow-cyan  (first click placed)
COLOR_HOVER = (200, 200, 200)  # grey crosshair
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (20, 20, 20)
COLOR_ACCENT = (255, 80, 180)  # magenta accent


# ── state ───────────────────────────────────────────────────────────────────
class PickerState:
    def __init__(self) -> None:
        self.rois: List[Tuple[str, int, int, int, int]] = []  # (name, x1,y1,x2,y2)
        self.pending: Optional[Tuple[int, int]] = None  # first click
        self.mouse_x = 0
        self.mouse_y = 0

    def next_name(self) -> str:
        return f"spot_{len(self.rois) + 1}"

    def click(self, x: int, y: int) -> Optional[str]:
        """Returns spot name if a ROI was completed, else None."""
        if self.pending is None:
            self.pending = (x, y)
            return None
        x1, y1 = self.pending
        x2, y2 = x, y
        # normalise so top-left is always smaller
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        if x2 - x1 < 5 or y2 - y1 < 5:
            self.pending = None
            return None
        name = self.next_name()
        self.rois.append((name, x1, y1, x2, y2))
        self.pending = None
        return name

    def undo(self) -> None:
        if self.rois:
            self.rois.pop()
        self.pending = None

    def reset(self) -> None:
        self.rois.clear()
        self.pending = None

    def to_yaml_dict(self) -> dict:
        return {name: [x1, y1, x2, y2] for name, x1, y1, x2, y2 in self.rois}

    def load_yaml_dict(self, rois: dict) -> None:
        loaded: List[Tuple[str, int, int, int, int]] = []
        for name, coords in rois.items():
            if not isinstance(coords, list) or len(coords) != 4:
                continue
            x1, y1, x2, y2 = (int(v) for v in coords)
            loaded.append((str(name), x1, y1, x2, y2))
        self.rois = loaded
        self.pending = None


# ── drawing ─────────────────────────────────────────────────────────────────
def draw_crosshair(canvas: np.ndarray, x: int, y: int) -> None:
    h, w = canvas.shape[:2]
    cv2.line(canvas, (0, y), (w, y), COLOR_HOVER, 1, cv2.LINE_AA)
    cv2.line(canvas, (x, 0), (x, h), COLOR_HOVER, 1, cv2.LINE_AA)


def draw_rois(canvas: np.ndarray, state: PickerState, scale: float) -> None:
    for name, x1, y1, x2, y2 in state.rois:
        sx1, sy1, sx2, sy2 = (int(v * scale) for v in (x1, y1, x2, y2))
        cv2.rectangle(canvas, (sx1, sy1), (sx2, sy2), COLOR_CONFIRMED, 2)
        # label background
        label = name
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(
            canvas, (sx1, sy1 - th - 8), (sx1 + tw + 6, sy1), COLOR_CONFIRMED, -1
        )
        cv2.putText(
            canvas,
            label,
            (sx1 + 3, sy1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_BG,
            1,
            cv2.LINE_AA,
        )

    # pending first click
    if state.pending is not None:
        px, py = int(state.pending[0] * scale), int(state.pending[1] * scale)
        mx, my = state.mouse_x, state.mouse_y
        # draw rubber-band rectangle
        rx1, rx2 = min(px, mx), max(px, mx)
        ry1, ry2 = min(py, my), max(py, my)
        cv2.rectangle(canvas, (rx1, ry1), (rx2, ry2), COLOR_PENDING, 1)
        cv2.circle(canvas, (px, py), 5, COLOR_PENDING, -1)


def draw_hud(canvas: np.ndarray, state: PickerState, scale: float) -> None:
    h, w = canvas.shape[:2]
    lines = [
        f"ROIs defined: {len(state.rois)}",
        f"Next spot: {state.next_name()}",
        "",
        "LEFT CLICK  1st corner",
        "LEFT CLICK  2nd corner",
        "U  undo last ROI",
        "R  reset all",
        "S  save & quit",
        "Q  quit",
    ]
    if state.pending:
        lines[0] = "Click 2nd corner..."

    pad = 12
    line_h = 20
    box_h = len(lines) * line_h + pad * 2
    box_w = 210

    overlay = canvas.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
    cv2.rectangle(canvas, (pad, pad), (pad + box_w, pad + box_h), COLOR_ACCENT, 1)

    for i, line in enumerate(lines):
        color = (
            COLOR_ACCENT
            if line.startswith("ROIs") or line.startswith("Next")
            else COLOR_TEXT
        )
        cv2.putText(
            canvas,
            line,
            (pad + 8, pad + pad + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    # coords
    coord_text = f"x={int(state.mouse_x / scale)}  y={int(state.mouse_y / scale)}"
    cv2.putText(
        canvas,
        coord_text,
        (w - 160, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        COLOR_HOVER,
        1,
        cv2.LINE_AA,
    )


def render(base: np.ndarray, state: PickerState, scale: float) -> np.ndarray:
    canvas = base.copy()
    draw_crosshair(canvas, state.mouse_x, state.mouse_y)
    draw_rois(canvas, state, scale)
    draw_hud(canvas, state, scale)
    return canvas


# ── save/load ───────────────────────────────────────────────────────────────
def make_source_key(source_path: Path, source_kind: str) -> str:
    try:
        normalized = source_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        normalized = source_path.resolve().as_posix()
    return f"{source_kind}:{normalized}"


def load_source_rois(config_path: Path, source_path: Path, source_kind: str) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        existing = yaml.safe_load(f) or {}

    source_key = make_source_key(source_path, source_kind)
    source_entry = (existing.get("roi_sources") or {}).get(source_key) or {}
    rois = source_entry.get("rois")
    if isinstance(rois, dict):
        return rois

    # Backward compatibility with older single-source configs.
    legacy_rois = existing.get("rois")
    if isinstance(legacy_rois, dict):
        return legacy_rois
    return {}


def save_config(
    state: PickerState, config_path: Path, source_path: Path, source_kind: str
) -> None:
    existing: dict = {}
    if config_path.exists():
        with open(config_path, encoding="utf-8") as f:
            existing = yaml.safe_load(f) or {}

    source_key = make_source_key(source_path, source_kind)
    roi_sources = existing.get("roi_sources") or {}
    roi_sources[source_key] = {
        "kind": source_kind,
        "path": str(source_path.resolve()),
        "rois": state.to_yaml_dict(),
    }

    existing["roi_sources"] = roi_sources
    # Keep the top-level field as the active source for compatibility with the rest of the app.
    existing["rois"] = state.to_yaml_dict()

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(existing, f, default_flow_style=False, sort_keys=False)
    print(f"\n✓ Saved {len(state.rois)} ROIs to {config_path}")
    print(f"  source: {source_key}")
    for name, x1, y1, x2, y2 in state.rois:
        print(f"  {name}: [{x1}, {y1}, {x2}, {y2}]")


# ── main ────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive ROI picker for smart-parking config.yaml"
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", help="Path to parking lot image")
    source.add_argument("--video", help="Path to parking lot video")
    p.add_argument(
        "--config", default="edge/config.yaml", help="Output config.yaml path"
    )
    p.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Display scale factor (e.g. 0.7 to shrink large images)",
    )
    p.add_argument(
        "--frame-number",
        type=int,
        default=0,
        help="Video frame number to load when using --video (default: 0).",
    )
    p.add_argument(
        "--time-sec",
        type=float,
        default=None,
        help="Video timestamp in seconds to load when using --video. Overrides --frame-number.",
    )
    return p.parse_args()


def load_source_frame(args: argparse.Namespace) -> tuple[np.ndarray, Path, str]:
    if args.image:
        source_path = Path(args.image)
        if not source_path.exists():
            sys.exit(f"Image not found: {source_path}")
        frame = cv2.imread(str(source_path))
        if frame is None:
            sys.exit(f"OpenCV could not read: {source_path}")
        return frame, source_path, "image"

    video_path = Path(str(args.video))
    if not video_path.exists():
        sys.exit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"OpenCV could not open video: {video_path}")

    try:
        if args.time_sec is not None:
            if args.time_sec < 0:
                sys.exit("--time-sec must be >= 0.")
            cap.set(cv2.CAP_PROP_POS_MSEC, float(args.time_sec) * 1000.0)
        else:
            if args.frame_number < 0:
                sys.exit("--frame-number must be >= 0.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame_number))

        ok, frame = cap.read()
        if not ok or frame is None:
            selector = (
                f"time {args.time_sec:.3f}s"
                if args.time_sec is not None
                else f"frame {args.frame_number}"
            )
            sys.exit(f"Could not read {selector} from video: {video_path}")
        return frame, video_path, "video"
    finally:
        cap.release()


def main() -> None:
    args = parse_args()
    frame, source_path, source_kind = load_source_frame(args)

    scale = float(args.scale)
    if scale != 1.0:
        dw = int(frame.shape[1] * scale)
        dh = int(frame.shape[0] * scale)
        base = cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
    else:
        base = frame.copy()

    state = PickerState()
    config_path = Path(args.config)
    existing_rois = load_source_rois(config_path, source_path, source_kind)
    if existing_rois:
        state.load_yaml_dict(existing_rois)
    WIN = "ROI Picker — Smart Parking"
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        state.mouse_x = x
        state.mouse_y = y
        if event == cv2.EVENT_LBUTTONDOWN:
            # convert display coords → original image coords
            ox, oy = int(x / scale), int(y / scale)
            name = state.click(ox, oy)
            if name:
                print(
                    f"  + {name}: [{int(x/scale)}, {int(y/scale)}, ...]  (2nd click to finish)"
                )

    cv2.setMouseCallback(WIN, on_mouse)

    print(
        f"\nOpened {source_kind}: {source_path}  ({frame.shape[1]}×{frame.shape[0]}px)"
    )
    if source_kind == "video":
        if args.time_sec is not None:
            print(f"Selected video timestamp: {args.time_sec:.3f}s")
        else:
            print(f"Selected video frame: {args.frame_number}")
    if state.rois:
        print(
            f"Loaded {len(state.rois)} existing ROIs for this source from {config_path}"
        )
    print("Click TOP-LEFT then BOTTOM-RIGHT of each parking spot.")
    print("Keys: U=undo  R=reset  S=save+quit  Q=quit\n")

    while True:
        canvas = render(base, state, scale)
        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord("q"), ord("Q"), 27):
            print("Quit without saving.")
            break
        elif key in (ord("s"), ord("S")):
            if not state.rois:
                print("No ROIs defined yet — nothing to save.")
            else:
                save_config(state, config_path, source_path, source_kind)
                break
        elif key in (ord("u"), ord("U")):
            state.undo()
            print(f"  Undo. ROIs remaining: {len(state.rois)}")
        elif key in (ord("r"), ord("R")):
            state.reset()
            print("  Reset all ROIs.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
