#!/usr/bin/env python3
"""Timed stability test for the integrated smart parking pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from edge.detect import (
    DEFAULT_BACKEND_URL,
    SmoothingBuffer,
    create_models,
    load_config,
    load_rois,
    normalize_rois,
    post_payload,
    resolve_settings,
    run_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a timed stability test for the edge pipeline."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image", help="Static image source. Reuses the same frame for the duration."
    )
    group.add_argument("--camera", help="Camera index to use for the duration.")

    parser.add_argument(
        "--stage1-detector",
        action="store_true",
        help="Use Stage 1 detector instead of fixed ROIs.",
    )
    parser.add_argument("--stage1-model", default=None)
    parser.add_argument("--stage2-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default="edge/config.yaml")
    parser.add_argument("--smooth-n", type=int, default=None)
    parser.add_argument("--stage2-threshold", type=float, default=None)
    parser.add_argument("--backend-url", default=DEFAULT_BACKEND_URL)
    parser.add_argument("--backend-timeout", type=float, default=0.75)
    parser.add_argument("--backend-retry-delay", type=float, default=10.0)
    parser.add_argument(
        "--post", action="store_true", help="POST payloads during the test."
    )
    parser.add_argument("--post-interval", type=float, default=None, metavar="SEC")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--log-format", choices=["csv", "json"], default=None)
    parser.add_argument("--latest-frame-path", default="logs/images/latest_frame.jpg")
    parser.add_argument("--stream-jpeg-quality", type=int, default=55)
    parser.add_argument("--stream-max-width", type=int, default=640)
    parser.add_argument("--stage1-imgsz", type=int, default=None)
    parser.add_argument("--stage1-sahi", action="store_true", default=True)
    parser.add_argument("--no-stage1-sahi", action="store_false", dest="stage1_sahi")
    parser.add_argument("--stage1-slice-size", type=int, default=None)
    parser.add_argument("--stage1-overlap", type=float, default=None)
    parser.add_argument("--save-annotated", default=None)
    parser.add_argument("--status-url", default="http://127.0.0.1:8000/status")
    parser.add_argument(
        "--duration", type=int, default=1800, help="Duration in seconds."
    )
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=250,
        help="Delay between iterations in milliseconds.",
    )
    parser.add_argument("--output", default="logs/stability_summary.json")
    return parser.parse_args()


def load_frame(image_path: str) -> Any:
    frame = cv2.imread(image_path)
    if frame is None:
        raise SystemExit(f"Could not read image: {image_path}")
    return frame


def fetch_backend_status(status_url: str) -> dict[str, Any]:
    response = requests.get(status_url, timeout=5)
    response.raise_for_status()
    body = response.json()
    return body if isinstance(body, dict) else {}


def save_summary(summary: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(Path(args.config))
    args = resolve_settings(args, cfg)

    # Load ROIs from backend GET /map first; fall back to config.yaml ROIs
    backend_base = args.backend_url.rsplit("/update", 1)[0]
    fixed_rois = load_rois(Path(args.config), backend_url=backend_base)
    if not fixed_rois:
        fixed_rois = normalize_rois(cfg.get("rois"))

    stage1_model, stage2_model = create_models(args)
    smoothing = SmoothingBuffer(window=args.smooth_n)

    capture = None
    static_frame = None
    if args.image:
        static_frame = load_frame(args.image)
    else:
        capture = cv2.VideoCapture(int(args.camera))
        if not capture.isOpened():
            raise SystemExit(f"Could not open camera {args.camera}")

    started_at = time.perf_counter()
    summary: dict[str, Any] = {
        "started_at_epoch_s": round(started_at, 3),
        "duration_s": args.duration,
        "iterations": 0,
        "successful_iterations": 0,
        "post_successes": 0,
        "post_failures": 0,
        "backend_status_checks": 0,
        "backend_status_matches": 0,
        "backend_status_failures": 0,
        "read_failures": 0,
        "errors": [],
        "stage1_detector": bool(args.stage1_detector),
        "stage1_model": args.stage1_model,
        "stage2_model": args.stage2_model,
        "device": args.device,
    }
    last_payload: dict[str, Any] | None = None

    try:
        while time.perf_counter() - started_at < args.duration:
            summary["iterations"] += 1
            if static_frame is not None:
                frame = static_frame.copy()
            else:
                ok, frame = capture.read()
                if not ok:
                    summary["read_failures"] += 1
                    time.sleep(args.frame_interval / 1000.0)
                    continue

            try:
                payload, _ = run_pipeline(
                    frame, fixed_rois, stage1_model, stage2_model, smoothing, args
                )
                last_payload = payload
                summary["successful_iterations"] += 1
                if args.post:
                    try:
                        post_payload(payload, args.backend_url)
                        summary["post_successes"] += 1
                    except requests.RequestException as exc:
                        summary["post_failures"] += 1
                        summary["errors"].append(f"POST failed: {exc}")

                    try:
                        summary["backend_status_checks"] += 1
                        status = fetch_backend_status(args.status_url)
                        if status == payload:
                            summary["backend_status_matches"] += 1
                        else:
                            summary["backend_status_failures"] += 1
                    except requests.RequestException as exc:
                        summary["backend_status_failures"] += 1
                        summary["errors"].append(f"GET /status failed: {exc}")
            except (
                Exception
            ) as exc:  # pragma: no cover - protection for long-running test
                summary["errors"].append(str(exc))

            time.sleep(args.frame_interval / 1000.0)
    finally:
        if capture is not None:
            capture.release()

    ended_at = time.perf_counter()
    elapsed = ended_at - started_at
    summary["elapsed_s"] = round(elapsed, 2)
    summary["avg_iterations_per_sec"] = (
        round(summary["iterations"] / elapsed, 3) if elapsed else 0.0
    )
    summary["last_payload"] = last_payload
    summary["passed"] = (
        summary["successful_iterations"] > 0
        and summary["read_failures"] == 0
        and summary["post_failures"] == 0
        and summary["backend_status_failures"] == 0
    )

    save_summary(summary, Path(args.output))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
