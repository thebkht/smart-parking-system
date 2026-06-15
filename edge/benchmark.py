#!/usr/bin/env python3
"""FPS benchmark: compare MPS, CPU, ONNX FP32, ONNX INT8, and Core ML.

Usage:
    python edge/benchmark.py --task classify --image path/to/parking.jpg --model artifacts/models/best.pt
    python edge/benchmark.py --task detect --image path/to/parking.jpg --model runs/stage1_det/.../best.pt

ONNX paths are inferred from the .pt path:
    best.pt       → best.onnx        (FP32)
    best.pt       → best_int8.onnx   (INT8)
    best.pt       → best.mlpackage   (Core ML, Apple Silicon only)

Skips any backend whose model file is not found.
"""
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge inference FPS benchmark.")
    parser.add_argument(
        "--task",
        choices=["classify", "detect"],
        default="classify",
        help="Benchmark task. Use classify for Stage 2 and detect for Stage 1.",
    )
    parser.add_argument("--image", required=True, help="Path to test image.")
    parser.add_argument("--model", required=True, help="Path to .pt model file.")
    parser.add_argument(
        "--imgsz",
        type=int,
        default=64,
        help="Input resolution. Use 64 for Stage 2 classifier and 768/640 for detectors.",
    )
    parser.add_argument(
        "--roi",
        nargs=4,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Optional ROI crop before benchmarking. Useful for Stage 2 patch inference.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Number of measured iterations.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (not measured).",
    )
    parser.add_argument(
        "--output-json",
        default="logs/benchmark_results.json",
        help="Path to save benchmark results as JSON.",
    )
    parser.add_argument(
        "--output-csv",
        default="logs/benchmark_results.csv",
        help="Path to save benchmark results as CSV.",
    )
    return parser.parse_args()


def clip_roi(frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    if roi is None:
        return frame
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = roi
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        raise SystemExit(f"Invalid ROI after clipping: {roi}")
    return frame[y1:y2, x1:x2]


def prepare_runtime_frame(
    frame: np.ndarray,
    *,
    task: str,
    imgsz: int,
    roi: Optional[Tuple[int, int, int, int]],
) -> np.ndarray:
    cropped = clip_roi(frame, roi)
    interpolation = (
        cv2.INTER_AREA
        if cropped.shape[0] >= imgsz and cropped.shape[1] >= imgsz
        else cv2.INTER_LINEAR
    )
    if task == "classify":
        return cv2.resize(cropped, (imgsz, imgsz), interpolation=interpolation)
    return cropped


def benchmark_yolo(
    model_path: str,
    frame: np.ndarray,
    device: str,
    iterations: int,
    warmup: int,
) -> Dict[str, Any]:
    """Benchmark ultralytics YOLO on a given device."""
    from ultralytics import YOLO

    model = YOLO(model_path)

    for _ in range(warmup):
        model(frame, device=device, verbose=False)

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model(frame, device=device, verbose=False)
        times.append(time.perf_counter() - t0)

    avg_latency = sum(times) / len(times)
    return {
        "backend": f"YOLO-{device}",
        "fps": round(1 / avg_latency, 1),
        "latency_ms": round(avg_latency * 1000, 1),
        "model_size_mb": round(Path(model_path).stat().st_size / 1e6, 1),
    }


def benchmark_onnx(
    onnx_path: str,
    frame: np.ndarray,
    *,
    task: str,
    imgsz: int,
    iterations: int,
    warmup: int,
) -> Optional[Dict[str, Any]]:
    """Benchmark ONNX Runtime on CPU. Returns None if the file does not exist."""
    if not Path(onnx_path).exists():
        return None

    try:
        import onnxruntime as ort
    except ImportError:
        print(f"  onnxruntime not installed — skipping {onnx_path}")
        return None

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    resized = cv2.resize(frame, (imgsz, imgsz))
    tensor = resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR→RGB, /255
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis]  # HWC → 1CHW

    for _ in range(warmup):
        session.run(None, {input_name: tensor})

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        session.run(None, {input_name: tensor})
        times.append(time.perf_counter() - t0)

    avg_latency = sum(times) / len(times)
    return {
        "backend": None,  # caller sets label
        "fps": round(1 / avg_latency, 1),
        "latency_ms": round(avg_latency * 1000, 1),
        "model_size_mb": round(Path(onnx_path).stat().st_size / 1e6, 1),
    }


def benchmark_coreml(
    mlpackage_path: Path,
    frame: np.ndarray,
    *,
    imgsz: int,
    iterations: int,
    warmup: int,
) -> Optional[Dict[str, Any]]:
    """Benchmark Core ML on Apple Silicon (ANE/GPU). Returns None if unavailable."""
    if not mlpackage_path.exists():
        return None

    try:
        import coremltools as ct
        import PIL.Image
    except ImportError:
        print("  coremltools or Pillow not installed — skipping Core ML")
        return None

    model = ct.models.MLModel(str(mlpackage_path))

    # Prepare input as PIL RGB image resized to imgsz
    resized = cv2.resize(frame, (imgsz, imgsz))
    pil_img = PIL.Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

    # Discover the image input name from the model spec
    spec = model.get_spec()
    input_name = spec.description.input[0].name

    for _ in range(warmup):
        model.predict({input_name: pil_img})

    times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        model.predict({input_name: pil_img})
        times.append(time.perf_counter() - t0)

    avg_latency = sum(times) / len(times)
    size_mb = (
        sum(f.stat().st_size for f in mlpackage_path.rglob("*") if f.is_file()) / 1e6
    )

    return {
        "backend": "CoreML-int8",
        "fps": round(1 / avg_latency, 1),
        "latency_ms": round(avg_latency * 1000, 1),
        "model_size_mb": round(size_mb, 1),
    }


def print_results_table(results: List[Dict[str, Any]]) -> None:
    if not results:
        print("No benchmark results.")
        return

    header = f"{'Backend':<20} {'FPS':>8} {'Latency (ms)':>14} {'Model Size (MB)':>16}"
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)
    for r in results:
        print(
            f"{r['backend']:<20} {r['fps']:>8.1f} {r['latency_ms']:>14.1f}"
            f" {r['model_size_mb']:>16.1f}"
        )
    print()


def save_results(
    results: List[Dict[str, Any]], json_path: Path, csv_path: Path
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)


def main() -> None:
    args = parse_args()

    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit(f"Cannot read image: {args.image}")

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(f"Model file not found: {args.model}")

    runtime_frame = prepare_runtime_frame(
        frame,
        task=args.task,
        imgsz=args.imgsz,
        roi=tuple(args.roi) if args.roi else None,
    )

    onnx_fp32_path = model_path.with_suffix(".onnx")
    onnx_int8_path = model_path.with_name(model_path.stem + "_int8.onnx")
    coreml_path = model_path.with_suffix(".mlpackage")

    results: List[Dict[str, Any]] = []

    # --- PyTorch backends (MPS + CPU) ---
    available_devices = ["cpu"]
    try:
        import torch

        if torch.backends.mps.is_available():
            available_devices.insert(0, "mps")
    except Exception:
        pass

    for device in available_devices:
        print(
            f"Benchmarking YOLO-{device} ({args.iterations} iters, {args.warmup} warmup)..."
        )
        try:
            r = benchmark_yolo(
                str(model_path), runtime_frame, device, args.iterations, args.warmup
            )
            results.append(r)
        except Exception as exc:
            print(f"  Skipped YOLO-{device}: {exc}")

    # --- ONNX backends ---
    for label, path in [("ONNX-fp32", onnx_fp32_path), ("ONNX-int8", onnx_int8_path)]:
        print(
            f"Benchmarking {label} ({args.iterations} iters, {args.warmup} warmup)..."
        )
        try:
            r = benchmark_onnx(
                str(path),
                runtime_frame,
                task=args.task,
                imgsz=args.imgsz,
                iterations=args.iterations,
                warmup=args.warmup,
            )
            if r is not None:
                r["backend"] = label
                results.append(r)
            else:
                print(f"  Skipped {label}: {path} not found")
        except Exception as exc:
            print(f"  Skipped {label}: {exc}")

    # --- Core ML backend (Apple Silicon only) ---
    print(
        f"Benchmarking CoreML-int8 ({args.iterations} iters, {args.warmup} warmup)..."
    )
    try:
        r = benchmark_coreml(
            coreml_path,
            runtime_frame,
            imgsz=args.imgsz,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        if r is not None:
            results.append(r)
        else:
            print(f"  Skipped CoreML-int8: {coreml_path} not found")
    except Exception as exc:
        print(f"  Skipped CoreML-int8: {exc}")

    print_results_table(results)
    save_results(results, Path(args.output_json), Path(args.output_csv))
    print(f"Saved JSON report to {args.output_json}")
    print(f"Saved CSV report to {args.output_csv}")


if __name__ == "__main__":
    main()
