#!/usr/bin/env python3
"""Export a trained YOLOv8 checkpoint to ONNX FP32 and ONNX INT8.

Outputs (placed in artifacts/models/):
  best.pt        — copy of the source checkpoint
  best.onnx      — FP32 ONNX export
  best_int8.onnx — INT8 quantized ONNX export

Usage:
  python ml/export.py --weights runs/parking/yolov8n_pklot/weights/best.pt
  python ml/export.py --weights runs/parking/yolov8s_pklot/weights/best.pt
"""

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

DEFAULT_OUTPUT_DIR = "artifacts/models"
DEFAULT_IMGSZ = 640


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLOv8 to ONNX FP32 and INT8.")
    p.add_argument(
        "--weights",
        required=True,
        help="Path to best.pt checkpoint from a training run.",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for exported artifacts.",
    )
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    p.add_argument(
        "--summary-json",
        default=None,
        help="Optional path to write export artifact metadata as JSON.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    weights = Path(args.weights)
    if not weights.exists():
        raise SystemExit(f"Checkpoint not found: {weights}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = weights.parent
    summary: dict[str, object] = {
        "source_weights": str(weights),
        "output_dir": str(out_dir),
        "imgsz": args.imgsz,
        "artifacts": {},
        "notes": [],
    }

    # Copy checkpoint
    dst_pt = out_dir / "best.pt"
    shutil.copy2(weights, dst_pt)
    print(f"Copied {weights} → {dst_pt}")
    summary["artifacts"] = {
        "best.pt": artifact_entry(dst_pt),
    }

    model = YOLO(str(weights))

    # FP32 ONNX
    print("Exporting FP32 ONNX ...")
    fp32_result = model.export(format="onnx", imgsz=args.imgsz, simplify=True, opset=20)
    fp32_src = Path(str(fp32_result))
    dst_onnx: Path | None = None
    if fp32_src.exists():
        dst_onnx = out_dir / "best.onnx"
        shutil.copy2(fp32_src, dst_onnx)
        size_mb = dst_onnx.stat().st_size / 1_048_576
        print(f"FP32 ONNX → {dst_onnx} ({size_mb:.1f} MB)")
        summary["artifacts"]["best.onnx"] = artifact_entry(dst_onnx)
    else:
        print(f"Warning: FP32 ONNX not found at expected path {fp32_src}")
        summary["notes"].append(f"FP32 ONNX export missing at expected path {fp32_src}")

    print("Exporting Core ML ...")
    coreml_result = model.export(format="coreml", imgsz=args.imgsz, int8=True)
    coreml_src = Path(str(coreml_result))
    if coreml_src.exists():
        dst_coreml = out_dir / "best.mlpackage"
        shutil.copytree(coreml_src, dst_coreml, dirs_exist_ok=True)
        size_mb = (
            sum(f.stat().st_size for f in dst_coreml.rglob("*") if f.is_file())
            / 1_048_576
        )
        print(f"Core ML INT8 → {dst_coreml} ({size_mb:.1f} MB)")
        summary["artifacts"]["best.mlpackage"] = artifact_entry(dst_coreml)
    else:
        summary["notes"].append("Core ML export did not produce a .mlpackage artifact.")

    # INT8 ONNX
    if dst_onnx is not None and dst_onnx.exists():
        print("Exporting INT8 ONNX ...")
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            dst_int8 = out_dir / "best_int8.onnx"
            quantize_dynamic(
                str(dst_onnx),
                str(dst_int8),
                weight_type=QuantType.QInt8,
            )
            if weights_dir != out_dir:
                shutil.copy2(dst_int8, weights_dir / "best_int8.onnx")
            size_mb = dst_int8.stat().st_size / 1_048_576
            print(f"INT8 ONNX → {dst_int8} ({size_mb:.1f} MB)")
            summary["artifacts"]["best_int8.onnx"] = artifact_entry(dst_int8)
        except ImportError:
            print(
                "Warning: onnxruntime quantization is unavailable; best_int8.onnx was not created."
            )
            summary["notes"].append(
                "onnxruntime quantization is unavailable; INT8 ONNX export was skipped."
            )
        except Exception as exc:
            print(f"Warning: INT8 quantization failed: {exc}")
            summary["notes"].append(f"INT8 quantization failed: {exc}")
    else:
        print("Warning: skipping INT8 ONNX because the FP32 export was not created.")
        summary["notes"].append(
            "INT8 ONNX export skipped because the FP32 ONNX export was missing."
        )

    print(f"\nArtifacts in {out_dir}:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}  ({artifact_size_mb(f):.1f} MB)")
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Summary JSON → {summary_path}")


def artifact_size_mb(path: Path) -> float:
    if path.is_dir():
        return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1_048_576
    return path.stat().st_size / 1_048_576


def artifact_entry(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "present": path.exists(),
        "size_mb": round(artifact_size_mb(path), 2) if path.exists() else None,
    }


if __name__ == "__main__":
    main()
