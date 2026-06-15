#!/usr/bin/env python3
"""Bucket ACPDS test patches into luminance-based sunny/overcast/low-light splits."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create luminance-based ACPDS weather splits."
    )
    parser.add_argument(
        "--patch-index", default="datasets/acpds_stage2/patch_index.json"
    )
    parser.add_argument("--output", default="datasets/acpds_stage2_weather")
    parser.add_argument("--split", default="test")
    parser.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument(
        "--summary-json", default="logs/week7/acpds_weather_buckets.json"
    )
    return parser.parse_args()


def load_patch_index(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Patch index must be a list: {path}")
    return data


def image_luminance(path: Path) -> float:
    frame = cv2.imread(str(path))
    if frame is None:
        raise SystemExit(f"Could not read source image: {path}")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def bucket_name(value: float, low_cut: float, high_cut: float) -> str:
    if value <= low_cut:
        return "low_light"
    if value >= high_cut:
        return "sunny"
    return "overcast"


def materialize_patch(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
        return
    os.symlink(source.resolve(), target)


def main() -> None:
    args = parse_args()
    patch_index = load_patch_index(Path(args.patch_index))
    entries = [item for item in patch_index if item.get("split") == args.split]
    if not entries:
        raise SystemExit(f"No entries found for split={args.split}")

    by_image: dict[str, dict[str, Any]] = {}
    for item in entries:
        image_key = str(item["image"])
        bucket = by_image.setdefault(
            image_key,
            {"image_path": str(item["image_path"]), "entries": []},
        )
        bucket["entries"].append(item)

    luminance_rows = []
    for image_key, item in sorted(by_image.items()):
        luminance_rows.append(
            {
                "image": image_key,
                "image_path": item["image_path"],
                "luminance": image_luminance(Path(item["image_path"])),
                "entry_count": len(item["entries"]),
            }
        )

    luminance_values = np.array(
        [row["luminance"] for row in luminance_rows], dtype=np.float32
    )
    low_cut = float(np.quantile(luminance_values, 1.0 / 3.0))
    high_cut = float(np.quantile(luminance_values, 2.0 / 3.0))

    output_root = Path(args.output)
    counts = {
        "sunny": {"free": 0, "occupied": 0},
        "overcast": {"free": 0, "occupied": 0},
        "low_light": {"free": 0, "occupied": 0},
    }
    for row in luminance_rows:
        row["bucket"] = bucket_name(float(row["luminance"]), low_cut, high_cut)
        for entry in by_image[row["image"]]["entries"]:
            label = str(entry["label"])
            source = Path(str(entry["patch_path"]))
            target = output_root / row["bucket"] / label / source.name
            materialize_patch(source, target, args.mode)
            counts[row["bucket"]][label] += 1

    summary = {
        "mode": "acpds_weather_buckets",
        "split": args.split,
        "materialization": args.mode,
        "patch_index": str(Path(args.patch_index).resolve()),
        "output_root": str(output_root.resolve()),
        "thresholds": {
            "low_light_max_luminance": round(low_cut, 4),
            "overcast_min_luminance": round(low_cut, 4),
            "sunny_min_luminance": round(high_cut, 4),
        },
        "counts": counts,
        "source_images": luminance_rows,
        "methodology": "Image-level ACPDS weather proxy from HSV V-channel tertiles on the selected split: bottom third=low_light, middle third=overcast, top third=sunny.",
    }

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
