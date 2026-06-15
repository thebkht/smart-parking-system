#!/usr/bin/env python3
"""Convert the Parking Space Detection Dataset into YOLO detection format."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import yaml

VALID_LABELS = {
    "free_space": 0,
    "not_free_space": 1,
    "partially_free_space": 2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export datasets/parking-space-detection as a YOLO detection dataset."
    )
    parser.add_argument(
        "--source",
        default="datasets/parking-space-detection",
        help="Dataset root containing annotations.xml and images/.",
    )
    parser.add_argument(
        "--output",
        default="datasets/parking-space-detection/yolo",
        help="Target YOLO dataset directory.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Shuffle seed for deterministic splits.",
    )
    parser.add_argument(
        "--collapse-classes",
        action="store_true",
        help="Map all labels to one 'space' class.",
    )
    return parser.parse_args()


def clip_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def xyxy_to_cxcywh(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float] | None:
    x1 = clip_unit(x1)
    y1 = clip_unit(y1)
    x2 = clip_unit(x2)
    y2 = clip_unit(y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def polygon_to_box(
    points: str, *, width: int, height: int
) -> tuple[float, float, float, float] | None:
    xs: list[float] = []
    ys: list[float] = []
    for pair in points.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        try:
            x_raw, y_raw = pair.split(",", 1)
        except ValueError:
            return None
        xs.append(float(x_raw) / width)
        ys.append(float(y_raw) / height)
    if not xs or not ys:
        return None
    return xyxy_to_cxcywh(min(xs), min(ys), max(xs), max(ys))


def split_items(
    items: list[dict[str, object]], *, val_ratio: float, test_ratio: float, seed: int
) -> dict[str, list[dict[str, object]]]:
    if val_ratio <= 0 or test_ratio <= 0 or (val_ratio + test_ratio) >= 1:
        raise SystemExit(
            "val_ratio and test_ratio must be positive and sum to less than 1."
        )
    if len(items) < 3:
        raise SystemExit(
            "Need at least 3 labeled images to create train/val/test splits."
        )

    shuffled = items[:]
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    test_count = max(1, round(total * test_ratio))
    val_count = max(1, round(total * val_ratio))
    train_count = total - val_count - test_count

    if train_count < 1:
        raise SystemExit("Split ratios leave no training images.")

    train_end = train_count
    val_end = train_end + val_count
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    annotations_path = source / "annotations.xml"
    images_dir = source / "images"

    if not annotations_path.exists():
        raise SystemExit(f"Missing annotations file: {annotations_path}")
    if not images_dir.exists():
        raise SystemExit(f"Missing images directory: {images_dir}")

    tree = ET.parse(annotations_path)
    records: list[dict[str, object]] = []
    missing_images: list[str] = []
    raw_label_counts: Counter[str] = Counter()
    kept_label_counts: Counter[str] = Counter()
    skipped_invalid = 0
    skipped_empty = 0

    for image_node in tree.getroot().findall("image"):
        image_rel = image_node.attrib.get("name", "")
        image_path = source / image_rel
        if not image_path.exists():
            missing_images.append(image_rel)
            continue

        width = int(image_node.attrib["width"])
        height = int(image_node.attrib["height"])
        lines: list[str] = []
        seen: set[tuple[int, float, float, float, float]] = set()

        for polygon in image_node.findall("polygon"):
            label = polygon.attrib.get("label", "").strip()
            if label not in VALID_LABELS:
                continue
            raw_label_counts[label] += 1
            box = polygon_to_box(
                polygon.attrib.get("points", ""),
                width=width,
                height=height,
            )
            if box is None:
                skipped_invalid += 1
                continue

            class_id = 0 if args.collapse_classes else VALID_LABELS[label]
            rounded = (class_id, *(round(value, 6) for value in box))
            if rounded in seen:
                continue
            seen.add(rounded)

            kept_label_counts[label] += 1
            lines.append(
                f"{class_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}"
            )

        if not lines:
            skipped_empty += 1
            continue

        records.append(
            {
                "stem": image_path.stem,
                "image_path": image_path,
                "suffix": image_path.suffix.lower(),
                "label_lines": lines,
            }
        )

    splits = split_items(
        records,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    if output.exists():
        shutil.rmtree(output)

    for split, items in splits.items():
        image_out = output / split / "images"
        label_out = output / split / "labels"
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)
        for item in items:
            image_path = Path(str(item["image_path"]))
            target_name = f"{item['stem']}{item['suffix']}"
            shutil.copy2(image_path, image_out / target_name)
            (label_out / f"{item['stem']}.txt").write_text(
                "\n".join(item["label_lines"]) + "\n",
                encoding="utf-8",
            )

    names = ["space"] if args.collapse_classes else list(VALID_LABELS.keys())
    dataset_yaml = {
        "path": str(output.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(names),
        "names": names,
    }
    with open(output / "dataset.yaml", "w", encoding="utf-8") as handle:
        yaml.dump(dataset_yaml, handle, default_flow_style=False, sort_keys=False)

    report = {
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "images_kept": len(records),
        "split_counts": {split: len(items) for split, items in splits.items()},
        "boxes_kept_total": sum(len(item["label_lines"]) for item in records),
        "raw_label_counts": dict(sorted(raw_label_counts.items())),
        "kept_label_counts": dict(sorted(kept_label_counts.items())),
        "missing_images": missing_images,
        "invalid_polygons_skipped": skipped_invalid,
        "empty_images_skipped": skipped_empty,
        "class_mode": "collapsed" if args.collapse_classes else "preserved",
        "names": names,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
    }
    with open(output / "conversion_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"YOLO dataset written to {output}")
    print(
        f"Images kept: {report['images_kept']} | Boxes kept: {report['boxes_kept_total']}"
    )
    print(f"Splits: {report['split_counts']}")
    print(f"Classes: {names}")


if __name__ == "__main__":
    main()
