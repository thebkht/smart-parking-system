#!/usr/bin/env python3
"""Prepare datasets for smart parking training.

Primary path:
  Stage 1 full-frame parking-slot detection with scene-aware holdout splits.
  Stage 2 occupancy classification on crops derived from those accepted slot labels.

Baseline path:
  Single-model full-frame occupancy detection uses the same deduped, scene-held-out
  source frames so comparison against the two-stage pipeline is fair.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Iterable

from PIL import Image
from sklearn.model_selection import train_test_split
import yaml

_FREE_DIRS = {"empty", "free", "0"}
_OCC_DIRS = {"occupied", "not_empty", "1"}
_ROBOFLOW_FREE_IDS = {0}
_ROBOFLOW_OCC_IDS = {1}
_SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}_jpg$")
_PARKING_LOT_RE = re.compile(r"^(parking_lot_\d+)_mp4-(\d+)_jpg$")
_TINY_BOX_AREA = 0.0025
_ROI_SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}

STAGE1_DATA_DIR = "datasets/stage1_data"
STAGE1_YAML = "ml/stage1.yaml"
STAGE2_DATA_DIR = "datasets/stage2_data"
STAGE2_WEATHER_DIR = "datasets/stage2_weather"
SINGLE_MODEL_DATA_DIR = "datasets/single_model_data"
SINGLE_MODEL_YAML = "ml/single_model.yaml"
PKLOT_TEST_DIR = "pklot_test"
CNRPARK_TEST_DIR = "cnrpark_test"
SANITY_REPORT = "dataset_report.json"
DETECTION_REPORT = "detection_dataset_report.json"
WEATHER_CONVENTION = (
    "Expected weather layout under the dataset root: "
    "<root>/<weather>/<class>/*.jpg where weather is one of sunny, cloudy, rainy."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare smart parking datasets. The recommended workflow is Stage 1 "
            "full-frame slot detection plus Stage 2 occupancy classification."
        )
    )
    parser.add_argument("--pklot-dir", required=True)
    parser.add_argument("--cnrpark-dir", default=None)
    parser.add_argument(
        "--parking-space-dir",
        default=None,
        help="Optional Parking Space Detection Dataset root with annotations.xml and images/ for Stage 1 augmentation.",
    )
    parser.add_argument(
        "--stage1", action="store_true", help="Prepare Stage 1 detection data."
    )
    parser.add_argument(
        "--stage2", action="store_true", help="Prepare Stage 2 classification data."
    )
    parser.add_argument(
        "--single-model",
        action="store_true",
        help="Prepare ML-only single-model occupancy detection data.",
    )
    parser.add_argument("--stage1-output", default=STAGE1_DATA_DIR)
    parser.add_argument("--stage1-yaml", default=STAGE1_YAML)
    parser.add_argument("--stage2-output", default=STAGE2_DATA_DIR)
    parser.add_argument("--weather-output", default=STAGE2_WEATHER_DIR)
    parser.add_argument("--single-model-output", default=SINGLE_MODEL_DATA_DIR)
    parser.add_argument("--single-model-yaml", default=SINGLE_MODEL_YAML)
    parser.add_argument("--patch-cache", default=None)
    parser.add_argument("--pklot-test-output", default=PKLOT_TEST_DIR)
    parser.add_argument("--cnrpark-test-output", default=CNRPARK_TEST_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _roboflow_splits(root: Path) -> list[tuple[str, Path, Path]]:
    result = []
    for split in ("train", "valid", "test"):
        image_dir = root / split / "images"
        label_dir = root / split / "labels"
        if not image_dir.exists() and split == "valid":
            image_dir = root / "val" / "images"
            label_dir = root / "val" / "labels"
            split = "val"
        if image_dir.exists() and label_dir.exists():
            result.append((split, image_dir, label_dir))
    return result


def _safe_name(src: Path, source_root: Path | None) -> str:
    if source_root is not None:
        try:
            rel = src.relative_to(source_root)
            prefix = "_".join(rel.parts[:-2])
            if prefix:
                return f"{prefix}__{src.name}"
        except ValueError:
            pass
    return src.name


def _image_files(path: Path) -> list[Path]:
    files: list[Path] = []
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(sorted(path.glob(suffix)))
    return files


def _clip_unit(value: float) -> float:
    return min(1.0, max(0.0, value))


def _xyxy_to_cxcywh(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float, float, float] | None:
    x1 = _clip_unit(x1)
    y1 = _clip_unit(y1)
    x2 = _clip_unit(x2)
    y2 = _clip_unit(y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, x2 - x1, y2 - y1)


def _label_geometry_to_box(
    values: list[float],
) -> tuple[float, float, float, float] | None:
    if not values or any((not math.isfinite(value)) for value in values):
        return None
    if len(values) == 4:
        cx, cy, bw, bh = values
        return _xyxy_to_cxcywh(
            cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0
        )
    if len(values) >= 6 and len(values) % 2 == 0:
        xs = values[0::2]
        ys = values[1::2]
        return _xyxy_to_cxcywh(min(xs), min(ys), max(xs), max(ys))
    return None


def normalize_source_stem(name: str) -> str:
    stem = Path(name).stem
    return re.sub(r"\.rf\.[A-Za-z0-9]+$", "", stem)


def scene_id_from_frame(
    normalized_stem: str, *, box_count: int, image_path: Path | None = None
) -> tuple[str, str]:
    if image_path is not None:
        parts = list(image_path.parts)
        if len(parts) > 3:
            metadata_parts = parts[:-3]
            if metadata_parts:
                return "/".join(metadata_parts), "path_metadata"

    parking_lot_match = _PARKING_LOT_RE.match(normalized_stem)
    if parking_lot_match:
        camera_id = parking_lot_match.group(1)
        frame_index = int(parking_lot_match.group(2))
        frame_block = frame_index // 25
        return f"{camera_id}_block_{frame_block}", "filename_camera_frame_block"

    if _TIMESTAMP_RE.match(normalized_stem):
        day = normalized_stem.split("_", 1)[0]
        return f"layout_{box_count}_date_{day}", "filename_date_plus_box_count"

    prefix = (
        normalized_stem.rsplit("_", 1)[0] if "_" in normalized_stem else normalized_stem
    )
    if prefix and prefix != normalized_stem:
        return prefix, "filename_prefix"
    return normalized_stem, "normalized_filename"


def iter_detection_boxes(
    label_path: Path,
) -> Iterable[tuple[int, tuple[float, float, float, float], str]]:
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw_line.strip().split()
        if len(parts) < 5:
            continue
        try:
            class_id = int(float(parts[0]))
            values = [float(value) for value in parts[1:]]
        except ValueError:
            continue
        box = _label_geometry_to_box(values)
        if box is None:
            continue
        source_kind = "box" if len(values) == 4 else "polygon"
        yield class_id, box, source_kind


def format_detection_box(class_id: int, box: tuple[float, float, float, float]) -> str:
    cx, cy, bw, bh = box
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def _pixel_polygon_to_unit_box(
    points: str,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    coords: list[float] = []
    for pair in points.split(";"):
        pair = pair.strip()
        if not pair:
            continue
        try:
            x_raw, y_raw = pair.split(",", 1)
            coords.extend((float(x_raw) / width, float(y_raw) / height))
        except ValueError:
            return None
    return _label_geometry_to_box(coords)


def validate_split_ratios(val_ratio: float, test_ratio: float) -> None:
    if val_ratio <= 0 or test_ratio <= 0 or (val_ratio + test_ratio) >= 1:
        raise SystemExit(
            "val_ratio and test_ratio must be positive and sum to less than 1."
        )


def stratified_split(
    class_images: dict[str, list[Path]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, dict[str, list[Path]]]:
    splits: dict[str, dict[str, list[Path]]] = {"train": {}, "val": {}, "test": {}}
    for class_name, images in class_images.items():
        if len(images) < 3:
            raise SystemExit(
                f"Need at least 3 images for class {class_name!r} to create train/val/test splits."
            )
        train_val, test = train_test_split(
            images, test_size=test_ratio, random_state=seed
        )
        val_adj = val_ratio / (1.0 - test_ratio)
        train, val = train_test_split(train_val, test_size=val_adj, random_state=seed)
        splits["train"][class_name] = list(train)
        splits["val"][class_name] = list(val)
        splits["test"][class_name] = list(test)
    return splits


def copy_images(
    splits: dict[str, dict[str, list[Path]]],
    dest_root: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, int]:
    collision_counts: dict[str, int] = defaultdict(int)
    for split_name, class_map in splits.items():
        for class_name, paths in class_map.items():
            target_dir = dest_root / split_name / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in paths:
                target = target_dir / _safe_name(src, source_root)
                if target.exists():
                    collision_counts[f"{split_name}/{class_name}"] += 1
                    stem, suffix = target.stem, target.suffix
                    target = (
                        target_dir
                        / f"{stem}__{abs(hash(str(src))) & 0xFFFFFF:06x}{suffix}"
                    )
                shutil.copy2(src, target)
    return dict(collision_counts)


def copy_test_flat(
    class_map: dict[str, list[Path]],
    dest_root: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, int]:
    collisions: dict[str, int] = defaultdict(int)
    for class_name, paths in class_map.items():
        target_dir = dest_root / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for src in paths:
            target = target_dir / _safe_name(src, source_root)
            if target.exists():
                collisions[class_name] += 1
                stem, suffix = target.stem, target.suffix
                target = (
                    target_dir / f"{stem}__{abs(hash(str(src))) & 0xFFFFFF:06x}{suffix}"
                )
            shutil.copy2(src, target)
    return dict(collisions)


def summarize_dimensions(
    paths: Iterable[Path], sample_limit: int = 25
) -> dict[str, object]:
    widths: list[int] = []
    heights: list[int] = []
    for path in list(paths)[:sample_limit]:
        with Image.open(path) as img:
            width, height = img.size
        widths.append(width)
        heights.append(height)
    if not widths:
        return {"sampled": 0}
    return {
        "sampled": len(widths),
        "width": {"min": min(widths), "max": max(widths)},
        "height": {"min": min(heights), "max": max(heights)},
    }


def report_duplicate_sources(class_images: dict[str, list[Path]]) -> list[str]:
    seen: Counter[str] = Counter()
    for paths in class_images.values():
        for path in paths:
            seen[str(path.resolve())] += 1
    return [path for path, count in seen.items() if count > 1]


def summarize_label_counts(counts: Iterable[int]) -> dict[str, object]:
    counter = Counter(counts)
    return {
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "distribution": dict(sorted(counter.items())),
    }


def write_detection_report(report_path: Path, summary: dict[str, object]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def _source_split_name(split_name: str) -> str:
    return _SPLIT_ALIASES.get(split_name, split_name)


def _boxes_for_task(
    label_path: Path,
    *,
    stage1_mode: bool,
) -> tuple[list[str], int, int, int]:
    lines: list[str] = []
    polygon_labels = 0
    duplicate_boxes = 0
    tiny_boxes = 0
    seen_boxes: set[tuple[float, float, float, float]] = set()

    for class_id, box, source_kind in iter_detection_boxes(label_path):
        if stage1_mode:
            mapped_class = 0
        else:
            if class_id not in (_ROBOFLOW_FREE_IDS | _ROBOFLOW_OCC_IDS):
                continue
            mapped_class = class_id

        rounded_box = tuple(round(value, 6) for value in box)
        if rounded_box in seen_boxes:
            duplicate_boxes += 1
            continue
        seen_boxes.add(rounded_box)

        if box[2] * box[3] < _TINY_BOX_AREA:
            tiny_boxes += 1
            continue

        lines.append(format_detection_box(mapped_class, box))
        if source_kind == "polygon":
            polygon_labels += 1

    return lines, polygon_labels, duplicate_boxes, tiny_boxes


def _discover_detection_records(
    root: Path,
    *,
    stage1_mode: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    duplicates_removed = 0
    empty_excluded = 0
    polygon_labels = 0
    duplicate_boxes = 0
    tiny_boxes = 0
    scene_rule_counts: Counter[str] = Counter()
    seen_normalized: set[str] = set()

    for source_split, image_dir, label_dir in _roboflow_splits(root):
        for image_path in _image_files(image_dir):
            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            normalized_stem = normalize_source_stem(image_path.name)
            if normalized_stem in seen_normalized:
                duplicates_removed += 1
                continue
            seen_normalized.add(normalized_stem)

            lines, poly_count, dup_count, tiny_count = _boxes_for_task(
                label_path,
                stage1_mode=stage1_mode,
            )
            polygon_labels += poly_count
            duplicate_boxes += dup_count
            tiny_boxes += tiny_count

            if not lines:
                empty_excluded += 1
                continue

            scene_id, scene_rule = scene_id_from_frame(
                normalized_stem,
                box_count=len(lines),
                image_path=image_path.relative_to(root),
            )
            scene_rule_counts[scene_rule] += 1
            records.append(
                {
                    "image_path": image_path,
                    "label_path": label_path,
                    "normalized_stem": normalized_stem,
                    "scene_id": scene_id,
                    "scene_rule": scene_rule,
                    "source_split": _source_split_name(source_split),
                    "label_lines": lines,
                    "box_count": len(lines),
                }
            )

    records.sort(key=lambda item: str(item["image_path"]))
    return records, {
        "duplicates_removed": duplicates_removed,
        "empty_label_frames_excluded": empty_excluded,
        "polygon_labels_converted": polygon_labels,
        "duplicate_boxes_excluded": duplicate_boxes,
        "tiny_boxes_excluded": tiny_boxes,
        "scene_id_rules": dict(sorted(scene_rule_counts.items())),
    }


def _discover_parking_space_detection_records(
    root: Path,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    annotations_path = root / "annotations.xml"
    images_dir = root / "images"
    if not annotations_path.exists():
        raise SystemExit(
            f"Parking Space Detection Dataset annotations.xml not found in {root}"
        )
    if not images_dir.exists():
        raise SystemExit(f"Parking Space Detection Dataset images/ not found in {root}")

    tree = ET.parse(annotations_path)
    records: list[dict[str, object]] = []
    empty_excluded = 0
    polygon_labels = 0
    tiny_boxes = 0
    class_counts: Counter[str] = Counter()
    scene_rule_counts: Counter[str] = Counter()

    for image_node in tree.getroot().findall("image"):
        image_rel = image_node.attrib.get("name", "")
        image_path = root / image_rel
        if not image_path.exists():
            continue

        width = int(image_node.attrib["width"])
        height = int(image_node.attrib["height"])
        lines: list[str] = []
        seen_boxes: set[tuple[float, float, float, float]] = set()

        for polygon in image_node.findall("polygon"):
            label = polygon.attrib.get("label", "").strip()
            if label not in {
                "free_parking_space",
                "not_free_parking_space",
                "partially_free_parking_space",
            }:
                continue
            box = _pixel_polygon_to_unit_box(
                polygon.attrib.get("points", ""),
                width=width,
                height=height,
            )
            if box is None:
                continue
            if box[2] * box[3] < _TINY_BOX_AREA:
                tiny_boxes += 1
                continue
            rounded_box = tuple(round(value, 6) for value in box)
            if rounded_box in seen_boxes:
                continue
            seen_boxes.add(rounded_box)
            lines.append(format_detection_box(0, box))
            polygon_labels += 1
            class_counts[label] += 1

        if not lines:
            empty_excluded += 1
            continue

        normalized_stem = f"parking_space_dataset__{image_path.stem}"
        scene_id = normalized_stem
        scene_rule = "parking_space_dataset_image"
        scene_rule_counts[scene_rule] += 1
        records.append(
            {
                "image_path": image_path,
                "label_path": annotations_path,
                "normalized_stem": normalized_stem,
                "scene_id": scene_id,
                "scene_rule": scene_rule,
                "source_split": "external",
                "label_lines": lines,
                "box_count": len(lines),
            }
        )

    records.sort(key=lambda item: str(item["image_path"]))
    return records, {
        "duplicates_removed": 0,
        "empty_label_frames_excluded": empty_excluded,
        "polygon_labels_converted": polygon_labels,
        "duplicate_boxes_excluded": 0,
        "tiny_boxes_excluded": tiny_boxes,
        "scene_id_rules": dict(sorted(scene_rule_counts.items())),
        "source_label_counts": dict(sorted(class_counts.items())),
    }


def assign_scene_splits(
    records: list[dict[str, object]],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, object]]]:
    scenes: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        scenes[str(record["scene_id"])].append(record)

    scene_ids = sorted(scenes)
    if len(scene_ids) < 3:
        raise SystemExit(
            f"Need at least 3 scene groups for scene holdout splitting, found {len(scene_ids)}."
        )

    train_scene_ids, test_scene_ids = train_test_split(
        scene_ids, test_size=test_ratio, random_state=seed
    )
    val_adj = val_ratio / (1.0 - test_ratio)
    train_scene_ids, val_scene_ids = train_test_split(
        train_scene_ids, test_size=val_adj, random_state=seed
    )

    split_map = {
        "train": set(train_scene_ids),
        "val": set(val_scene_ids),
        "test": set(test_scene_ids),
    }
    return {
        split: [
            record
            for record in records
            if str(record["scene_id"]) in scene_ids_for_split
        ]
        for split, scene_ids_for_split in split_map.items()
    }


def _detection_audit(records: list[dict[str, object]]) -> dict[str, object]:
    counts_by_scene: dict[str, list[int]] = defaultdict(list)
    ordered_by_scene: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        scene_id = str(record["scene_id"])
        counts_by_scene[scene_id].append(int(record["box_count"]))
        ordered_by_scene[scene_id].append(record)

    suspicious_low = 0
    count_jumps = 0
    for scene_id, counts in counts_by_scene.items():
        scene_median = median(counts)
        suspicious_low += sum(
            1 for count in counts if scene_median and count < (scene_median * 0.5)
        )
        prior = None
        for record in sorted(
            ordered_by_scene[scene_id], key=lambda item: str(item["normalized_stem"])
        ):
            current = int(record["box_count"])
            if prior is not None and abs(current - prior) >= max(10, int(prior * 0.5)):
                count_jumps += 1
            prior = current

    return {
        "scene_median_box_counts": {
            scene_id: round(float(median(counts)), 2)
            for scene_id, counts in sorted(counts_by_scene.items())
        },
        "large_count_jumps": count_jumps,
        "suspiciously_low_count_frames": suspicious_low,
    }


def _split_scene_summary(
    records_by_split: dict[str, list[dict[str, object]]]
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for split, records in records_by_split.items():
        scene_ids = sorted({str(record["scene_id"]) for record in records})
        normalized_ids = sorted({str(record["normalized_stem"]) for record in records})
        counts = [int(record["box_count"]) for record in records]
        summary[split] = {
            "images_kept": len(records),
            "boxes_kept": sum(counts),
            "scene_count": len(scene_ids),
            "scene_ids": scene_ids,
            "normalized_frame_ids": normalized_ids,
            "kept_label_count_summary": summarize_label_counts(counts),
            "source_split_counts": dict(
                sorted(
                    Counter(str(record["source_split"]) for record in records).items()
                )
            ),
        }
    return summary


def _leakage_summary(split_summary: dict[str, dict[str, object]]) -> dict[str, object]:
    split_names = ("train", "val", "test")
    scene_overlap: dict[str, list[str]] = {}
    frame_overlap: dict[str, list[str]] = {}
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            key = f"{left}-{right}"
            left_scenes = set(split_summary[left]["scene_ids"])
            right_scenes = set(split_summary[right]["scene_ids"])
            left_frames = set(split_summary[left]["normalized_frame_ids"])
            right_frames = set(split_summary[right]["normalized_frame_ids"])
            scene_overlap[key] = sorted(left_scenes & right_scenes)
            frame_overlap[key] = sorted(left_frames & right_frames)
    return {
        "scene_overlap": scene_overlap,
        "normalized_frame_overlap": frame_overlap,
        "scene_leakage_detected": any(scene_overlap.values()),
        "frame_leakage_detected": any(frame_overlap.values()),
    }


def sanity_check_stage2(
    combined: dict[str, dict[str, list[Path]]],
    *,
    all_images: dict[str, list[Path]],
    collisions: dict[str, int],
    report_path: Path,
    scene_split_summary: dict[str, object] | None = None,
) -> None:
    duplicate_sources = report_duplicate_sources(all_images)
    summary = {
        "class_counts": {cls: len(paths) for cls, paths in all_images.items()},
        "splits": {
            split: {cls: len(paths) for cls, paths in class_map.items()}
            for split, class_map in combined.items()
        },
        "duplicate_sources": duplicate_sources,
        "copy_collisions": collisions,
        "dimension_summary": {
            cls: summarize_dimensions(paths) for cls, paths in all_images.items()
        },
    }
    if scene_split_summary is not None:
        summary["scene_holdout"] = scene_split_summary

    missing = [
        f"{split}/{cls}"
        for split, class_map in combined.items()
        for cls, paths in class_map.items()
        if not paths
    ]
    if missing:
        raise SystemExit(f"Empty Stage 2 split detected: {', '.join(missing)}")

    if duplicate_sources:
        print(f"[warn] duplicate source paths detected: {len(duplicate_sources)}")
    if collisions:
        print(f"[warn] filename collisions handled during copy: {collisions}")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[Stage 2] Sanity checks")
    print(f"  class counts : {summary['class_counts']}")
    print(f"  collisions   : {collisions or 'none'}")
    print(f"  report       : {report_path}")
    for cls, dims in summary["dimension_summary"].items():
        print(f"  dims {cls:8s}: {dims}")


def _write_detection_dataset(
    records_by_split: dict[str, list[dict[str, object]]],
    *,
    out_dir: Path,
) -> None:
    for split, records in records_by_split.items():
        out_image = out_dir / split / "images"
        out_label = out_dir / split / "labels"
        out_image.mkdir(parents=True, exist_ok=True)
        out_label.mkdir(parents=True, exist_ok=True)
        for record in records:
            image_path = Path(str(record["image_path"]))
            target_name = f"{record['normalized_stem']}{image_path.suffix.lower()}"
            shutil.copy2(image_path, out_image / target_name)
            (out_label / f"{record['normalized_stem']}.txt").write_text(
                "\n".join(record["label_lines"]) + "\n",
                encoding="utf-8",
            )


def _write_detection_yaml(out_dir: Path, yaml_path: Path, *, names: list[str]) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "path": str(out_dir.resolve()),
                "train": "train/images",
                "val": "val/images",
                "test": "test/images",
                "nc": len(names),
                "names": names,
            },
            f,
            default_flow_style=False,
            sort_keys=False,
        )


def _prepare_detection_dataset(
    pklot_dir: Path,
    out_dir: Path,
    yaml_path: Path,
    *,
    parking_space_dir: Path | None,
    stage1_mode: bool,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, object]:
    validate_split_ratios(val_ratio, test_ratio)
    records, discovery = _discover_detection_records(pklot_dir, stage1_mode=stage1_mode)
    source_summaries: dict[str, dict[str, object]] = {"pklot": discovery}
    if stage1_mode and parking_space_dir is not None:
        extra_records, extra_discovery = _discover_parking_space_detection_records(
            parking_space_dir
        )
        records.extend(extra_records)
        records.sort(key=lambda item: str(item["image_path"]))
        source_summaries["parking_space_dataset"] = extra_discovery
    if not records:
        raise SystemExit(f"No labeled detection frames found in {pklot_dir}")

    records_by_split = assign_scene_splits(
        records,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_summary = _split_scene_summary(records_by_split)
    leakage = _leakage_summary(split_summary)
    audit = _detection_audit(records)

    _write_detection_dataset(records_by_split, out_dir=out_dir)
    _write_detection_yaml(
        out_dir, yaml_path, names=["space"] if stage1_mode else ["free", "occupied"]
    )

    report = {
        "track": "stage1" if stage1_mode else "single_model",
        "split_strategy": "scene_holdout",
        "annotation_policy": {
            "classes": ["space"] if stage1_mode else ["free", "occupied"],
            "one_box_per_slot": True,
            "polygon_conversion": "polygon_bounds_to_tight_box",
            "ignore_rules": [
                "slots with insufficient visible geometry",
                "slots mostly outside the frame",
                "slots too occluded to localize consistently",
            ],
        },
        "scene_id_strategy": {
            "source": "path metadata when available, otherwise deterministic filename heuristics",
            "rules_used": discovery["scene_id_rules"],
        },
        "images_kept_total": len(records),
        "boxes_kept_total": sum(int(record["box_count"]) for record in records),
        "duplicates_removed": discovery["duplicates_removed"],
        "empty_label_frames_excluded": discovery["empty_label_frames_excluded"],
        "polygon_labels_converted": discovery["polygon_labels_converted"],
        "sources": source_summaries,
        "audit": {
            "duplicate_boxes_excluded": discovery["duplicate_boxes_excluded"],
            "tiny_boxes_excluded": discovery["tiny_boxes_excluded"],
            **audit,
        },
        "splits": split_summary,
        "leakage_checks": leakage,
    }
    write_detection_report(out_dir / DETECTION_REPORT, report)
    return report


def prepare_stage1(
    pklot_dir: Path,
    out_dir: Path,
    yaml_path: Path,
    *,
    parking_space_dir: Path | None = None,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    print(
        f"\n[Stage 1] Building scene-held-out full-frame slot detector dataset -> {out_dir}/"
    )
    report = _prepare_detection_dataset(
        pklot_dir,
        out_dir,
        yaml_path,
        parking_space_dir=parking_space_dir,
        stage1_mode=True,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    print(
        f"  total       : {report['images_kept_total']} images  {report['boxes_kept_total']} boxes"
    )
    print(f"  yaml        : {yaml_path}")
    print(f"  report      : {out_dir / DETECTION_REPORT}")


def prepare_single_model_detection(
    pklot_dir: Path,
    out_dir: Path,
    yaml_path: Path,
    *,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> None:
    print(
        f"\n[Single Model] Building scene-held-out occupancy detection baseline -> {out_dir}/"
    )
    report = _prepare_detection_dataset(
        pklot_dir,
        out_dir,
        yaml_path,
        parking_space_dir=None,
        stage1_mode=False,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    print(
        f"  total       : {report['images_kept_total']} images  {report['boxes_kept_total']} boxes"
    )
    print(f"  yaml        : {yaml_path}")
    print(f"  report      : {out_dir / DETECTION_REPORT}")


def _crop_patch(
    image_path: Path,
    box_line: str,
    target: Path,
) -> None:
    _, cx_raw, cy_raw, bw_raw, bh_raw = box_line.split()
    cx, cy, bw, bh = (float(cx_raw), float(cy_raw), float(bw_raw), float(bh_raw))
    with Image.open(image_path) as img:
        width, height = img.size
        x1 = max(0, int((cx - bw / 2) * width))
        y1 = max(0, int((cy - bh / 2) * height))
        x2 = min(width, int((cx + bw / 2) * width))
        y2 = min(height, int((cy + bh / 2) * height))
        if x2 <= x1 or y2 <= y1:
            return
        img.crop((x1, y1, x2, y2)).save(target)


def _polygon_points_to_box_line(
    polygon: list[list[float]],
    *,
    class_id: int,
) -> str | None:
    coords: list[float] = []
    for point in polygon:
        if len(point) != 2:
            return None
        x, y = point
        try:
            coords.extend((float(x), float(y)))
        except (TypeError, ValueError):
            return None
    box = _label_geometry_to_box(coords)
    if box is None:
        return None
    return format_detection_box(class_id, box)


def _is_roi_annotations_dataset(root: Path) -> bool:
    return (root / "annotations.json").exists() and (root / "images").exists()


def prepare_stage2_from_roi_annotations(args: argparse.Namespace) -> None:
    source_root = Path(args.pklot_dir)
    annotations_path = source_root / "annotations.json"
    images_dir = source_root / "images"
    stage2_output = Path(args.stage2_output)

    if not annotations_path.exists():
        raise SystemExit(f"ROI annotations file not found: {annotations_path}")
    if not images_dir.exists():
        raise SystemExit(f"ROI images directory not found: {images_dir}")

    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    combined: dict[str, dict[str, list[Path]]] = {
        "train": {"free": [], "occupied": []},
        "val": {"free": [], "occupied": []},
        "test": {"free": [], "occupied": []},
    }
    all_images: dict[str, list[Path]] = {"free": [], "occupied": []}
    invalid_polygons = 0
    missing_images: list[str] = []

    patch_cache = (
        Path(args.patch_cache)
        if args.patch_cache
        else source_root / "stage2_patch_cache"
    )
    patch_cache.mkdir(parents=True, exist_ok=True)
    print(f"\n[Stage 2] Cropping ROI patches from annotations.json -> {patch_cache}/")

    for source_split, payload in data.items():
        split_name = _ROI_SPLIT_ALIASES.get(source_split.lower())
        if split_name is None:
            continue
        file_names = payload.get("file_names", [])
        rois_list = payload.get("rois_list", [])
        occupancy_list = payload.get("occupancy_list", [])
        if not (len(file_names) == len(rois_list) == len(occupancy_list)):
            raise SystemExit(
                f"ROI annotation split {source_split!r} has mismatched lengths: "
                f"files={len(file_names)} rois={len(rois_list)} occupancy={len(occupancy_list)}"
            )

        split_counts = {"free": 0, "occupied": 0}
        for file_name, polygons, occupancies in zip(
            file_names, rois_list, occupancy_list
        ):
            image_path = images_dir / str(file_name)
            if not image_path.exists():
                missing_images.append(str(file_name))
                continue
            if len(polygons) != len(occupancies):
                raise SystemExit(
                    f"ROI annotation image {file_name!r} has mismatched polygon/occupancy lengths: "
                    f"{len(polygons)} vs {len(occupancies)}"
                )
            normalized_stem = normalize_source_stem(str(file_name))
            for index, (polygon, occupied) in enumerate(zip(polygons, occupancies)):
                class_name = "occupied" if bool(occupied) else "free"
                class_id = 1 if class_name == "occupied" else 0
                box_line = _polygon_points_to_box_line(polygon, class_id=class_id)
                if box_line is None:
                    invalid_polygons += 1
                    continue
                target = (
                    patch_cache
                    / split_name
                    / class_name
                    / f"{normalized_stem}__{index:04d}.jpg"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                _crop_patch(image_path, box_line, target)
                if target.exists():
                    combined[split_name][class_name].append(target)
                    all_images[class_name].append(target)
                    split_counts[class_name] += 1
        print(
            f"  {split_name:5s}: {split_counts['free']:7d} free  "
            f"{split_counts['occupied']:7d} occupied"
        )

    print(f"\n[Stage 2] Writing ROI classification dataset -> {stage2_output}/")
    collisions = copy_images(combined, stage2_output, source_root=patch_cache)
    pklot_collisions = copy_test_flat(
        combined["test"],
        Path(args.pklot_test_output),
        source_root=patch_cache,
    )
    sanity_check_stage2(
        combined,
        all_images=all_images,
        collisions={
            **collisions,
            **{f"pklot_test/{k}": v for k, v in pklot_collisions.items()},
        },
        report_path=stage2_output / SANITY_REPORT,
        scene_split_summary={
            "source": "roi_annotations_presplit",
            "scene_counts": {split: 0 for split in ("train", "val", "test")},
            "leakage_checks": {
                "scene_leakage_detected": False,
                "shared_scene_ids": [],
                "shared_normalized_stems": [],
            },
        },
    )

    report_path = stage2_output / SANITY_REPORT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source"] = str(source_root.resolve())
    report["split_strategy"] = "source_presplit"
    report["annotation_source"] = "annotations.json"
    report["invalid_polygons_skipped"] = invalid_polygons
    report["missing_images"] = missing_images
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n[Stage 2] Split summary")
    for split_name in ("train", "val", "test"):
        counts = {cls: len(paths) for cls, paths in combined[split_name].items()}
        print(f"  {split_name:5s}: {counts}")
    print(
        f"  pklot_test : {{'free': {len(combined['test']['free'])}, 'occupied': {len(combined['test']['occupied'])}}}"
    )
    print(f"  report     : {report_path}")


def collect_roboflow_patches(root: Path, patch_output: Path) -> dict[str, list[Path]]:
    patches: dict[str, list[Path]] = {"free": [], "occupied": []}
    print(f"\n[Stage 2] Cropping PKLot patches -> {patch_output}/")
    records, _ = _discover_detection_records(root, stage1_mode=False)
    for record in records:
        split_name = str(record["source_split"])
        image_path = Path(str(record["image_path"]))
        for index, line in enumerate(record["label_lines"]):
            class_name = "free" if line.startswith("0 ") else "occupied"
            target = (
                patch_output
                / split_name
                / class_name
                / f"{record['normalized_stem']}__{index:04d}.jpg"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            _crop_patch(image_path, line, target)
            if target.exists():
                patches[class_name].append(target)

    for split_name in ("train", "val", "test"):
        split_counts = {
            class_name: len(
                list((patch_output / split_name / class_name).glob("*.jpg"))
            )
            for class_name in ("free", "occupied")
        }
        if any(split_counts.values()):
            print(
                f"  {split_name:5s}: {split_counts['free']:7d} free  "
                f"{split_counts['occupied']:7d} occupied"
            )
    return patches


def collect_legacy_patches(root: Path) -> dict[str, list[Path]]:
    patches: dict[str, list[Path]] = {"free": [], "occupied": []}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parent = path.parent.name.lower()
        if parent in _FREE_DIRS:
            patches["free"].append(path)
        elif parent in _OCC_DIRS:
            patches["occupied"].append(path)
    return patches


def _cnr_weather_from_path(path: Path) -> str | None:
    for part in path.parts:
        lowered = part.lower()
        if lowered == "sunny":
            return "sunny"
        if lowered == "overcast":
            return "cloudy"
        if lowered == "cloudy":
            return "cloudy"
        if lowered == "rainy":
            return "rainy"
    return None


def _empty_weather_map() -> dict[str, dict[str, list[Path]]]:
    return {
        "sunny": {"free": [], "occupied": []},
        "cloudy": {"free": [], "occupied": []},
        "rainy": {"free": [], "occupied": []},
    }


def _append_weather_path(
    weather_map: dict[str, dict[str, list[Path]]],
    path: Path,
    class_name: str,
) -> None:
    weather = _cnr_weather_from_path(path)
    if weather:
        weather_map[weather][class_name].append(path)


def _resolve_cnr_relative_path(root: Path, rel_path: str) -> Path | None:
    normalized = rel_path.strip().lstrip("./").replace("\\", "/")
    candidates = [
        root / normalized,
        root / "PATCHES" / normalized,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def collect_cnrpark_patches(
    root: Path,
) -> tuple[dict[str, list[Path]], dict[str, dict[str, list[Path]]]]:
    labels_dir = root / "LABELS"
    weather_map = _empty_weather_map()
    if labels_dir.exists():
        labeled_paths: dict[Path, str] = {}
        for list_path in sorted(labels_dir.rglob("*.txt")):
            for raw_line in list_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                rel_path, label = line.rsplit(maxsplit=1)
                if label not in {"0", "1"}:
                    continue
                image_path = _resolve_cnr_relative_path(root, rel_path)
                if image_path is None or image_path.suffix.lower() not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                }:
                    continue
                class_name = "free" if label == "0" else "occupied"
                previous = labeled_paths.get(image_path)
                if previous is not None and previous != class_name:
                    raise SystemExit(
                        f"Conflicting labels for CNR image {image_path}: {previous} vs {class_name}"
                    )
                labeled_paths[image_path] = class_name

        patches = {"free": [], "occupied": []}
        for image_path, class_name in sorted(labeled_paths.items()):
            patches[class_name].append(image_path)
            _append_weather_path(weather_map, image_path, class_name)
        return patches, weather_map

    patches = collect_legacy_patches(root)
    for class_name, paths in patches.items():
        for path in paths:
            _append_weather_path(weather_map, path, class_name)
    return patches, weather_map


def copy_weather_flat(
    weather_images: dict[str, dict[str, list[Path]]],
    dest_root: Path,
    *,
    source_root: Path | None = None,
) -> dict[str, int]:
    collisions: dict[str, int] = defaultdict(int)
    for weather, class_map in weather_images.items():
        for class_name, paths in class_map.items():
            target_dir = dest_root / weather / class_name
            target_dir.mkdir(parents=True, exist_ok=True)
            for src in paths:
                target = target_dir / _safe_name(src, source_root)
                if target.exists():
                    collisions[f"{weather}/{class_name}"] += 1
                    stem, suffix = target.stem, target.suffix
                    target = (
                        target_dir
                        / f"{stem}__{abs(hash(str(src))) & 0xFFFFFF:06x}{suffix}"
                    )
                shutil.copy2(src, target)
    return dict(collisions)


def prepare_stage2(args: argparse.Namespace) -> None:
    if _is_roi_annotations_dataset(Path(args.pklot_dir)):
        prepare_stage2_from_roi_annotations(args)
        return

    validate_split_ratios(args.val_ratio, args.test_ratio)
    pklot_dir = Path(args.pklot_dir)
    patch_cache = (
        Path(args.patch_cache)
        if args.patch_cache
        else pklot_dir.parent / f"{pklot_dir.name}_patches"
    )
    stage2_output = Path(args.stage2_output)

    records, discovery = _discover_detection_records(pklot_dir, stage1_mode=False)
    if not records:
        raise SystemExit(
            "No PKLot patches were created. Check the Roboflow export layout and labels."
        )
    records_by_split = assign_scene_splits(
        records,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    pklot_splits = {
        "train": {"free": [], "occupied": []},
        "val": {"free": [], "occupied": []},
        "test": {"free": [], "occupied": []},
    }
    patch_cache.mkdir(parents=True, exist_ok=True)
    print(
        f"\n[Stage 2] Cropping PKLot patches with inherited scene holdout -> {patch_cache}/"
    )
    for split_name, split_records in records_by_split.items():
        split_counts = {"free": 0, "occupied": 0}
        for record in split_records:
            image_path = Path(str(record["image_path"]))
            for index, line in enumerate(record["label_lines"]):
                class_name = "free" if line.startswith("0 ") else "occupied"
                target = (
                    patch_cache
                    / split_name
                    / class_name
                    / f"{record['normalized_stem']}__{index:04d}.jpg"
                )
                target.parent.mkdir(parents=True, exist_ok=True)
                _crop_patch(image_path, line, target)
                if target.exists():
                    pklot_splits[split_name][class_name].append(target)
                    split_counts[class_name] += 1
        print(
            f"  {split_name:5s}: {split_counts['free']:7d} free  "
            f"{split_counts['occupied']:7d} occupied"
        )

    cnrpark_splits = {
        "train": {"free": [], "occupied": []},
        "val": {"free": [], "occupied": []},
        "test": {"free": [], "occupied": []},
    }
    cnr_source_root: Path | None = None
    cnr_images = {"free": [], "occupied": []}
    cnr_weather = _empty_weather_map()
    if args.cnrpark_dir:
        cnr_source_root = Path(args.cnrpark_dir)
        if not cnr_source_root.exists():
            raise SystemExit(f"CNRPark directory not found: {cnr_source_root}")
        cnr_images, cnr_weather = collect_cnrpark_patches(cnr_source_root)
        print(
            f"\n[Stage 2] Collected CNRPark patches: free={len(cnr_images['free'])} occupied={len(cnr_images['occupied'])}"
        )
        if any(cnr_images.values()):
            cnrpark_splits = stratified_split(
                cnr_images,
                val_ratio=args.val_ratio,
                test_ratio=args.test_ratio,
                seed=args.seed,
            )

    combined: dict[str, dict[str, list[Path]]] = {}
    for split in ("train", "val", "test"):
        combined[split] = {}
        for class_name in ("free", "occupied"):
            combined[split][class_name] = pklot_splits[split].get(
                class_name, []
            ) + cnrpark_splits[split].get(class_name, [])

    print(f"\n[Stage 2] Writing combined dataset -> {stage2_output}/")
    collisions = copy_images(combined, stage2_output, source_root=patch_cache)
    pklot_collisions = copy_test_flat(
        pklot_splits["test"],
        Path(args.pklot_test_output),
        source_root=patch_cache,
    )
    if args.cnrpark_dir and any(cnr_images.values()):
        cnr_collisions = copy_test_flat(
            cnrpark_splits["test"],
            Path(args.cnrpark_test_output),
            source_root=cnr_source_root,
        )
        weather_collisions = copy_weather_flat(
            cnr_weather,
            Path(args.weather_output),
            source_root=cnr_source_root,
        )
    else:
        cnr_collisions = {}
        weather_collisions = {}

    all_images = {
        "free": (
            pklot_splits["train"]["free"]
            + pklot_splits["val"]["free"]
            + pklot_splits["test"]["free"]
            + cnr_images["free"]
        ),
        "occupied": (
            pklot_splits["train"]["occupied"]
            + pklot_splits["val"]["occupied"]
            + pklot_splits["test"]["occupied"]
            + cnr_images["occupied"]
        ),
    }
    split_summary = _split_scene_summary(records_by_split)
    sanity_check_stage2(
        combined,
        all_images=all_images,
        collisions={
            **collisions,
            **{f"pklot_test/{k}": v for k, v in pklot_collisions.items()},
            **{f"cnrpark_test/{k}": v for k, v in cnr_collisions.items()},
            **{f"weather/{k}": v for k, v in weather_collisions.items()},
        },
        report_path=stage2_output / SANITY_REPORT,
        scene_split_summary={
            "source": "pklot_scene_holdout",
            "scene_counts": {
                split: split_summary[split]["scene_count"] for split in split_summary
            },
            "leakage_checks": _leakage_summary(split_summary),
            "duplicates_removed": discovery["duplicates_removed"],
        },
    )

    print("\n[Stage 2] Split summary")
    for split_name in ("train", "val", "test"):
        counts = {cls: len(paths) for cls, paths in combined[split_name].items()}
        print(f"  {split_name:5s}: {counts}")
    pklot_counts = {cls: len(paths) for cls, paths in pklot_splits["test"].items()}
    print(f"  pklot_test : {pklot_counts}")
    if args.cnrpark_dir and any(cnr_images.values()):
        cnr_counts = {cls: len(paths) for cls, paths in cnrpark_splits["test"].items()}
        print(f"  cnrpark_test: {cnr_counts}")
        weather_counts = {
            weather: {cls: len(paths) for cls, paths in class_map.items()}
            for weather, class_map in cnr_weather.items()
            if any(class_map.values())
        }
        if weather_counts:
            print(f"  weather export: {weather_counts}")


def weather_split_paths(root: Path) -> dict[str, Path]:
    paths = {name: root / name for name in ("sunny", "cloudy", "rainy")}
    if not all(path.exists() for path in paths.values()):
        raise SystemExit(
            f"Per-weather evaluation requested but weather splits are unavailable. {WEATHER_CONVENTION}"
        )
    return paths


def main() -> None:
    args = parse_args()
    if not args.stage1 and not args.stage2 and not args.single_model:
        args.stage2 = True

    pklot_dir = Path(args.pklot_dir)
    if not pklot_dir.exists():
        raise SystemExit(f"PKLot directory not found: {pklot_dir}")
    if not _roboflow_splits(pklot_dir) and not _is_roi_annotations_dataset(pklot_dir):
        raise SystemExit(f"No Roboflow train/valid/test splits found in {pklot_dir}")
    parking_space_dir = Path(args.parking_space_dir) if args.parking_space_dir else None
    if parking_space_dir is not None and not parking_space_dir.exists():
        raise SystemExit(
            f"Parking Space Detection Dataset directory not found: {parking_space_dir}"
        )

    if args.stage1:
        prepare_stage1(
            pklot_dir,
            Path(args.stage1_output),
            Path(args.stage1_yaml),
            parking_space_dir=parking_space_dir,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    if args.single_model:
        prepare_single_model_detection(
            pklot_dir,
            Path(args.single_model_output),
            Path(args.single_model_yaml),
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    if args.stage2:
        prepare_stage2(args)


if __name__ == "__main__":
    main()
