#!/usr/bin/env python3
"""Extract ACPDS Stage 2 classification patches from a normalized manifest."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from ml.patch_geometry import order_corners, square_patch, warp_patch
except ModuleNotFoundError:
    # Support `python ml/extract_patches.py`, where the repo root is not on sys.path.
    from patch_geometry import order_corners, square_patch, warp_patch

VALID_SPLITS = ("train", "val", "test")
VALID_CLASSES = ("free", "occupied")
DEFAULT_OUTPUT = "datasets/acpds_stage2"
DEFAULT_INDEX = "patch_index.json"
DEFAULT_REPORT = "dataset_report.json"
DEFAULT_VALIDATION_REPORT = "validation_report.json"
DEFAULT_MAP_SAMPLE = "map_sample.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ACPDS quadrilateral patches into a Stage 2 dataset.")
    parser.add_argument("--dataset-root", required=True, help="ACPDS dataset root containing images plus a manifest.")
    parser.add_argument("--manifest", default=None, help="Optional manifest path. Defaults to <dataset-root>/manifest.json.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pooling", choices=["quad", "square"], default="quad")
    parser.add_argument("--validation-samples", type=int, default=20)
    parser.add_argument("--run-validation", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--validation-status",
        choices=["pending", "passed", "failed"],
        default="pending",
        help="Human review result recorded in validation_report.json.",
    )
    return parser.parse_args()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def is_uniform_patch(patch: np.ndarray) -> bool:
    if patch.size == 0:
        return True
    flattened = patch.reshape(-1, patch.shape[-1]) if patch.ndim == 3 else patch.reshape(-1, 1)
    return bool(np.all(flattened == flattened[0]))


def is_uniform_patch(patch: np.ndarray) -> bool:
    if patch.size == 0:
        return True
    flattened = patch.reshape(-1, patch.shape[-1]) if patch.ndim == 3 else patch.reshape(-1, 1)
    return bool(np.all(flattened == flattened[0]))


def _resolve_manifest_path(dataset_root: Path, manifest: str | None) -> Path:
    path = Path(manifest) if manifest else dataset_root / "manifest.json"
    if not path.exists():
        raise SystemExit(f"ACPDS manifest not found: {path}")
    return path


def load_manifest(dataset_root: Path, manifest_path: Path) -> list[dict[str, Any]]:
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "samples" in raw:
        samples = raw["samples"]
    elif isinstance(raw, list):
        samples = raw
    elif isinstance(raw, dict):
        samples = []
        for split, items in raw.items():
            if split not in VALID_SPLITS:
                continue
            for item in items:
                enriched = dict(item)
                enriched.setdefault("split", split)
                samples.append(enriched)
    else:
        raise SystemExit(f"Unsupported manifest structure in {manifest_path}")

    normalized: list[dict[str, Any]] = []
    for item in samples:
        normalized.append(normalize_manifest_item(item, dataset_root))
    return normalized


def normalize_manifest_item(item: dict[str, Any], dataset_root: Path) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("each manifest sample must be an object")
    split = str(item.get("split", "")).lower()
    if split not in VALID_SPLITS:
        raise ValueError(f"invalid split {split!r}")

    image_rel = item.get("image") or item.get("image_path")
    if not image_rel:
        raise ValueError("manifest sample missing image")
    image_path = dataset_root / str(image_rel)

    spot_id = item.get("spot_id")
    if spot_id is None or str(spot_id).strip() == "":
        raise ValueError("manifest sample missing spot_id")

    label = item.get("label")
    occupancy = item.get("occupancy")
    if label is None:
        if isinstance(occupancy, bool):
            label = "occupied" if occupancy else "free"
        elif isinstance(occupancy, str):
            label = occupancy.lower()
    label = str(label).lower()
    if label not in VALID_CLASSES:
        raise ValueError(f"invalid occupancy label {label!r}")

    corners = normalize_corners(item.get("corners"))
    return {
        "split": split,
        "image": str(image_rel),
        "image_path": str(image_path),
        "spot_id": str(spot_id),
        "label": label,
        "corners": corners,
    }


def normalize_corners(corners: Any) -> list[list[float]]:
    # The ACPDS author code expects each ROI to be a quadrilateral whose
    # four points are ordered consistently around the polygon. We canonicalize
    # incoming corners to a stable TL,TR,BR,BL winding here so later warps and
    # any downstream ROI-grid logic operate on the same contract.
    return order_corners(corners).tolist()


def build_patch_filename(entry: dict[str, Any]) -> str:
    return f"{Path(entry['image']).stem}__{entry['spot_id']}.jpg"


def extract_dataset(
    dataset_root: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    size: int,
    seed: int,
    pooling: str = "quad",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    del seed
    entries = load_manifest(dataset_root, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    counts = {split: {label: 0 for label in VALID_CLASSES} for split in VALID_SPLITS}
    source_images = {split: set() for split in VALID_SPLITS}
    invalid_polygons: list[dict[str, Any]] = []
    missing_images: list[str] = []
    uniform_patches: list[dict[str, Any]] = []
    patch_index: list[dict[str, Any]] = []

    for entry in entries:
        image_path = Path(entry["image_path"])
        if not image_path.exists():
            missing_images.append(entry["image"])
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            missing_images.append(entry["image"])
            continue

        try:
            if pooling == "square":
                patch = square_patch(frame, entry["corners"], size=size)
            else:
                patch = warp_patch(frame, entry["corners"], size=size)
        except ValueError as exc:
            invalid_polygons.append({"image": entry["image"], "spot_id": entry["spot_id"], "error": str(exc)})
            continue
        if is_uniform_patch(patch):
            uniform_patches.append(
                {
                    "image": entry["image"],
                    "spot_id": entry["spot_id"],
                    "split": entry["split"],
                    "label": entry["label"],
                }
            )
            continue

        split = str(entry["split"])
        label = str(entry["label"])
        patch_path = output_dir / split / label / build_patch_filename(entry)
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(patch_path), patch):
            raise SystemExit(f"Failed to write patch: {patch_path}")

        counts[split][label] += 1
        source_images[split].add(entry["image"])
        patch_index.append(
            {
                **entry,
                "ordered_corners": entry["corners"],
                "patch_path": str(patch_path),
                "patch_size": [size, size],
            }
        )

    report = {
        "source_root": str(dataset_root.resolve()),
        "manifest": str(manifest_path.resolve()),
        "output": str(output_dir.resolve()),
        "counts": counts,
        "invalid_polygons_skipped": len(invalid_polygons),
        "invalid_polygons": invalid_polygons,
        "uniform_patches_skipped": len(uniform_patches),
        "uniform_patches": uniform_patches,
        "missing_images": sorted(set(missing_images)),
        "unique_source_images": {split: len(paths) for split, paths in source_images.items()},
        "patch_size": {"width": size, "height": size},
        "pooling": pooling,
        "generated_at": utc_now_iso(),
    }
    write_json(output_dir / DEFAULT_INDEX, patch_index)
    write_json(output_dir / DEFAULT_REPORT, report)
    write_map_sample(output_dir, patch_index)
    return patch_index, report


def write_map_sample(output_dir: Path, patch_index: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in patch_index:
        grouped[str(item["image"])].append(item)
    if not grouped:
        return
    image_name = sorted(grouped)[0]
    sample_entries = sorted(grouped[image_name], key=lambda item: item["spot_id"])
    first_image = Path(sample_entries[0]["image_path"])
    frame = cv2.imread(str(first_image))
    if frame is None:
        return
    height, width = frame.shape[:2]
    payload = {
        "image": image_name,
        "image_width": width,
        "image_height": height,
        "spots": [
            {"spot_id": item["spot_id"], "corners": item["ordered_corners"], "label": item["label"]}
            for item in sample_entries
        ],
    }
    write_json(output_dir / DEFAULT_MAP_SAMPLE, payload)


def load_patch_index(output_dir: Path) -> list[dict[str, Any]]:
    path = output_dir / DEFAULT_INDEX
    if not path.exists():
        raise SystemExit(f"Patch index not found: {path}. Run extraction first.")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Invalid patch index format: {path}")
    return data


def validate_patches(
    output_dir: Path,
    *,
    sample_count: int,
    seed: int,
    status: str,
) -> dict[str, Any]:
    patch_index = load_patch_index(output_dir)
    selected = sample_validation_entries(patch_index, sample_count=sample_count, seed=seed)
    validation_dir = output_dir / "validation_samples"
    validation_dir.mkdir(parents=True, exist_ok=True)

    report_items: list[dict[str, Any]] = []
    for index, item in enumerate(selected, start=1):
        src = Path(item["patch_path"])
        target = validation_dir / f"{index:02d}__{src.name}"
        shutil.copy2(src, target)
        patch = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
        blank_like = bool(patch is not None and is_uniform_patch(patch))
        report_items.append(
            {
                "source_image": item["image"],
                "source_image_path": item["image_path"],
                "spot_id": item["spot_id"],
                "split": item["split"],
                "label": item["label"],
                "ordered_corners": item["ordered_corners"],
                "output_patch_path": str(target),
                "blank_like": blank_like,
            }
        )

    report = {
        "status": status,
        "sample_count": len(report_items),
        "requested_sample_count": sample_count,
        "seed": seed,
        "generated_at": utc_now_iso(),
        "items": report_items,
        "human_gate": {
            "pass_criteria": "no twisted or blank-looking perspective warps in reviewed samples",
            "fail_criteria": "twisted corners, extreme clipping, or blank output in any reviewed sample",
        },
    }
    write_json(output_dir / DEFAULT_VALIDATION_REPORT, report)
    return report


def sample_validation_entries(
    entries: list[dict[str, Any]],
    *,
    sample_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if sample_count <= 0:
        return []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        buckets[(str(entry["split"]), str(entry["label"]))].append(entry)

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()

    for key in sorted(buckets):
        bucket = list(buckets[key])
        rng.shuffle(bucket)
        candidate = bucket[0] if bucket else None
        if candidate is not None:
            selected.append(candidate)
            seen.add(str(candidate["patch_path"]))

    remaining = [entry for entry in entries if str(entry["patch_path"]) not in seen]
    rng.shuffle(remaining)
    for entry in remaining:
        if len(selected) >= sample_count:
            break
        selected.append(entry)
    return selected[: min(sample_count, len(entries))]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output)
    manifest_path = _resolve_manifest_path(dataset_root, args.manifest)

    if not args.validate_only:
        _, report = extract_dataset(
            dataset_root,
            manifest_path,
            output_dir,
            size=args.size,
            seed=args.seed,
            pooling=args.pooling,
        )
        print(f"[ACPDS] patches written to {output_dir}")
        print(f"[ACPDS] counts: {report['counts']}")
        print(f"[ACPDS] map sample: {output_dir / DEFAULT_MAP_SAMPLE}")

    if args.run_validation or args.validate_only:
        report = validate_patches(
            output_dir,
            sample_count=args.validation_samples,
            seed=args.seed,
            status=args.validation_status,
        )
        print(f"[ACPDS] validation status: {report['status']}")
        print(f"[ACPDS] validation samples: {output_dir / 'validation_samples'}")


if __name__ == "__main__":
    main()
