#!/usr/bin/env python3
"""Convert ACPDS annotations.json into the normalized manifest format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build datasets/acpds/manifest.json from annotations.json."
    )
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="ACPDS dataset root containing annotations.json and images/.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Manifest output path. Defaults to <dataset-root>/manifest.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    annotations_path = dataset_root / "annotations.json"
    images_dir = dataset_root / "images"
    output_path = Path(args.output) if args.output else dataset_root / "manifest.json"

    if not annotations_path.exists():
        raise SystemExit(f"annotations.json not found: {annotations_path}")
    if not images_dir.exists():
        raise SystemExit(f"images directory not found: {images_dir}")

    data = load_json(annotations_path)
    samples: list[dict] = []
    split_counts = {split: 0 for split in ("train", "val", "test")}

    for source_split, payload in data.items():
        split = SPLIT_ALIASES.get(source_split.lower())
        if split is None:
            continue

        file_names = payload.get("file_names", [])
        rois_list = payload.get("rois_list", [])
        occupancy_list = payload.get("occupancy_list", [])
        if not (len(file_names) == len(rois_list) == len(occupancy_list)):
            raise SystemExit(
                f"Mismatched lengths in split {source_split!r}: "
                f"files={len(file_names)} rois={len(rois_list)} occupancy={len(occupancy_list)}"
            )

        for file_name, polygons, occupancies in zip(
            file_names, rois_list, occupancy_list
        ):
            image_path = images_dir / str(file_name)
            if not image_path.exists():
                raise SystemExit(f"Referenced image missing: {image_path}")
            if len(polygons) != len(occupancies):
                raise SystemExit(
                    f"Mismatched polygon/occupancy counts for {file_name}: "
                    f"{len(polygons)} vs {len(occupancies)}"
                )

            with Image.open(image_path) as image:
                width, height = image.size

            for index, (polygon, occupied) in enumerate(zip(polygons, occupancies)):
                corners = []
                for point in polygon:
                    if len(point) != 2:
                        raise SystemExit(
                            f"Invalid point in {file_name} polygon {index}: {point!r}"
                        )
                    x, y = point
                    corners.append(
                        [round(float(x) * width, 3), round(float(y) * height, 3)]
                    )
                samples.append(
                    {
                        "image": f"images/{file_name}",
                        "split": split,
                        "spot_id": f"{Path(file_name).stem}_{index:04d}",
                        "corners": corners,
                        "occupancy": bool(occupied),
                    }
                )
                split_counts[split] += 1

    manifest = {
        "source": str(annotations_path.resolve()),
        "samples": samples,
        "split_counts": split_counts,
    }
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Samples: {len(samples)}")
    print(f"Split counts: {split_counts}")


if __name__ == "__main__":
    main()
