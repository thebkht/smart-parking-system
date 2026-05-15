#!/usr/bin/env python3
"""Localize a parked car by matching a query photo against per-spot references.

Supported reference inputs:

1. A manifest JSON:
   {
     "spot_1": ["refs/spot_1/a.jpg", "refs/spot_1/b.jpg"],
     "spot_2": "refs/spot_2/a.jpg"
   }

2. A directory layout:
   references/
     spot_1/
       a.jpg
       b.jpg
     spot_2/
       a.jpg

3. Flat files named with a spot prefix:
   references/
     spot_1__a.jpg
     spot_2__b.jpg
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
IGNORED_REFERENCE_DIR_NAMES = {"query_candidates", "unlabeled_pool", "__pycache__"}


@dataclass(frozen=True)
class ReferenceImage:
    spot_id: str
    image_path: Path


@dataclass
class ImageFeatures:
    image_path: Path
    keypoints: list[Any]
    descriptors: np.ndarray | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SIFT + FLANN parking spot localization.")
    parser.add_argument("--query", required=True, help="Query image path.")
    parser.add_argument(
        "--references",
        required=True,
        help="Reference manifest JSON or a directory of per-spot images.",
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument("--ratio-threshold", type=float, default=0.75)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def create_sift_detector():
    if not hasattr(cv2, "SIFT_create"):
        raise SystemExit(
            "OpenCV SIFT support is unavailable in this environment. "
            "Install a build with SIFT enabled, such as opencv-contrib-python."
        )
    return cv2.SIFT_create()


def build_matcher():
    return cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))


def load_grayscale_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise SystemExit(f"Failed to read image: {path}")
    return image


def extract_image_features(detector, image_path: Path) -> ImageFeatures:
    image = load_grayscale_image(image_path)
    keypoints, descriptors = detector.detectAndCompute(image, None)
    return ImageFeatures(image_path=image_path, keypoints=keypoints or [], descriptors=descriptors)


def load_references(references_path: Path) -> list[ReferenceImage]:
    if not references_path.exists():
        raise SystemExit(f"Reference input not found: {references_path}")
    if references_path.is_file():
        return load_reference_manifest(references_path)
    return discover_reference_directory(references_path)


def load_reference_manifest(path: Path) -> list[ReferenceImage]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    base_dir = path.parent
    references: list[ReferenceImage] = []

    if isinstance(raw, dict) and "spots" in raw:
        for item in raw["spots"]:
            if not isinstance(item, dict):
                raise SystemExit(f"Invalid manifest entry in {path}")
            spot_id = str(item.get("spot_id", "")).strip()
            image_values = item.get("images", item.get("image"))
            references.extend(reference_images_for_value(spot_id, image_values, base_dir))
        return references

    if isinstance(raw, dict):
        for spot_id, image_values in raw.items():
            references.extend(reference_images_for_value(str(spot_id), image_values, base_dir))
        return references

    raise SystemExit(f"Unsupported reference manifest structure: {path}")


def reference_images_for_value(spot_id: str, image_values: Any, base_dir: Path) -> list[ReferenceImage]:
    if not spot_id:
        raise SystemExit("Reference manifest entry is missing spot_id.")
    values = image_values if isinstance(image_values, list) else [image_values]
    references: list[ReferenceImage] = []
    for value in values:
        if not value:
            continue
        image_path = (base_dir / str(value)).resolve()
        references.append(ReferenceImage(spot_id=spot_id, image_path=image_path))
    if not references:
        raise SystemExit(f"Reference manifest entry for {spot_id} has no images.")
    return references


def discover_reference_directory(root: Path) -> list[ReferenceImage]:
    references: list[ReferenceImage] = []
    direct_images = list(iter_image_paths(root))
    if direct_images:
        for image_path in direct_images:
            stem = image_path.stem
            if "__" in stem:
                spot_id = stem.split("__", 1)[0]
            else:
                spot_id = stem
            references.append(ReferenceImage(spot_id=spot_id, image_path=image_path.resolve()))
        return references

    for subdir in sorted(path for path in root.iterdir() if path.is_dir()):
        references.extend(discover_references_from_subdir(subdir))
    if not references:
        raise SystemExit(f"No reference images found in {root}")
    return references


def discover_references_from_subdir(root: Path) -> list[ReferenceImage]:
    if root.name in IGNORED_REFERENCE_DIR_NAMES:
        return []

    direct_images = list(iter_image_paths(root))
    if direct_images:
        return [
            ReferenceImage(spot_id=root.name, image_path=image_path.resolve())
            for image_path in direct_images
        ]

    references: list[ReferenceImage] = []
    for subdir in sorted(path for path in root.iterdir() if path.is_dir()):
        references.extend(discover_references_from_subdir(subdir))
    return references


def iter_image_paths(root: Path) -> Iterable[Path]:
    for path in sorted(root.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def ratio_filter(matches: Iterable[list[Any]], ratio_threshold: float) -> list[Any]:
    good_matches: list[Any] = []
    for pair in matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < ratio_threshold * second.distance:
            good_matches.append(first)
    return good_matches


def estimate_inliers(
    query_keypoints: list[Any],
    reference_keypoints: list[Any],
    matches: list[Any],
    ransac_threshold: float,
) -> tuple[bool, int]:
    if len(matches) < 4:
        return False, 0

    src = np.float32([query_keypoints[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst = np.float32([reference_keypoints[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)
    _homography, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_threshold)
    if mask is None:
        return False, 0
    inlier_count = int(mask.ravel().sum())
    return inlier_count >= 4, inlier_count


def evaluate_reference(
    matcher,
    query_features: ImageFeatures,
    reference_features: ImageFeatures,
    *,
    ratio_threshold: float,
    min_matches: int,
    min_inliers: int,
    ransac_threshold: float,
) -> dict[str, object]:
    if query_features.descriptors is None:
        return failure_result(reference_features.image_path, "query_has_no_descriptors")
    if reference_features.descriptors is None:
        return failure_result(reference_features.image_path, "reference_has_no_descriptors")

    matches = matcher.knnMatch(query_features.descriptors, reference_features.descriptors, k=2)
    good_matches = ratio_filter(matches, ratio_threshold)
    if len(good_matches) < min_matches:
        return {
            "reference_image": str(reference_features.image_path),
            "score": 0.0,
            "match_count": len(good_matches),
            "inlier_count": 0,
            "passed": False,
            "failure_reason": "too_few_matches",
        }

    homography_found, inlier_count = estimate_inliers(
        query_features.keypoints,
        reference_features.keypoints,
        good_matches,
        ransac_threshold,
    )
    if not homography_found or inlier_count < min_inliers:
        return {
            "reference_image": str(reference_features.image_path),
            "score": 0.0,
            "match_count": len(good_matches),
            "inlier_count": inlier_count,
            "passed": False,
            "failure_reason": "homography_rejected",
        }

    score = round(float(inlier_count) + (len(good_matches) / 1000.0), 3)
    return {
        "reference_image": str(reference_features.image_path),
        "score": score,
        "match_count": len(good_matches),
        "inlier_count": inlier_count,
        "passed": True,
    }


def failure_result(reference_image: Path, reason: str) -> dict[str, object]:
    return {
        "reference_image": str(reference_image),
        "score": 0.0,
        "match_count": 0,
        "inlier_count": 0,
        "passed": False,
        "failure_reason": reason,
    }


def localize_query(
    query_path: Path,
    references_path: Path,
    *,
    ratio_threshold: float,
    min_matches: int,
    min_inliers: int,
    ransac_threshold: float,
    top_k: int,
) -> dict[str, object]:
    started_at = time.perf_counter()
    detector = create_sift_detector()
    matcher = build_matcher()
    query_features = extract_image_features(detector, query_path)
    references = load_references(references_path)

    candidates: list[dict[str, object]] = []
    for reference in references:
        result = evaluate_reference(
            matcher,
            query_features,
            extract_image_features(detector, reference.image_path),
            ratio_threshold=ratio_threshold,
            min_matches=min_matches,
            min_inliers=min_inliers,
            ransac_threshold=ransac_threshold,
        )
        result["spot_id"] = reference.spot_id
        candidates.append(result)

    candidates.sort(
        key=lambda item: (
            bool(item["passed"]),
            float(item["score"]),
            int(item["inlier_count"]),
            int(item["match_count"]),
        ),
        reverse=True,
    )
    best = candidates[0] if candidates else None
    elapsed_ms = round((time.perf_counter() - started_at) * 1000.0, 2)
    return {
        "query_image": str(query_path.resolve()),
        "spot_id": best["spot_id"] if best and best["passed"] else None,
        "score": best["score"] if best else 0.0,
        "match_count": best["match_count"] if best else 0,
        "inlier_count": best["inlier_count"] if best else 0,
        "elapsed_ms": elapsed_ms,
        "failure_reason": None if best and best["passed"] else (best["failure_reason"] if best else "no_references"),
        "candidates": candidates[: max(top_k, 0)],
    }


def main() -> None:
    args = parse_args()
    result = localize_query(
        Path(args.query).resolve(),
        Path(args.references).resolve(),
        ratio_threshold=args.ratio_threshold,
        min_matches=args.min_matches,
        min_inliers=args.min_inliers,
        ransac_threshold=args.ransac_threshold,
        top_k=args.top_k,
    )
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
