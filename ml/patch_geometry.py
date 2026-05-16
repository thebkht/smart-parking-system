"""Shared parking-spot geometry helpers for extraction and runtime inference."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def order_corners(corners: Any) -> np.ndarray:
    try:
        pts = np.asarray(corners, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("corners must be numeric [x, y] pairs") from exc
    if pts.shape != (4, 2):
        raise ValueError("corners must contain exactly four [x, y] points")

    unique = np.unique(pts, axis=0)
    if len(unique) != 4:
        raise ValueError("corners must not contain duplicate points")

    hull = cv2.convexHull(pts).reshape(-1, 2)
    if hull.shape != (4, 2):
        raise ValueError("corners must form a convex quadrilateral")

    center = hull.mean(axis=0)
    angles = np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0])
    ordered = hull[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0).astype(np.float32)

    if _has_self_intersection(ordered):
        raise ValueError("corners produce a self-crossing quadrilateral")
    area = cv2.contourArea(ordered)
    if not np.isfinite(area) or area <= 1.0:
        raise ValueError("corners must span a visible quadrilateral area")
    return ordered


def _has_self_intersection(points: np.ndarray) -> bool:
    return _segments_intersect(points[0], points[1], points[2], points[3]) or _segments_intersect(
        points[1], points[2], points[3], points[0]
    )


def _segments_intersect(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> bool:
    def orient(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        return float((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

    o1 = orient(a, b, c)
    o2 = orient(a, b, d)
    o3 = orient(c, d, a)
    o4 = orient(c, d, b)
    return (o1 * o2 < 0) and (o3 * o4 < 0)


def warp_patch(image: np.ndarray, corners: Any, size: int = 128) -> np.ndarray:
    ordered = order_corners(corners)
    dst = np.array([[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]], dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    return cv2.warpPerspective(image, matrix, (size, size))


def square_patch(image: np.ndarray, corners: Any, size: int = 128) -> np.ndarray:
    ordered = order_corners(corners)
    xs = ordered[:, 0]
    ys = ordered[:, 1]
    side = max(float(xs.max() - xs.min()), float(ys.max() - ys.min()))
    if not np.isfinite(side) or side <= 1.0:
        raise ValueError("corners must span a visible quadrilateral area")
    cx = float(xs.min() + xs.max()) / 2.0
    cy = float(ys.min() + ys.max()) / 2.0
    half_side = side / 2.0

    height, width = image.shape[:2]
    x1 = max(0, int(np.floor(cx - half_side)))
    y1 = max(0, int(np.floor(cy - half_side)))
    x2 = min(width, int(np.ceil(cx + half_side)))
    y2 = min(height, int(np.ceil(cy + half_side)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bounding square collapsed outside the image")

    patch = image[y1:y2, x1:x2]
    if patch.size == 0:
        raise ValueError("bounding square produced an empty patch")
    return cv2.resize(patch, (size, size), interpolation=cv2.INTER_LINEAR)


def box_to_corners(box: Any) -> np.ndarray:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError("box must be [x1, y1, x2, y2]")
    x1, y1, x2, y2 = (float(v) for v in box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box must have positive width and height")
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def spot_geometry_from_box(box: Any) -> np.ndarray:
    return order_corners(box_to_corners(box))
