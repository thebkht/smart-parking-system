#!/usr/bin/env python3
"""Compare ACPDS Stage 2 quad vs square pooling evaluation results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Week 7 pooling comparison summary JSON.")
    parser.add_argument("--quad-json", default="logs/week7/pooling_quad_test.json")
    parser.add_argument("--square-json", default="logs/week7/pooling_square_test.json")
    parser.add_argument("--output", default="logs/week7/pooling_comparison.json")
    return parser.parse_args()


def load_row(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, dict) else []
    if len(rows) != 1:
        raise SystemExit(f"Expected exactly one evaluation row in {path}")
    row = rows[0]
    if not isinstance(row, dict):
        raise SystemExit(f"Invalid evaluation row in {path}")
    return row


def metric_delta(square: dict[str, Any], quad: dict[str, Any], key: str) -> float:
    return round(float(square[key]) - float(quad[key]), 4)


def main() -> None:
    args = parse_args()
    quad_path = Path(args.quad_json)
    square_path = Path(args.square_json)
    quad = load_row(quad_path)
    square = load_row(square_path)

    payload = {
        "mode": "stage2_pooling_comparison",
        "quad_source": str(quad_path),
        "square_source": str(square_path),
        "quad": {
            "pooling": "quad",
            **quad,
        },
        "square": {
            "pooling": "square",
            **square,
        },
        "delta_square_minus_quad": {
            "top1_accuracy": metric_delta(square, quad, "top1_accuracy"),
            "precision": metric_delta(square, quad, "precision"),
            "recall": metric_delta(square, quad, "recall"),
            "f1": metric_delta(square, quad, "f1"),
        },
        "summary": {
            "better_pooling_by_accuracy": "quad" if float(quad["top1_accuracy"]) >= float(square["top1_accuracy"]) else "square",
            "quad_accuracy": float(quad["top1_accuracy"]),
            "square_accuracy": float(square["top1_accuracy"]),
            "accuracy_gap_pp": round((float(square["top1_accuracy"]) - float(quad["top1_accuracy"])) * 100.0, 2),
        },
        "narrative": [
            "Quadrilateral pooling is the production path because it preserves the train/serve geometry contract.",
            "Square pooling acts as the coarse ACPDS Table 2 baseline by resizing a bounding square instead of perspective-correcting the spot polygon.",
            "The accuracy delta quantifies the cost of replacing perspective-corrected pooling with a simpler square crop on the same YOLOv8n Stage 2 architecture.",
        ],
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
