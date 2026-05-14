#!/usr/bin/env python3
"""Evaluate SIFT localization on a small labeled query set."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.localize import localize_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate localization queries against labeled spot references.")
    parser.add_argument("--queries", required=True, help="JSON file describing query images and expected spot ids.")
    parser.add_argument("--references", required=True, help="Reference manifest JSON or labeled reference directory.")
    parser.add_argument("--output-json", default=None, help="Optional JSON output path.")
    parser.add_argument("--output-csv", default=None, help="Optional CSV output path.")
    parser.add_argument("--ratio-threshold", type=float, default=0.75)
    parser.add_argument("--min-matches", type=int, default=8)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--ransac-threshold", type=float, default=5.0)
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def load_queries(path: Path) -> list[dict[str, str]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise SystemExit(f"Expected a JSON array in {path}")

    queries: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise SystemExit(f"Invalid query entry in {path}: {item!r}")
        image = str(item.get("image", "")).strip()
        expected_spot_id = str(item.get("expected_spot_id", "")).strip()
        if not image or not expected_spot_id:
            raise SystemExit(f"Each query entry must include image and expected_spot_id: {item!r}")
        queries.append({"image": image, "expected_spot_id": expected_spot_id})
    return queries


def evaluate_queries(args: argparse.Namespace) -> dict[str, Any]:
    query_path = Path(args.queries).resolve()
    reference_path = Path(args.references).resolve()
    rows: list[dict[str, Any]] = []

    for item in load_queries(query_path):
        result = localize_query(
            (query_path.parent / item["image"]).resolve(),
            reference_path,
            ratio_threshold=args.ratio_threshold,
            min_matches=args.min_matches,
            min_inliers=args.min_inliers,
            ransac_threshold=args.ransac_threshold,
            top_k=args.top_k,
        )
        rows.append(
            {
                "image": item["image"],
                "expected_spot_id": item["expected_spot_id"],
                "predicted_spot_id": result["spot_id"],
                "correct": result["spot_id"] == item["expected_spot_id"],
                "score": result["score"],
                "match_count": result["match_count"],
                "inlier_count": result["inlier_count"],
                "elapsed_ms": result["elapsed_ms"],
                "failure_reason": result["failure_reason"],
            }
        )

    correct = sum(1 for row in rows if row["correct"])
    summary = {
        "query_count": len(rows),
        "correct_count": correct,
        "accuracy": round(correct / len(rows), 4) if rows else 0.0,
        "rows": rows,
    }
    return summary


def write_json(path: str | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_csv(path: str | None, rows: list[dict[str, Any]]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary = evaluate_queries(args)
    print(json.dumps(summary, indent=2))
    write_json(args.output_json, summary)
    write_csv(args.output_csv, summary["rows"])


if __name__ == "__main__":
    main()
