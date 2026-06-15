#!/usr/bin/env python3
"""Bandwidth analysis: JSON payload vs equivalent H.264 video stream.

Compares the bandwidth used by the smart parking system's JSON POST approach
against a conservative estimate for a continuous H.264 parking camera feed.

Usage:
  python ml/bandwidth.py                          # use default assumptions
  python ml/bandwidth.py --posts-per-sec 0.5      # 1 POST every 2 seconds
  python ml/bandwidth.py --h264-kbps 4000         # higher quality H.264
  python ml/bandwidth.py --log logs/parking_log_2026-04-10.csv
"""

import argparse
import json
from pathlib import Path

DEFAULT_H264_KBPS = 2000  # kbps for 1080p H.264 parking camera
DEFAULT_POSTS_PER_SEC = 0.5  # 1 POST every 2 seconds (matches DEFAULT_POST_INTERVAL_S)
DEFAULT_LOG_DIR = "logs"
OUTPUT_FILE = "logs/bandwidth_analysis.txt"

# JSON contract shape (worst-case: 6 spots)
SAMPLE_PAYLOAD = {
    "spots": {
        "spot_1": "occupied",
        "spot_2": "free",
        "spot_3": "occupied",
        "spot_4": "free",
        "spot_5": "occupied",
        "spot_6": "free",
    },
    "confidence": {
        "spot_1": 0.95,
        "spot_2": 0.17,
        "spot_3": 0.88,
        "spot_4": 0.21,
        "spot_5": 0.93,
        "spot_6": 0.16,
    },
    "timestamp": "2026-04-21T00:00:00Z",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bandwidth comparison: JSON vs H.264.")
    p.add_argument(
        "--posts-per-sec",
        type=float,
        default=DEFAULT_POSTS_PER_SEC,
        help="How often the edge device POSTs a result (per second).",
    )
    p.add_argument(
        "--h264-kbps",
        type=int,
        default=DEFAULT_H264_KBPS,
        help="Assumed H.264 stream bitrate in kbps.",
    )
    p.add_argument(
        "--log",
        default=None,
        help="Path to a JSONL or CSV inference log to measure real payload sizes.",
    )
    p.add_argument(
        "--output",
        default=OUTPUT_FILE,
        help="Path to write the analysis text report.",
    )
    return p.parse_args()


def measure_real_payloads(log_path: Path) -> list[int]:
    """Read JSONL or CSV logs and estimate per-row JSON payload sizes in bytes."""
    sizes: list[int] = []
    if log_path.suffix.lower() == ".json":
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                sizes.append(len(line.encode("utf-8")))
        return sizes

    import csv

    with open(log_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            payload = {
                "spots": {k: v for k, v in row.items() if k != "timestamp"},
                "timestamp": row.get("timestamp"),
            }
            sizes.append(len(json.dumps(payload).encode()))
    return sizes


def format_bytes(n: float) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f} MB"
    if n >= 1_000:
        return f"{n / 1_000:.2f} KB"
    return f"{n:.0f} B"


def main() -> None:
    args = parse_args()

    # --- JSON side ---
    sample_bytes = len(json.dumps(SAMPLE_PAYLOAD).encode())

    if args.log:
        log_path = Path(args.log)
        if log_path.exists():
            sizes = measure_real_payloads(log_path)
            avg_bytes = sum(sizes) / len(sizes) if sizes else sample_bytes
            print(
                f"Real log: {len(sizes)} records, avg payload = {avg_bytes:.0f} bytes"
            )
        else:
            print(f"Log not found ({log_path}), using sample payload size.")
            avg_bytes = sample_bytes
    else:
        avg_bytes = sample_bytes

    json_bytes_per_sec = avg_bytes * args.posts_per_sec

    # HTTP overhead (headers ~300 bytes per POST)
    http_overhead_per_sec = 300 * args.posts_per_sec
    total_json_bps = json_bytes_per_sec + http_overhead_per_sec
    total_json_bph = total_json_bps * 3600

    # --- H.264 side ---
    h264_bytes_per_sec = args.h264_kbps * 1000 / 8
    h264_bytes_per_hour = h264_bytes_per_sec * 3600

    # --- Savings ---
    savings_pct = (1 - total_json_bps / h264_bytes_per_sec) * 100

    lines = [
        "=" * 60,
        "Smart Parking System — Bandwidth Analysis",
        "=" * 60,
        "",
        "Assumptions:",
        f"  JSON posts per second : {args.posts_per_sec}",
        f"  JSON payload size     : {avg_bytes:.0f} bytes (sample: {sample_bytes} bytes)",
        "  HTTP overhead         : ~300 bytes per POST",
        f"  H.264 stream          : {args.h264_kbps} kbps (1080p parking camera)",
        "",
        "JSON POST bandwidth:",
        f"  Per POST              : {avg_bytes:.0f} bytes payload + ~300 bytes headers",
        f"  Per second            : {format_bytes(total_json_bps)} / sec",
        f"  Per hour              : {format_bytes(total_json_bph)} / hr",
        "",
        "H.264 video stream bandwidth:",
        f"  Per second            : {format_bytes(h264_bytes_per_sec)} / sec",
        f"  Per hour              : {format_bytes(h264_bytes_per_hour)} / hr",
        "",
        "Comparison:",
        f"  Bandwidth savings     : {savings_pct:.1f}%",
        f"  JSON uses             : {total_json_bps / h264_bytes_per_sec * 100:.4f}% of H.264 bandwidth",
        f"  Reduction factor      : {h264_bytes_per_sec / total_json_bps:.0f}x less data",
        "",
        "=" * 60,
    ]

    report = "\n".join(lines)
    print(report)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report + "\n")
    print(f"\nReport saved to {out}")


if __name__ == "__main__":
    main()
