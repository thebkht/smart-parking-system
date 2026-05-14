#!/usr/bin/env python3
"""Generate final-project artifact summaries and acceptance checks."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


EXPECTED_STAGE2_MODELS = ("n", "s", "m")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize smart parking final artifacts.")
    parser.add_argument("--stage1-report", default="datasets/stage1_data/detection_dataset_report.json")
    parser.add_argument("--stage2-dir", default="datasets/stage2_data")
    parser.add_argument("--pklot-test", default="datasets/pklot_test")
    parser.add_argument("--cnrpark-test", default="datasets/cnrpark_test")
    parser.add_argument("--logs-dir", default="logs")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--output-json", default="artifacts/final_manifest.json")
    parser.add_argument("--output-md", default="docs/final-artifact-summary.md")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, skipinitialspace=True))

    normalized_rows: list[dict[str, Any]] = []
    for row in rows:
        normalized: dict[str, Any] = {}
        for key, value in row.items():
            if key is None:
                continue
            clean_key = key.strip()
            if isinstance(value, str):
                normalized[clean_key] = value.strip()
            else:
                normalized[clean_key] = value
        if normalized:
            normalized_rows.append(normalized)
    return normalized_rows


def latest_csv_row(path: Path) -> dict[str, Any] | None:
    rows = csv_rows(path)
    return rows[-1] if rows else None


def best_csv_row(path: Path, score_key: str) -> dict[str, Any] | None:
    rows = csv_rows(path)
    if not rows:
        return None

    best = None
    best_score = float("-inf")
    for row in rows:
        try:
            score = float(row.get(score_key, "nan"))
        except (TypeError, ValueError):
            continue
        if score >= best_score:
            best = row
            best_score = score
    return best or rows[-1]


def count_images(path: Path) -> int:
    count = 0
    for suffix in ("*.jpg", "*.jpeg", "*.png"):
        count += len(list(path.glob(suffix)))
    return count


def stage2_inventory(stage2_dir: Path) -> dict[str, Any]:
    inventory: dict[str, Any] = {"present": stage2_dir.exists(), "splits": {}}
    if not stage2_dir.exists():
        return inventory

    for split in ("train", "val", "test"):
        split_dir = stage2_dir / split
        if not split_dir.exists():
            continue
        inventory["splits"][split] = {
            "free": count_images(split_dir / "free"),
            "occupied": count_images(split_dir / "occupied"),
        }
    return inventory


def flat_test_inventory(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"present": False}
    return {
        "present": True,
        "free": count_images(path / "free"),
        "occupied": count_images(path / "occupied"),
    }


def checkpoint_entry(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": path.exists(),
        "size_mb": round(path.stat().st_size / 1_048_576, 2) if path.exists() else None,
    }


def first_existing_checkpoint(paths: Iterable[Path]) -> dict[str, Any]:
    candidates = list(paths)
    for path in candidates:
        if path.exists():
            return checkpoint_entry(path)
    return checkpoint_entry(candidates[0])


def collect_checkpoints(runs_dir: Path, artifacts_dir: Path) -> dict[str, Any]:
    stage1_s = runs_dir / "stage1_det" / "yolov8s_stage1" / "weights" / "best.pt"
    stage1_n = runs_dir / "stage1_det" / "yolov8n_stage1" / "weights" / "best.pt"
    stage1_m = runs_dir / "stage1_det" / "yolov8m_stage1" / "weights" / "best.pt"
    stage2 = {
        variant: first_existing_checkpoint(
            [
                runs_dir / "acpds_cls" / f"yolov8{variant}_stage2" / "weights" / "best.pt",
                runs_dir / "stage2_cls" / f"yolov8{variant}_stage2" / "weights" / "best.pt",
            ]
        )
        for variant in EXPECTED_STAGE2_MODELS
    }
    exports = {
        "best_pt": checkpoint_entry(artifacts_dir / "models" / "best.pt"),
        "best_onnx": checkpoint_entry(artifacts_dir / "models" / "best.onnx"),
        "best_int8_onnx": checkpoint_entry(artifacts_dir / "models" / "best_int8.onnx"),
    }
    return {
        "stage1_s": checkpoint_entry(stage1_s),
        "stage1_m": checkpoint_entry(stage1_m),
        "stage2": stage2,
        "exports": exports,
    }


def required_artifact_checks(manifest: dict[str, Any]) -> dict[str, bool]:
    checkpoints = manifest["checkpoints"]
    stage2_logs = manifest["metrics"]
    datasets = manifest["datasets"]
    return {
        "stage1_detector_checkpoint": checkpoints["stage1_s"]["present"] or checkpoints["stage1_m"]["present"],
        "stage2_n_checkpoint": checkpoints["stage2"]["n"]["present"],
        "stage2_s_checkpoint": checkpoints["stage2"]["s"]["present"],
        "stage2_m_checkpoint": checkpoints["stage2"]["m"]["present"],
        "stage1_eval_table": stage2_logs["stage1_evaluation"] is not None,
        "stage2_eval_table": stage2_logs["stage2_evaluation"] is not None,
        "stage2_model_comparison": stage2_logs["stage2_model_comparison"] is not None,
        "threshold_sweep": stage2_logs["stage2_threshold_sweep"] is not None,
        "cross_dataset_eval": (
            stage2_logs["stage2_cross_dataset"] is not None
            if datasets["cnrpark_test"]["present"] or datasets["pklot_test"]["present"]
            else True
        ),
        "per_weather_eval": (
            stage2_logs["stage2_per_weather"] is not None
            if datasets.get("stage2_weather", {}).get("present")
            else True
        ),
        "benchmark_results": stage2_logs["benchmark_results"] is not None,
        "bandwidth_report": stage2_logs["bandwidth_report_present"],
        "stability_summary": stage2_logs["stability_summary"] is not None,
    }


def load_metrics(logs_dir: Path) -> dict[str, Any]:
    return {
        "stage1_evaluation": latest_csv_row(logs_dir / "stage1_evaluation.csv"),
        "stage2_evaluation": latest_csv_row(logs_dir / "stage2_evaluation.csv"),
        "stage2_model_comparison": best_csv_row(logs_dir / "stage2_model_comparison.csv", "f1"),
        "stage2_threshold_sweep": best_csv_row(logs_dir / "stage2_threshold_sweep.csv", "f1"),
        "stage2_cross_dataset": latest_csv_row(logs_dir / "stage2_cross_dataset.csv"),
        "stage2_per_weather": latest_csv_row(logs_dir / "stage2_per_weather.csv"),
        "benchmark_results": read_json(logs_dir / "benchmark_results.json"),
        "stability_summary": read_json(logs_dir / "stability_summary.json"),
        "bandwidth_report_present": (logs_dir / "bandwidth_analysis.txt").exists(),
    }


def stage1_inventory(report: dict[str, Any] | None) -> dict[str, Any]:
    if not report:
        return {"present": False}
    splits = {
        name: {
            "images_kept": values.get("images_kept", 0),
            "boxes_kept": values.get("boxes_kept", 0),
            "scene_count": values.get("scene_count", 0),
        }
        for name, values in report.get("splits", {}).items()
    }
    return {
        "present": True,
        "track": report.get("track"),
        "images_kept_total": report.get("images_kept_total"),
        "boxes_kept_total": report.get("boxes_kept_total"),
        "duplicates_removed": report.get("duplicates_removed"),
        "empty_label_frames_excluded": report.get("empty_label_frames_excluded"),
        "polygon_labels_converted": report.get("polygon_labels_converted"),
        "scene_leakage_detected": report.get("leakage_checks", {}).get("scene_leakage_detected"),
        "splits": splits,
    }


def weather_inventory(path: Path) -> dict[str, Any]:
    inventory: dict[str, Any] = {"present": path.exists(), "splits": {}}
    if not path.exists():
        return inventory
    for weather in ("sunny", "cloudy", "rainy"):
        weather_dir = path / weather
        if not weather_dir.exists():
            continue
        inventory["splits"][weather] = {
            "free": count_images(weather_dir / "free"),
            "occupied": count_images(weather_dir / "occupied"),
        }
    return inventory


def lines_from_stage2_splits(splits: dict[str, Any]) -> Iterable[str]:
    for split, counts in splits.items():
        yield f"| {split} | {counts.get('free', 0)} | {counts.get('occupied', 0)} |"


def lines_from_stage1_splits(splits: dict[str, Any]) -> Iterable[str]:
    for split, values in splits.items():
        yield (
            f"| {split} | {values.get('images_kept', 0)} | {values.get('boxes_kept', 0)} | "
            f"{values.get('scene_count', 0)} |"
        )


def write_markdown(manifest: dict[str, Any], output_path: Path) -> None:
    stage1 = manifest["datasets"]["stage1"]
    stage2 = manifest["datasets"]["stage2"]
    checks = manifest["checks"]
    checkpoints = manifest["checkpoints"]
    metrics = manifest["metrics"]

    lines = [
        "# Final Artifact Summary",
        "",
        "## Dataset Inventory",
        "",
        "### Stage 1",
        "",
        "| Split | Images | Boxes | Scenes |",
        "| --- | ---: | ---: | ---: |",
        *lines_from_stage1_splits(stage1.get("splits", {})),
        "",
        "### Stage 2",
        "",
        "| Split | Free | Occupied |",
        "| --- | ---: | ---: |",
        *lines_from_stage2_splits(stage2.get("splits", {})),
        "",
        "### Cross-Dataset Exports",
        "",
        f"- `pklot_test`: present={manifest['datasets']['pklot_test']['present']}, free={manifest['datasets']['pklot_test'].get('free', 0)}, occupied={manifest['datasets']['pklot_test'].get('occupied', 0)}",
        f"- `cnrpark_test`: present={manifest['datasets']['cnrpark_test']['present']}, free={manifest['datasets']['cnrpark_test'].get('free', 0)}, occupied={manifest['datasets']['cnrpark_test'].get('occupied', 0)}",
        "",
        "### Weather Export",
        "",
        f"- `stage2_weather`: present={manifest['datasets']['stage2_weather']['present']} splits={manifest['datasets']['stage2_weather'].get('splits', {})}",
        "",
        "## Checkpoints",
        "",
        f"- Stage 1 `yolov8s`: present={checkpoints['stage1_s']['present']} path=`{checkpoints['stage1_s']['path']}`",
        f"- Stage 1 `yolov8m`: present={checkpoints['stage1_m']['present']} path=`{checkpoints['stage1_m']['path']}`",
        f"- Stage 2 `yolov8n-cls`: present={checkpoints['stage2']['n']['present']} path=`{checkpoints['stage2']['n']['path']}`",
        f"- Stage 2 `yolov8s-cls`: present={checkpoints['stage2']['s']['present']} path=`{checkpoints['stage2']['s']['path']}`",
        f"- Stage 2 `yolov8m-cls`: present={checkpoints['stage2']['m']['present']} path=`{checkpoints['stage2']['m']['path']}`",
        "",
        "## Acceptance Checks",
        "",
    ]
    for name, passed in checks.items():
        lines.append(f"- {name}: {'PASS' if passed else 'MISSING'}")

    lines.extend(
        [
            "",
            "## Latest Metrics Snapshot",
            "",
            f"- Stage 1 evaluation: `{json.dumps(metrics['stage1_evaluation'], sort_keys=True) if metrics['stage1_evaluation'] else 'missing'}`",
            f"- Stage 2 evaluation: `{json.dumps(metrics['stage2_evaluation'], sort_keys=True) if metrics['stage2_evaluation'] else 'missing'}`",
            f"- Stage 2 model comparison: `{json.dumps(metrics['stage2_model_comparison'], sort_keys=True) if metrics['stage2_model_comparison'] else 'missing'}`",
            f"- Stage 2 threshold sweep: `{json.dumps(metrics['stage2_threshold_sweep'], sort_keys=True) if metrics['stage2_threshold_sweep'] else 'missing'}`",
            f"- Stage 2 cross-dataset: `{json.dumps(metrics['stage2_cross_dataset'], sort_keys=True) if metrics['stage2_cross_dataset'] else 'missing'}`",
            f"- Stage 2 per-weather: `{json.dumps(metrics['stage2_per_weather'], sort_keys=True) if metrics['stage2_per_weather'] else 'missing'}`",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    stage1_report = read_json(Path(args.stage1_report))
    manifest = {
        "datasets": {
            "stage1": stage1_inventory(stage1_report),
            "stage2": stage2_inventory(Path(args.stage2_dir)),
            "stage2_weather": weather_inventory(Path("datasets/stage2_weather")),
            "pklot_test": flat_test_inventory(Path(args.pklot_test)),
            "cnrpark_test": flat_test_inventory(Path(args.cnrpark_test)),
        },
        "checkpoints": collect_checkpoints(Path(args.runs_dir), Path(args.artifacts_dir)),
        "metrics": load_metrics(Path(args.logs_dir)),
    }
    manifest["checks"] = required_artifact_checks(manifest)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    write_markdown(manifest, Path(args.output_md))

    print(json.dumps(manifest["checks"], indent=2))
    print(f"Saved manifest to {output_json}")
    print(f"Saved markdown summary to {args.output_md}")


if __name__ == "__main__":
    main()
