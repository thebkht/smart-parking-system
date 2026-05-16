#!/usr/bin/env python3
"""Evaluate smart parking models with task-specific metrics.

Stage 2 classification is the primary evaluation path and reports:
  top-1 accuracy, precision, recall, F1, confusion matrix, and per-class support.

Stage 1 detection support remains available for optional detector experiments.
Single-model occupancy detection is supported as an ML-only comparison baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from ultralytics import YOLO
import yaml

DEFAULT_STAGE1_DATA = "ml/stage1.yaml"
DEFAULT_STAGE2_DATA = "datasets/acpds_stage2"
DEFAULT_SINGLE_MODEL_DATA = "ml/single_model.yaml"
DEFAULT_LOG_DIR = "logs"
DEFAULT_STAGE1_IMGSZ = 768
DEFAULT_DETECT_IMGSZ = 768
CLASS_NAMES = ("free", "occupied")
WEATHER_NAMES = ("sunny", "cloudy", "rainy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage 1, Stage 2, or single-model smart parking tracks."
    )
    parser.add_argument("--weights", help="Path to model checkpoint (.pt or .onnx).")
    parser.add_argument("--data", default=None, help="Stage 1 YAML or Stage 2 dataset root.")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument(
        "--batch",
        type=int,
        default=64,
        help="Stage 2 evaluation batch size for classifier inference.",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--stage1", action="store_true", help="Evaluate a Stage 1 detector.")
    mode.add_argument("--stage2", action="store_true", help="Evaluate a Stage 2 classifier.")
    mode.add_argument(
        "--single-model",
        action="store_true",
        help="Evaluate the ML-only single-model occupancy detector baseline.",
    )
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument(
        "--compare",
        nargs="+",
        metavar="WEIGHTS",
        help="Compare multiple checkpoints on the selected split for the chosen mode.",
    )
    parser.add_argument(
        "--cross-dataset",
        metavar="DATASET_DIR",
        help="Evaluate a Stage 2 classifier on another dataset root with free/occupied subdirs.",
    )
    parser.add_argument(
        "--per-weather",
        action="store_true",
        help="Evaluate Stage 2 weather subdirs if the dataset root exposes sunny/cloudy/rainy.",
    )
    parser.add_argument(
        "--weather-labels",
        default="sunny,cloudy,rainy",
        help="Comma-separated weather split names used with --per-weather.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Classifier threshold sweep target used for deployed occupied/free decisions.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep Stage 2 classifier occupied threshold on the selected split.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write the evaluation payload as JSON.",
    )
    return parser.parse_args()


def append_csv(rows: list[dict], log_dir: Path, filename: str) -> None:
    if not rows:
        return
    log_dir.mkdir(parents=True, exist_ok=True)
    output = log_dir / filename
    write_header = not output.exists()
    with open(output, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def print_rows(rows: list[dict], title: str) -> None:
    if not rows:
        return
    print(f"\n{title}")
    headers = list(rows[0].keys())
    widths = {header: max(len(header), max(len(str(row.get(header, ""))) for row in rows)) for header in headers}
    template = "  ".join(f"{{:<{widths[header]}}}" for header in headers)
    print(template.format(*headers))
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print(template.format(*[str(row.get(header, "")) for header in headers]))


def write_json(payload: dict[str, Any], path: str | None) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def stage2_root(data: str | None) -> Path:
    root = Path(data or DEFAULT_STAGE2_DATA)
    if not root.exists():
        raise SystemExit(f"Stage 2 dataset root not found: {root}")
    return root


def stage1_data(data: str | None) -> str:
    path = Path(data or DEFAULT_STAGE1_DATA)
    if not path.exists():
        raise SystemExit(f"Stage 1 data YAML not found: {path}")
    return str(path)


def single_model_data(data: str | None) -> str:
    path = Path(data or DEFAULT_SINGLE_MODEL_DATA)
    if not path.exists():
        raise SystemExit(f"Single-model data YAML not found: {path}")
    return str(path)


def evaluation_mode(args: argparse.Namespace) -> str:
    if args.stage1:
        return "stage1"
    if args.single_model:
        return "single_model"
    return "stage2"


def _yaml_dataset_root(yaml_path: str) -> Path:
    data = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    root = data.get("path")
    if not root:
        raise SystemExit(f"Dataset YAML missing path: {yaml_path}")
    return Path(root)


def detection_report_for_yaml(yaml_path: str) -> dict[str, Any] | None:
    report_path = _yaml_dataset_root(yaml_path) / "detection_dataset_report.json"
    if not report_path.exists():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def detection_scene_rows(report: dict[str, Any] | None) -> list[dict[str, object]]:
    if not report:
        return []
    rows = []
    for split in ("train", "val", "test"):
        split_report = report.get("splits", {}).get(split)
        if not split_report:
            continue
        rows.append(
            {
                "split": split,
                "scene_count": split_report.get("scene_count", 0),
                "image_count": split_report.get("images_kept", 0),
                "box_count": split_report.get("boxes_kept", 0),
            }
        )
    return rows


def classification_samples(dataset_dir: Path) -> list[tuple[Path, str]]:
    samples: list[tuple[Path, str]] = []
    for class_name in CLASS_NAMES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            raise SystemExit(f"Expected class directory missing: {class_dir}")
        for suffix in ("*.jpg", "*.jpeg", "*.png"):
            for path in sorted(class_dir.glob(suffix)):
                samples.append((path, class_name))
    if not samples:
        raise SystemExit(f"No classification samples found in {dataset_dir}")
    return samples


def stage2_probabilities(
    weights: str,
    dataset_dir: Path,
    *,
    device: str,
    imgsz: int,
    batch: int,
) -> list[tuple[str, float]]:
    model = YOLO(weights, task="classify")
    probabilities: list[tuple[str, float]] = []
    samples = classification_samples(dataset_dir)
    effective_batch = stage2_inference_batch(weights, batch)

    for start in range(0, len(samples), effective_batch):
        chunk = samples[start:start + effective_batch]
        results = model(
            [str(image_path) for image_path, _expected in chunk],
            device=device,
            imgsz=imgsz,
            verbose=False,
        )
        for (_image_path, expected), result in zip(chunk, results):
            probabilities.append((expected, occupied_probability(result)))
    return probabilities


def stage2_inference_batch(weights: str, requested_batch: int) -> int:
    suffix = Path(weights).suffix.lower()
    if suffix in {".onnx", ".mlpackage"}:
        return 1
    return max(1, requested_batch)


def occupied_probability(result) -> float:
    names = {int(k): v for k, v in result.names.items()}
    occupied_index = next((idx for idx, name in names.items() if name == "occupied"), None)
    if occupied_index is None:
        top1 = int(result.probs.top1)
        top1_conf = float(result.probs.top1conf)
        return top1_conf if names.get(top1) == "occupied" else 0.0
    return float(result.probs.data[occupied_index])


def classify_probabilities(
    probabilities: list[tuple[str, float]],
    *,
    threshold: float,
) -> dict[str, object]:
    y_true: list[str] = []
    y_pred: list[str] = []

    for expected, occ_prob in probabilities:
        predicted = "occupied" if occ_prob >= threshold else "free"
        y_true.append(expected)
        y_pred.append(predicted)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(CLASS_NAMES),
        zero_division=0,
    )
    matrix = confusion_matrix(y_true, y_pred, labels=list(CLASS_NAMES))
    return {
        "top1_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision[1]), 4),
        "recall": round(float(recall[1]), 4),
        "f1": round(float(f1[1]), 4),
        "support": {cls: int(count) for cls, count in zip(CLASS_NAMES, support)},
        "confusion_matrix": matrix.tolist(),
        "sample_count": len(y_true),
    }


def classify_dataset(
    weights: str,
    dataset_dir: Path,
    *,
    device: str,
    imgsz: int,
    threshold: float,
    batch: int,
) -> dict[str, object]:
    probabilities = stage2_probabilities(
        weights,
        dataset_dir,
        device=device,
        imgsz=imgsz,
        batch=batch,
    )
    return classify_probabilities(probabilities, threshold=threshold)


def evaluate_stage1(args: argparse.Namespace) -> None:
    if not args.weights:
        raise SystemExit("--weights is required for Stage 1 evaluation.")
    data_yaml = stage1_data(args.data)
    model = YOLO(args.weights)
    metrics = model.val(
        data=data_yaml,
        split=args.split,
        device=args.device,
        imgsz=max(args.imgsz, DEFAULT_STAGE1_IMGSZ),
        verbose=False,
    ).results_dict
    report = detection_report_for_yaml(data_yaml)
    split_report = report.get("splits", {}).get(args.split, {}) if report else {}
    leakage = report.get("leakage_checks", {}) if report else {}
    row = {
        "model": Path(args.weights).parent.parent.name,
        "split": args.split,
        "mAP50": round(float(metrics.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(metrics.get("metrics/mAP50-95(B)", 0.0)), 4),
        "precision": round(float(metrics.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(metrics.get("metrics/recall(B)", 0.0)), 4),
        "scene_count": split_report.get("scene_count", 0),
        "scene_leakage": leakage.get("scene_leakage_detected", False),
    }
    print_rows([row], "Stage 1 Detection Evaluation")
    append_csv([row], Path(args.log_dir), "stage1_evaluation.csv")
    scene_rows = detection_scene_rows(report)
    if scene_rows:
        print_rows(scene_rows, "Stage 1 Holdout Scene Summary")


def evaluate_single_model(args: argparse.Namespace) -> None:
    if not args.weights:
        raise SystemExit("--weights is required for single-model evaluation.")
    data_yaml = single_model_data(args.data)
    model = YOLO(args.weights)
    metrics = model.val(
        data=data_yaml,
        split=args.split,
        device=args.device,
        imgsz=max(args.imgsz, DEFAULT_DETECT_IMGSZ),
        verbose=False,
    ).results_dict
    report = detection_report_for_yaml(data_yaml)
    split_report = report.get("splits", {}).get(args.split, {}) if report else {}
    leakage = report.get("leakage_checks", {}) if report else {}
    row = {
        "model": Path(args.weights).parent.parent.name,
        "split": args.split,
        "mAP50": round(float(metrics.get("metrics/mAP50(B)", 0.0)), 4),
        "mAP50_95": round(float(metrics.get("metrics/mAP50-95(B)", 0.0)), 4),
        "precision": round(float(metrics.get("metrics/precision(B)", 0.0)), 4),
        "recall": round(float(metrics.get("metrics/recall(B)", 0.0)), 4),
        "scene_count": split_report.get("scene_count", 0),
        "scene_leakage": leakage.get("scene_leakage_detected", False),
    }
    print_rows([row], "Single-Model Detection Evaluation")
    append_csv([row], Path(args.log_dir), "single_model_evaluation.csv")
    scene_rows = detection_scene_rows(report)
    if scene_rows:
        print_rows(scene_rows, "Single-Model Holdout Scene Summary")


def evaluate_stage2(weights: str, dataset_dir: Path, args: argparse.Namespace, label: str) -> dict[str, object]:
    metrics = classify_dataset(
        weights,
        dataset_dir,
        device=args.device,
        imgsz=args.imgsz,
        threshold=args.confidence_threshold,
        batch=args.batch,
    )
    return {
        "model": model_label(weights),
        "dataset": label,
        "threshold": args.confidence_threshold,
        "top1_accuracy": metrics["top1_accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "sample_count": metrics["sample_count"],
        "support_free": metrics["support"]["free"],
        "support_occupied": metrics["support"]["occupied"],
        "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
    }


def model_label(weights: str) -> str:
    path = Path(weights)
    if not path.exists():
        return path.name
    if path.parent.name == "weights" and path.parent.parent.name:
        return path.parent.parent.name
    return path.stem


def eval_split_dir(root: Path, split: str) -> Path:
    split_dir = root / split
    if not split_dir.exists():
        raise SystemExit(f"Split directory not found: {split_dir}")
    return split_dir


def evaluate_compare(args: argparse.Namespace) -> None:
    mode = evaluation_mode(args)
    if mode == "stage2":
        root = stage2_root(args.data)
        dataset_dir = eval_split_dir(root, args.split)
        rows = []
        for weights in args.compare:
            if not Path(weights).exists():
                print(f"Skipping missing checkpoint: {weights}")
                continue
            row = evaluate_stage2(weights, dataset_dir, args, f"{root.name}/{args.split}")
            row["size_mb"] = round(Path(weights).stat().st_size / 1_048_576, 2)
            rows.append(row)
        print_rows(rows, "Stage 2 Model Comparison")
        append_csv(rows, Path(args.log_dir), "stage2_model_comparison.csv")
        write_json({"mode": "stage2_compare", "rows": rows}, args.output_json)
        return

    data_yaml = stage1_data(args.data) if mode == "stage1" else single_model_data(args.data)
    title = "Stage 1 Detection Comparison" if mode == "stage1" else "Single-Model Detection Comparison"
    filename = "stage1_model_comparison.csv" if mode == "stage1" else "single_model_comparison.csv"
    rows = []
    for weights in args.compare:
        if not Path(weights).exists():
            print(f"Skipping missing checkpoint: {weights}")
            continue
        model = YOLO(weights)
        metrics = model.val(
            data=data_yaml,
            split=args.split,
            device=args.device,
            imgsz=max(args.imgsz, DEFAULT_STAGE1_IMGSZ if mode == "stage1" else DEFAULT_DETECT_IMGSZ),
            verbose=False,
        ).results_dict
        rows.append(
            {
                "model": Path(weights).parent.parent.name,
                "split": args.split,
                "size_mb": round(Path(weights).stat().st_size / 1_048_576, 2),
                "mAP50": round(float(metrics.get("metrics/mAP50(B)", 0.0)), 4),
                "mAP50_95": round(float(metrics.get("metrics/mAP50-95(B)", 0.0)), 4),
                "precision": round(float(metrics.get("metrics/precision(B)", 0.0)), 4),
                "recall": round(float(metrics.get("metrics/recall(B)", 0.0)), 4),
            }
        )
    print_rows(rows, title)
    append_csv(rows, Path(args.log_dir), filename)
    write_json({"mode": mode, "rows": rows}, args.output_json)


def evaluate_cross_dataset(args: argparse.Namespace) -> None:
    if not args.weights:
        raise SystemExit("--weights is required for cross-dataset evaluation.")
    dataset_dir = Path(args.cross_dataset)
    if not dataset_dir.exists():
        raise SystemExit(f"Cross-dataset directory not found: {dataset_dir}")
    row = evaluate_stage2(args.weights, dataset_dir, args, dataset_dir.name)
    print_rows([row], "Stage 2 Cross-Dataset Evaluation")
    append_csv([row], Path(args.log_dir), "stage2_cross_dataset.csv")
    write_json({"mode": "stage2_cross_dataset", "rows": [row]}, args.output_json)


def evaluate_per_weather(args: argparse.Namespace) -> None:
    if not args.weights:
        raise SystemExit("--weights is required for per-weather evaluation.")
    root = stage2_root(args.data)
    weather_names = tuple(label.strip() for label in str(args.weather_labels).split(",") if label.strip())
    if not weather_names:
        raise SystemExit("--weather-labels must include at least one split name.")
    rows = []
    for weather in weather_names:
        weather_dir = root / weather
        if not weather_dir.exists():
            raise SystemExit(
                "Per-weather evaluation requested but weather labels are not exposed in "
                f"{root}. Expected <root>/<weather>/<class>/*.jpg for {', '.join(weather_names)}."
            )
        rows.append(evaluate_stage2(args.weights, weather_dir, args, weather))
    print_rows(rows, "Stage 2 Per-Weather Evaluation")
    append_csv(rows, Path(args.log_dir), "stage2_per_weather.csv")
    write_json({"mode": "stage2_per_weather", "rows": rows}, args.output_json)


def evaluate_threshold_sweep(args: argparse.Namespace) -> None:
    if not args.weights:
        raise SystemExit("--weights is required for threshold sweep.")
    root = stage2_root(args.data)
    dataset_dir = eval_split_dir(root, args.split)
    probabilities = stage2_probabilities(
        args.weights,
        dataset_dir,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
    )
    rows = []
    best_row = None
    for step in range(10, 95, 5):
        threshold = round(step / 100, 2)
        metrics = classify_probabilities(probabilities, threshold=threshold)
        row = {
            "model": Path(args.weights).parent.parent.name if Path(args.weights).exists() else Path(args.weights).name,
            "dataset": f"{root.name}/{args.split}",
            "threshold": threshold,
            "top1_accuracy": metrics["top1_accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "sample_count": metrics["sample_count"],
            "support_free": metrics["support"]["free"],
            "support_occupied": metrics["support"]["occupied"],
            "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        }
        rows.append(row)
        if best_row is None or row["f1"] > best_row["f1"]:
            best_row = dict(row)
    print_rows([best_row] if best_row else [], "Best Threshold Sweep Result")
    append_csv(rows, Path(args.log_dir), "stage2_threshold_sweep.csv")
    write_json(
        {"mode": "stage2_threshold_sweep", "rows": rows, "best_row": best_row},
        args.output_json,
    )


def main() -> None:
    args = parse_args()
    if args.compare:
        evaluate_compare(args)
        return

    mode = evaluation_mode(args)
    if args.sweep:
        if mode != "stage2":
            raise SystemExit("--sweep is only supported for Stage 2 classification.")
        evaluate_threshold_sweep(args)
        return
    if args.cross_dataset:
        if mode != "stage2":
            raise SystemExit("--cross-dataset is only supported for Stage 2 classification.")
        evaluate_cross_dataset(args)
        return
    if args.per_weather:
        if mode != "stage2":
            raise SystemExit("--per-weather is only supported for Stage 2 classification.")
        evaluate_per_weather(args)
        return
    if not args.weights:
        raise SystemExit("--weights is required unless using --compare")
    if mode == "stage2":
        root = stage2_root(args.data)
        row = evaluate_stage2(args.weights, eval_split_dir(root, args.split), args, f"{root.name}/{args.split}")
        print_rows([row], "Stage 2 Classification Evaluation")
        append_csv([row], Path(args.log_dir), "stage2_evaluation.csv")
        write_json({"mode": "stage2", "rows": [row]}, args.output_json)
        return
    if mode == "single_model":
        evaluate_single_model(args)
        write_json(
            {"mode": "single_model", "rows": [latest_csv_row(Path(args.log_dir) / "single_model_evaluation.csv")]},
            args.output_json,
        )
        return
    evaluate_stage1(args)
    write_json(
        {"mode": "stage1", "rows": [latest_csv_row(Path(args.log_dir) / "stage1_evaluation.csv")]},
        args.output_json,
    )


def latest_csv_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


if __name__ == "__main__":
    main()
