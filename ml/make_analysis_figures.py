#!/usr/bin/env python3
"""Regenerate the report's analysis graphs with the shared architecture palette.

Reads already-saved evaluation data (no model inference needed) and writes the
bar charts and training-curve figures used in the technical report directly
into ``outputs/figures/``:

  * w11_per_weather.png      -- per-weather accuracy/precision/recall (test)
  * w11_val_test_gap.png     -- val vs test accuracy for the n/s/m variants
  * w11_pooling_compare.png  -- quadrilateral vs square pooling (test)
  * yolov8n_results.png      -- YOLOv8n-cls training curves
  * w11_yolov8s_curves.png   -- YOLOv8s-cls training curves
  * w11_yolov8m_curves.png   -- YOLOv8m-cls training curves

Usage:
    python ml/make_analysis_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from report_style import (
    CYAN,
    FREE_GREEN,
    MAGENTA,
    OCC_RED,
    PURPLE,
    SERIES,
    apply_style,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS = REPO_ROOT / "logs"
RUNS = REPO_ROOT / "runs" / "acpds_cls"
OUT = REPO_ROOT / "outputs" / "figures"


def _bar_labels(ax, bars, fmt="{:.4f}") -> None:
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=90,
        )


def grouped_bars(groups, series, title, out_path, ylabel="Score", ymin=None):
    """series: list of (label, values, color). values aligned with groups."""
    n_groups = len(groups)
    n_series = len(series)
    x = np.arange(n_groups)
    width = 0.8 / n_series
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    all_vals = []
    for i, (label, values, color) in enumerate(series):
        offset = (i - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color,
                      edgecolor="white", linewidth=0.5)
        _bar_labels(ax, bars)
        all_vals.extend(values)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    lo = min(all_vals)
    if ymin is None:
        ymin = max(0.0, lo - 0.04)
    ax.set_ylim(ymin, 1.005)
    ax.legend(loc="upper right", ncol=min(n_series, 3), fontsize=9)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


# --- Per-weather ------------------------------------------------------------
def per_weather() -> None:
    rows = {}
    with open(LOGS / "stage2_per_weather.csv") as fh:
        for r in csv.DictReader(fh):
            if r["model"] == "acpds_cls":
                rows[r["dataset"]] = r
    order = [("sunny", "Sunny"), ("overcast", "Overcast"), ("low_light", "Low-light")]
    labels = [f"{disp}\n(n={rows[key]['sample_count']})" for key, disp in order]
    acc = [float(rows[k]["top1_accuracy"]) for k, _ in order]
    prec = [float(rows[k]["precision"]) for k, _ in order]
    rec = [float(rows[k]["recall"]) for k, _ in order]
    grouped_bars(
        labels,
        [("Accuracy", acc, CYAN), ("Precision", prec, PURPLE), ("Recall", rec, MAGENTA)],
        "YOLOv8n-cls — Per-Weather Accuracy (Test Split)",
        OUT / "w11_per_weather.png",
        ymin=0.90,
    )


# --- Val vs test gap --------------------------------------------------------
def val_test_gap() -> None:
    data = json.load(open(LOGS / "week7" / "val_test_gap.json"))
    variants = {v["variant"]: v for v in data["variants"]}
    order = [("n", "YOLOv8n"), ("s", "YOLOv8s"), ("m", "YOLOv8m")]
    labels = [disp for _, disp in order]
    val = [variants[k]["val"]["top1_accuracy"] for k, _ in order]
    test = [variants[k]["test"]["top1_accuracy"] for k, _ in order]
    grouped_bars(
        labels,
        [("Val accuracy", val, CYAN), ("Test accuracy", test, PURPLE)],
        "Val vs Test Accuracy — All Variants",
        OUT / "w11_val_test_gap.png",
        ymin=0.96,
    )


# --- Pooling comparison -----------------------------------------------------
def pooling_compare() -> None:
    data = json.load(open(LOGS / "week7" / "pooling_comparison.json"))
    quad, square = data["quad"], data["square"]
    metrics = [("top1_accuracy", "Accuracy"), ("precision", "Precision"),
               ("recall", "Recall"), ("f1", "F1")]
    labels = [disp for _, disp in metrics]
    q = [quad[k] for k, _ in metrics]
    s = [square[k] for k, _ in metrics]
    grouped_bars(
        labels,
        [("Quadrilateral pooling (method a)", q, CYAN),
         ("Square pooling (method b)", s, PURPLE)],
        "Pooling Method Comparison — YOLOv8n-cls Test Split",
        OUT / "w11_pooling_compare.png",
        ymin=0.92,
    )


# --- Training curves --------------------------------------------------------
def _read_results(csv_path: Path):
    epochs, tr_loss, val_loss, acc = [], [], [], []
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            epochs.append(float(r["epoch"]))
            tr_loss.append(float(r["train/loss"]))
            val_loss.append(float(r["val/loss"]))
            acc.append(float(r["metrics/accuracy_top1"]))
    return (np.array(epochs), np.array(tr_loss), np.array(val_loss), np.array(acc))


def training_curves(run_name: str, title: str, out_path: Path) -> None:
    epochs, tr_loss, val_loss, acc = _read_results(RUNS / run_name / "results.csv")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.5))
    ax1.plot(epochs, tr_loss, color=CYAN, lw=2, label="train/loss")
    ax1.plot(epochs, val_loss, color=OCC_RED, lw=2, label="val/loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Loss")
    ax1.legend(loc="upper right", fontsize=9)
    ax2.plot(epochs, acc, color=FREE_GREEN, lw=2, label="val top-1 acc")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Top-1 accuracy")
    ax2.set_title("Validation accuracy")
    ax2.legend(loc="lower right", fontsize=9)
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)
    per_weather()
    val_test_gap()
    pooling_compare()
    training_curves("yolov8n_stage2", "YOLOv8n-cls — Training Curves",
                    OUT / "yolov8n_results.png")
    training_curves("yolov8s_stage2", "YOLOv8s-cls — Training Curves (32 epochs)",
                    OUT / "w11_yolov8s_curves.png")
    training_curves("yolov8m_stage2", "YOLOv8m-cls — Training Curves (26 epochs)",
                    OUT / "w11_yolov8m_curves.png")


if __name__ == "__main__":
    main()
