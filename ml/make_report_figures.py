#!/usr/bin/env python3
"""Render report-ready figures for the promoted Stage 2 classifier.

Runs the promoted ``yolov8n-cls`` checkpoint over the ACPDS test split and
saves two figures under ``artifacts/figures/``:

  * ``confusion_matrix.png`` -- raw counts + row-normalized, at threshold 0.5
  * ``pr_curve.png``         -- precision-recall curve with the 0.5 operating
                                point marked

Both are derived from the same per-sample occupied-probability scores, so the
confusion matrix and PR curve are guaranteed consistent with each other.

Usage:
    python ml/make_report_figures.py
    python ml/make_report_figures.py --weights acpds_cls/weights/best.pt \
        --data datasets/acpds_stage2 --split test --device mps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
)

from evaluate import (  # noqa: E402  (import after matplotlib backend is set)
    CLASS_NAMES,
    classify_probabilities,
    eval_split_dir,
    stage2_probabilities,
    stage2_root,
)
from report_style import (  # noqa: E402
    CYAN,
    INK,
    OCC_RED,
    PALETTE_BLUES,
    apply_style,
)

DEFAULT_WEIGHTS = "acpds_cls/weights/best.pt"
DEFAULT_OUTPUT_DIR = "artifacts/figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS)
    parser.add_argument("--data", default=None, help="Stage 2 dataset root.")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--normalized-out",
        default=None,
        help="If set, also save a single-panel normalized confusion matrix here.",
    )
    return parser.parse_args()


def plot_confusion_matrix(matrix: np.ndarray, split: str, out_path: Path) -> None:
    normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    titles = ("Raw counts", "Row-normalized")
    data = (matrix, normalized)
    fmts = ("d", ".3f")
    for ax, title, grid, fmt in zip(axes, titles, data, fmts):
        im = ax.imshow(grid, cmap=PALETTE_BLUES, vmin=0, vmax=grid.max())
        ax.set_title(title)
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_yticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES])
        ax.set_yticklabels([c.capitalize() for c in CLASS_NAMES])
        ax.set_xlabel("True label")
        ax.set_ylabel("Predicted label")
        threshold = grid.max() / 2.0
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(
                    j,
                    i,
                    format(grid[i, j], fmt),
                    ha="center",
                    va="center",
                    fontsize=13,
                    fontweight="bold",
                    color="white" if grid[i, j] > threshold else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"YOLOv8n-cls — Confusion Matrix (ACPDS {split} split, "
        f"{int(matrix.sum())} patches)",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_normalized(matrix: np.ndarray, split: str, out_path: Path) -> None:
    """Single-panel row-normalized confusion matrix in the report palette."""
    normalized = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(normalized, cmap=PALETTE_BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels([c.capitalize() for c in CLASS_NAMES])
    ax.set_yticklabels([c.capitalize() for c in CLASS_NAMES])
    ax.set_xlabel("True label")
    ax.set_ylabel("Predicted label")
    ax.set_title(
        f"YOLOv8n-cls — Normalized Confusion ({split} split)", fontweight="bold"
    )
    ax.grid(False)
    for i in range(normalized.shape[0]):
        for j in range(normalized.shape[1]):
            ax.text(
                j,
                i,
                format(normalized[i, j], ".3f"),
                ha="center",
                va="center",
                fontsize=15,
                fontweight="bold",
                color="white" if normalized[i, j] > 0.5 else INK,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_pr_curve(
    y_true_occ: np.ndarray,
    scores: np.ndarray,
    operating_threshold: float,
    out_path: Path,
) -> None:
    precision, recall, thresholds = precision_recall_curve(y_true_occ, scores)
    ap = average_precision_score(y_true_occ, scores)
    pr_auc = auc(recall, precision)

    # Operating point: precision/recall at the deployed threshold.
    op_pred = scores >= operating_threshold
    tp = int(np.sum(op_pred & (y_true_occ == 1)))
    fp = int(np.sum(op_pred & (y_true_occ == 0)))
    fn = int(np.sum(~op_pred & (y_true_occ == 1)))
    op_precision = tp / (tp + fp) if (tp + fp) else 0.0
    op_recall = tp / (tp + fn) if (tp + fn) else 0.0

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot(recall, precision, color=CYAN, lw=2, label=f"YOLOv8n-cls (AP={ap:.4f})")
    ax.scatter(
        [op_recall],
        [op_precision],
        color=OCC_RED,
        zorder=5,
        s=60,
        label=f"Operating point (t={operating_threshold:g})",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve — YOLOv8n-cls", fontweight="bold")
    ax.set_xlim(0.0, 1.01)
    ax.set_ylim(min(0.4, precision.min() - 0.05), 1.01)
    ax.grid(True, ls=":", alpha=0.5)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return pr_auc, ap


def main() -> None:
    args = parse_args()
    apply_style()
    dataset_dir = eval_split_dir(stage2_root(args.data), args.split)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scoring {args.weights} on {dataset_dir} ...")
    probabilities = stage2_probabilities(
        args.weights,
        dataset_dir,
        device=args.device,
        imgsz=args.imgsz,
        batch=args.batch,
    )

    metrics = classify_probabilities(probabilities, threshold=args.threshold)
    matrix = np.array(metrics["confusion_matrix"], dtype=int)

    y_true_occ = np.array(
        [1 if expected == "occupied" else 0 for expected, _ in probabilities]
    )
    scores = np.array([occ for _, occ in probabilities], dtype=float)

    cm_path = output_dir / "confusion_matrix.png"
    pr_path = output_dir / "pr_curve.png"
    plot_confusion_matrix(matrix, args.split, cm_path)
    plot_pr_curve(y_true_occ, scores, args.threshold, pr_path)
    if args.normalized_out:
        norm_path = Path(args.normalized_out)
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        plot_confusion_normalized(matrix, args.split, norm_path)
        print(f"Saved {norm_path}")

    print(
        f"Accuracy {metrics['top1_accuracy']:.4f} | "
        f"precision {metrics['precision']:.4f} | "
        f"recall {metrics['recall']:.4f} | f1 {metrics['f1']:.4f}"
    )
    print(f"Confusion matrix (rows=pred, cols=true) {CLASS_NAMES}: {matrix.tolist()}")
    print(f"Saved {cm_path}")
    print(f"Saved {pr_path}")


if __name__ == "__main__":
    main()
