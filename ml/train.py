#!/usr/bin/env python3
"""Train YOLOv8 models for the smart parking v3 pipeline.

Primary path:
  Stage 2 patch classification with YOLOv8*-cls.

Secondary path:
  Stage 1 spot detection for optional ROI discovery experiments.

ML-only baseline:
  Single-model full-frame occupancy detector with `free` / `occupied` classes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
import shutil
import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.engine import trainer as ultralytics_trainer
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.tal import TaskAlignedAssigner

STAGE1_YAML = "ml/stage1.yaml"
STAGE2_DATA_DIR = "datasets/acpds_stage2"
SINGLE_MODEL_YAML = "ml/single_model.yaml"

STAGE1_EPOCHS = 50
STAGE2_EPOCHS = 50
SINGLE_MODEL_EPOCHS = 50
STAGE1_IMGSZ = 768
STAGE2_IMGSZ = 128
SINGLE_MODEL_IMGSZ = 640
DEFAULT_BATCH = 16
STAGE2_BATCH = 64
STAGE1_LR = 0.003
STAGE2_LR = 0.001
SINGLE_MODEL_LR = 0.002
STAGE1_PATIENCE = 15
STAGE2_PATIENCE = 20
SINGLE_MODEL_PATIENCE = 15
STAGE1_PROJECT = "runs/stage1_det"
STAGE2_PROJECT = "runs/acpds_cls"
SINGLE_MODEL_PROJECT = "runs/single_model_det"
MODEL_DIR = "models"
STAGE2_PROMOTED_CHECKPOINT = "acpds_cls/weights/best.pt"
MIN_ULTRALYTICS = (8, 4, 38)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Stage 1 detector, Stage 2 classifier, or the ML-only single-model baseline."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage1", action="store_true", help="Train Stage 1 detector.")
    group.add_argument("--stage2", action="store_true", help="Train Stage 2 classifier.")
    group.add_argument(
        "--single-model",
        action="store_true",
        help="Train the ML-only single-model occupancy detector baseline.",
    )
    parser.add_argument("--variant", choices=["n", "s", "m"], default="s")
    parser.add_argument("--data", default=None, help="Override data path.")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None, dest="lr0")
    parser.add_argument("--optimizer", default="AdamW", help="Ultralytics optimizer name.")
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-batch-fallback", action="store_true")
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--weights", default=None, help="Override starting weights/checkpoint path.")
    parser.add_argument("--project", default=None, help="Override Ultralytics project/output directory.")
    parser.add_argument("--name", default=None, help="Override Ultralytics run name.")
    parser.add_argument("--freeze", type=int, default=None, help="Freeze the first N model layers during training.")
    parser.add_argument("--amp", dest="amp", action="store_true", default=None, help="Enable AMP.")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow reusing an existing run directory.")
    parser.add_argument("--degrees", type=float, default=0.0)
    parser.add_argument("--fliplr", type=float, default=0.5)
    parser.add_argument("--flipud", type=float, default=0.0)
    parser.add_argument("--scale", type=float, default=0.15)
    parser.add_argument("--erasing", type=float, default=0.2)
    parser.add_argument("--mixup", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--cos-lr", action="store_true")
    return parser.parse_args()


def _check_version() -> None:
    version = tuple(int(part) for part in ultralytics.__version__.split(".")[:3])
    if version < MIN_ULTRALYTICS:
        required = ".".join(str(part) for part in MIN_ULTRALYTICS)
        print(
            f"[warn] ultralytics {ultralytics.__version__} detected; {required}+ recommended.",
            file=sys.stderr,
        )


def _patch_ultralytics_assigner_for_iou_mismatch() -> None:
    """Patch TaskAlignedAssigner.get_box_metrics to avoid MPS boolean-index shape divergence."""
    if getattr(TaskAlignedAssigner, "_smart_parking_iou_patch", False):
        return

    def _get_box_metrics_safe(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        mask_idx = mask_gt.nonzero(as_tuple=True)
        num_pairs = mask_idx[0].numel()
        if num_pairs:
            cls_idx = gt_labels.squeeze(-1).long().clamp_(0, pd_scores.shape[-1] - 1)
            cls_scores = pd_scores.permute(0, 2, 1).gather(1, cls_idx.unsqueeze(-1).expand(-1, -1, na))
            bbox_scores[mask_idx] = cls_scores[mask_idx]
            pd_boxes_all = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)
            gt_boxes_all = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)
            pd_boxes = pd_boxes_all[mask_idx]
            gt_boxes = gt_boxes_all[mask_idx]

            n = min(pd_boxes.shape[0], gt_boxes.shape[0], num_pairs)
            if n != num_pairs:
                print(
                    f"[patch] aligning IoU pair count from {num_pairs} to {n} to avoid tensor mismatch.",
                    file=sys.stderr,
                )
            if n:
                overlaps_vals = self.iou_calculation(gt_boxes[:n], pd_boxes[:n])
                overlaps[mask_idx[0][:n], mask_idx[1][:n], mask_idx[2][:n]] = overlaps_vals

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    TaskAlignedAssigner.get_box_metrics = _get_box_metrics_safe
    TaskAlignedAssigner._smart_parking_iou_patch = True


def _state_dict_is_finite(state_dict: dict[str, object]) -> bool:
    return all(
        torch.isfinite(value).all()
        for value in state_dict.values()
        if isinstance(value, torch.Tensor)
    )


def _checkpoint_paths(project_dir: str, run_name: str) -> tuple[Path, Path]:
    weights_dir = Path(project_dir) / run_name / "weights"
    return weights_dir / "best.pt", weights_dir / "last.pt"


def _existing_checkpoint(best_ckpt: Path, last_ckpt: Path) -> Path | None:
    if best_ckpt.exists():
        return best_ckpt
    if last_ckpt.exists():
        return last_ckpt
    return None


def _sync_run_outputs(actual_dir: Path, expected_dir: Path) -> None:
    if actual_dir == expected_dir or not actual_dir.exists():
        return
    expected_dir.mkdir(parents=True, exist_ok=True)
    for src in actual_dir.iterdir():
        dst = expected_dir / src.name
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)


def _patch_ultralytics_trainer_for_nan_checkpoints() -> None:
    """Patch trainer checkpoint handling so NaN recovery does not cascade into missing-last.pt failures."""
    if getattr(BaseTrainer, "_smart_parking_nan_patch", False):
        return

    def _save_model_safe(self):
        import io
        from datetime import datetime

        ema = deepcopy(ultralytics_trainer.unwrap_model(self.ema.ema)).half()
        used_model_fallback = False
        if not _state_dict_is_finite(ema.state_dict()):
            model_snapshot = deepcopy(ultralytics_trainer.unwrap_model(self.model)).half()
            if not _state_dict_is_finite(model_snapshot.state_dict()):
                ultralytics_trainer.LOGGER.warning(
                    f"Skipping checkpoint save at epoch {self.epoch}: EMA and model contain NaN/Inf"
                )
                return False
            ultralytics_trainer.LOGGER.warning(
                f"EMA contains NaN/Inf at epoch {self.epoch}; saving fallback checkpoint from current model weights."
            )
            ema = model_snapshot
            used_model_fallback = True

        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,
                "ema": ema,
                "updates": self.ema.updates,
                "optimizer": ultralytics_trainer.convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),
                "train_metrics": {
                    **self.metrics,
                    **{"fitness": self.fitness, "ema_fallback_to_model": used_model_fallback},
                },
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": ultralytics_trainer.__version__,
                "git": {
                    "root": str(ultralytics_trainer.GIT.root),
                    "branch": ultralytics_trainer.GIT.branch,
                    "commit": ultralytics_trainer.GIT.commit,
                    "origin": ultralytics_trainer.GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()

        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)
        return True

    def _handle_nan_recovery_safe(self, epoch):
        loss_nan = self.loss is not None and not self.loss.isfinite()
        fitness_nan = self.fitness is not None and not ultralytics_trainer.np.isfinite(self.fitness)
        fitness_collapse = self.best_fitness and self.best_fitness > 0 and self.fitness == 0
        corrupted = ultralytics_trainer.RANK in {-1, 0} and loss_nan and (fitness_nan or fitness_collapse)
        reason = "Loss NaN/Inf" if loss_nan else "Fitness NaN/Inf" if fitness_nan else "Fitness collapse"
        if ultralytics_trainer.RANK != -1:
            broadcast_list = [corrupted if ultralytics_trainer.RANK == 0 else None]
            ultralytics_trainer.dist.broadcast_object_list(broadcast_list, 0)
            corrupted = broadcast_list[0]
        if not corrupted:
            return False
        if not self.last.exists():
            raise RuntimeError(
                f"{reason} detected before a recoverable checkpoint was written at {self.last}. "
                "Training aborted because continuing would only cascade into a missing-checkpoint failure. "
                "Reduce augmentation or learning rate and retry."
            )
        total_attempts = getattr(self, "nan_recovery_total_attempts", 0) + 1
        self.nan_recovery_total_attempts = total_attempts
        self.nan_recovery_attempts = getattr(self, "nan_recovery_attempts", 0) + 1
        if total_attempts > 3:
            raise RuntimeError(
                f"Training failed after {total_attempts} NaN recoveries. "
                f"Best checkpoint so far should remain available at {self.best}. "
                "Retry with a safer optimizer or lower learning rate on MPS."
            )
        ultralytics_trainer.LOGGER.warning(
            f"{reason} detected (recovery {total_attempts}/3), recovering from last.pt..."
        )
        self._model_train()
        _, ckpt = ultralytics_trainer.load_checkpoint(self.last)
        ema_state = ckpt["ema"].float().state_dict()
        if not _state_dict_is_finite(ema_state):
            raise RuntimeError(f"Checkpoint {self.last} is corrupted with NaN/Inf weights")
        ultralytics_trainer.unwrap_model(self.model).load_state_dict(ema_state)
        self._load_checkpoint_state(ckpt)
        del ckpt, ema_state
        self.scheduler.last_epoch = epoch - 1
        return True

    BaseTrainer.save_model = _save_model_safe
    BaseTrainer._handle_nan_recovery = _handle_nan_recovery_safe
    BaseTrainer._smart_parking_nan_patch = True


def _train_with_fallback(model: YOLO, kwargs: dict, args: argparse.Namespace):
    try:
        return model.train(**kwargs)
    except RuntimeError as exc:
        if args.device != "mps" or args.no_batch_fallback or "shape mismatch" not in str(exc):
            raise
        new_batch = max(1, kwargs["batch"] // 2)
        print(f"[MPS fallback] retrying with batch={new_batch}", file=sys.stderr)
        retry_kwargs = {**kwargs, "batch": new_batch, "exist_ok": True}
        return YOLO(model.ckpt_path).train(**retry_kwargs)


def resolve_stage1_data(data: str | None) -> str:
    path = Path(data or STAGE1_YAML)
    if not path.exists():
        raise SystemExit(
            f"Stage 1 YAML not found: {path}\n"
            "Run: python ml/prepare_dataset.py --stage1 --pklot-dir <roboflow-export>"
        )
    return str(path)


def resolve_stage2_data(data: str | None) -> str:
    path = Path(data or STAGE2_DATA_DIR)
    if not path.exists():
        raise SystemExit(
            f"Stage 2 data not found: {path}\n"
            "Run: python ml/extract_patches.py --dataset-root <acpds-root> --output datasets/acpds_stage2 "
            "--run-validation --validation-status passed"
        )
    return str(path)


def resolve_single_model_data(data: str | None) -> str:
    path = Path(data or SINGLE_MODEL_YAML)
    if not path.exists():
        raise SystemExit(
            f"Single-model YAML not found: {path}\n"
            "Run: python ml/prepare_dataset.py --single-model --pklot-dir <roboflow-export>"
        )
    return str(path)


def task_defaults(args: argparse.Namespace) -> dict[str, object]:
    if args.stage2:
        return {
            "task": "classify",
            "track": "stage2",
            "data_path": resolve_stage2_data(args.data),
            "weights": args.weights or f"yolov8{args.variant}-cls.pt",
            "epochs": args.epochs or STAGE2_EPOCHS,
            "imgsz": args.imgsz or STAGE2_IMGSZ,
            "batch": args.batch or STAGE2_BATCH,
            "project_dir": args.project or STAGE2_PROJECT,
            "run_name": args.name or f"yolov8{args.variant}_stage2",
            "report_name": f"stage2_{args.variant}_report.json",
            "lr0": args.lr0 if args.lr0 is not None else STAGE2_LR,
            "patience": args.patience if args.patience is not None else STAGE2_PATIENCE,
            "dropout": args.dropout if args.dropout is not None else 0.1,
            "cos_lr": args.cos_lr or True,
            "promoted_checkpoint": STAGE2_PROMOTED_CHECKPOINT,
        }
    if args.single_model:
        return {
            "task": "detect",
            "track": "single_model",
            "data_path": resolve_single_model_data(args.data),
            "weights": args.weights or f"yolov8{args.variant}.pt",
            "epochs": args.epochs or SINGLE_MODEL_EPOCHS,
            "imgsz": args.imgsz or SINGLE_MODEL_IMGSZ,
            "batch": args.batch or DEFAULT_BATCH,
            "project_dir": args.project or SINGLE_MODEL_PROJECT,
            "run_name": args.name or f"yolov8{args.variant}_single_model",
            "report_name": f"single_model_{args.variant}_report.json",
            "lr0": args.lr0 if args.lr0 is not None else SINGLE_MODEL_LR,
            "patience": args.patience if args.patience is not None else SINGLE_MODEL_PATIENCE,
            "dropout": args.dropout if args.dropout is not None else 0.0,
            "cos_lr": args.cos_lr,
        }
    return {
        "task": "detect",
        "track": "stage1",
        "data_path": resolve_stage1_data(args.data),
        "weights": args.weights or f"yolov8{args.variant}.pt",
        "epochs": args.epochs or STAGE1_EPOCHS,
        "imgsz": args.imgsz or STAGE1_IMGSZ,
        "batch": args.batch or DEFAULT_BATCH,
        "project_dir": args.project or STAGE1_PROJECT,
        "run_name": args.name or f"yolov8{args.variant}_stage1",
        "report_name": f"stage1_{args.variant}_report.json",
        "lr0": args.lr0 if args.lr0 is not None else STAGE1_LR,
        "patience": args.patience if args.patience is not None else STAGE1_PATIENCE,
        "dropout": args.dropout if args.dropout is not None else 0.0,
        "cos_lr": args.cos_lr,
    }


def extract_metrics(task: str, results) -> dict[str, object]:
    metrics = results.results_dict
    if task == "classify":
        top1 = metrics.get("metrics/accuracy_top1", metrics.get("val/acc_top1"))
        top5 = metrics.get("metrics/accuracy_top5", metrics.get("val/acc_top5"))
        return {"top1_accuracy": top1, "top5_accuracy": top5}
    return {
        "mAP50": metrics.get("metrics/mAP50(B)", metrics.get("val/map50")),
        "mAP50_95": metrics.get("metrics/mAP50-95(B)", metrics.get("val/map")),
        "precision": metrics.get("metrics/precision(B)"),
        "recall": metrics.get("metrics/recall(B)"),
    }


def ensure_stage2_validation_passed(data_path: str) -> None:
    report_path = Path(data_path) / "validation_report.json"
    if not report_path.exists():
        raise SystemExit(
            f"Stage 2 validation gate missing: {report_path}\n"
            "Run: python ml/extract_patches.py --dataset-root <acpds-root> --output datasets/acpds_stage2 "
            "--run-validation --validation-status passed"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "passed":
        raise SystemExit(
            f"Stage 2 validation gate not passed: {report_path}\n"
            "Re-run validation after visual review with --validation-status passed."
        )


def promote_stage2_checkpoint(checkpoint_path: Path, destination: str) -> Path:
    if not checkpoint_path.exists():
        raise SystemExit(f"Stage 2 checkpoint not found for promotion: {checkpoint_path}")
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_path, destination_path)
    return destination_path


def main() -> None:
    args = parse_args()
    _check_version()
    _patch_ultralytics_assigner_for_iou_mismatch()
    _patch_ultralytics_trainer_for_nan_checkpoints()
    defaults = task_defaults(args)

    print(f"Task   : {defaults['task']}")
    print(f"Model  : {defaults['weights']}")
    print(f"Data   : {defaults['data_path']}")
    print(f"Device : {args.device}")
    print(
        f"Epochs : {defaults['epochs']}  imgsz={defaults['imgsz']}  "
        f"batch={defaults['batch']}"
    )
    print(f"Track  : {defaults['track']}  lr0={defaults['lr0']}  patience={defaults['patience']}")
    if defaults["track"] == "stage2":
        ensure_stage2_validation_passed(str(defaults["data_path"]))

    model = YOLO(defaults["weights"])
    train_kwargs = {
        "data": defaults["data_path"],
        "epochs": defaults["epochs"],
        "imgsz": defaults["imgsz"],
        "batch": defaults["batch"],
        "optimizer": args.optimizer,
        "lr0": defaults["lr0"],
        "patience": defaults["patience"],
        "device": args.device,
        "project": defaults["project_dir"],
        "name": defaults["run_name"],
        "resume": args.resume,
        "exist_ok": args.exist_ok,
        "verbose": True,
        "deterministic": False,
        "degrees": args.degrees,
        "fliplr": args.fliplr,
        "flipud": args.flipud,
        "scale": args.scale,
        "erasing": args.erasing,
        "mixup": args.mixup,
        "dropout": defaults["dropout"],
        "cos_lr": defaults["cos_lr"],
    }
    if args.freeze is not None:
        train_kwargs["freeze"] = args.freeze
    if args.amp is not None:
        train_kwargs["amp"] = args.amp

    started_at = time.perf_counter()
    results = _train_with_fallback(model, train_kwargs, args)
    elapsed = time.perf_counter() - started_at

    expected_run_dir = Path(defaults["project_dir"]) / str(defaults["run_name"])
    actual_run_dir = Path(str(getattr(results, "save_dir", expected_run_dir)))
    _sync_run_outputs(actual_run_dir, expected_run_dir)

    best_ckpt, last_ckpt = _checkpoint_paths(defaults["project_dir"], defaults["run_name"])
    selected_ckpt = _existing_checkpoint(best_ckpt, last_ckpt)
    metric_report = extract_metrics(defaults["task"], results)
    promoted_ckpt = None
    if defaults["track"] == "stage2":
        if selected_ckpt is None:
            raise SystemExit("Stage 2 training completed without writing best.pt or last.pt; promotion aborted.")
        promoted_ckpt = promote_stage2_checkpoint(Path(selected_ckpt), str(defaults["promoted_checkpoint"]))
    report = {
        "task": defaults["task"],
        "track": defaults["track"],
        "variant": args.variant,
        "model": defaults["weights"],
        "data": defaults["data_path"],
        "epochs": defaults["epochs"],
        "imgsz": defaults["imgsz"],
        "batch": defaults["batch"],
        "lr0": defaults["lr0"],
        "patience": defaults["patience"],
        "augmentation": {
            "degrees": args.degrees,
            "fliplr": args.fliplr,
            "flipud": args.flipud,
            "scale": args.scale,
            "erasing": args.erasing,
            "mixup": args.mixup,
            "dropout": defaults["dropout"],
            "cos_lr": defaults["cos_lr"],
        },
        "train_time_s": round(elapsed, 2),
        "best_ckpt": str(best_ckpt),
        "last_ckpt": str(last_ckpt),
        "actual_run_dir": str(actual_run_dir),
        "expected_run_dir": str(expected_run_dir),
        "selected_ckpt": str(selected_ckpt) if selected_ckpt else None,
        "promoted_ckpt": str(promoted_ckpt) if promoted_ckpt else None,
        **metric_report,
    }

    print(f"\nTraining complete ({elapsed:.0f}s)")
    print(f"Best checkpoint : {best_ckpt}")
    print(f"Last checkpoint : {last_ckpt}")
    if selected_ckpt:
        print(f"Selected ckpt  : {selected_ckpt}")
    else:
        print("Selected ckpt  : none written")
    if promoted_ckpt:
        print(f"Promoted ckpt  : {promoted_ckpt}")
    if defaults["task"] == "classify":
        print(f"Top-1 accuracy : {metric_report.get('top1_accuracy')}")
        print(f"Top-5 accuracy : {metric_report.get('top5_accuracy')}")
    else:
        print(f"mAP@50         : {metric_report.get('mAP50')}")
        print(f"mAP@50-95      : {metric_report.get('mAP50_95')}")

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path = model_dir / str(defaults["report_name"])
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved   : {report_path}")
    if defaults["track"] == "stage2":
        print("Comparison set : run variants n, s, m and compare Top-1 accuracy, size, and latency.")
    elif defaults["track"] == "single_model":
        print("Baseline role  : compare this occupancy detector against the separate Stage 1 + Stage 2 pipeline.")


if __name__ == "__main__":
    main()
