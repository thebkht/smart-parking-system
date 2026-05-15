import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import finalize


def write_csv(path: Path, header: str, row: str) -> None:
    path.write_text(header + "\n" + row + "\n", encoding="utf-8")


def test_stage2_inventory_counts_split_images(tmp_path):
    for split in ("train", "val", "test"):
        for class_name in ("free", "occupied"):
            path = tmp_path / "stage2_data" / split / class_name / f"{split}_{class_name}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"jpg")

    inventory = finalize.stage2_inventory(tmp_path / "stage2_data")

    assert inventory["present"] is True
    assert inventory["splits"]["train"] == {"free": 1, "occupied": 1}


def test_required_artifact_checks_mark_missing_items(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    write_csv(logs / "stage2_evaluation.csv", "model,top1_accuracy", "best,0.97")

    manifest = {
        "datasets": {
            "pklot_test": {"present": True},
            "cnrpark_test": {"present": False},
            "stage2_weather": {"present": False},
        },
        "checkpoints": {
            "stage1_s": {"present": False},
            "stage1_m": {"present": True},
            "stage2": {
                "n": {"present": True},
                "s": {"present": False},
                "m": {"present": False},
            },
        },
        "metrics": {
            "stage1_evaluation": None,
            "stage2_evaluation": {"top1_accuracy": "0.97"},
            "stage2_model_comparison": None,
            "stage2_threshold_sweep": None,
            "stage2_cross_dataset": None,
            "stage2_per_weather": None,
            "benchmark_results": None,
            "bandwidth_report_present": False,
            "stability_summary": None,
        },
    }

    checks = finalize.required_artifact_checks(manifest)

    assert checks["stage1_detector_checkpoint"] is True
    assert checks["stage2_n_checkpoint"] is True
    assert checks["stage2_s_checkpoint"] is False
    assert checks["stage2_eval_table"] is True
    assert checks["cross_dataset_eval"] is False
    assert checks["per_weather_eval"] is True
    assert checks["benchmark_results"] is False


def test_write_markdown_emits_summary_file(tmp_path):
    manifest = {
        "datasets": {
            "stage1": {"splits": {"train": {"images_kept": 1, "boxes_kept": 2, "scene_count": 3}}},
            "stage2": {"splits": {"val": {"free": 4, "occupied": 5}}},
            "stage2_weather": {"present": True, "splits": {"sunny": {"free": 2, "occupied": 3}}},
            "pklot_test": {"present": True, "free": 6, "occupied": 7},
            "cnrpark_test": {"present": False},
        },
        "checkpoints": {
            "stage1_s": {"present": True, "path": "runs/stage1_det/yolov8s_stage1/weights/best.pt"},
            "stage1_m": {"present": False, "path": "runs/stage1_det/yolov8m_stage1/weights/best.pt"},
            "stage2": {
                "n": {"present": True, "path": "n.pt"},
                "s": {"present": True, "path": "s.pt"},
                "m": {"present": True, "path": "m.pt"},
            },
        },
        "metrics": {
            "stage1_evaluation": {"mAP50": "0.7"},
            "stage2_evaluation": {"top1_accuracy": "0.98"},
            "stage2_model_comparison": {"model": "yolov8s_stage2"},
            "stage2_threshold_sweep": {"threshold": "0.55"},
            "stage2_cross_dataset": {"dataset": "pklot_test"},
            "stage2_per_weather": {"dataset": "sunny"},
        },
        "checks": {"stage1_detector_checkpoint": True},
    }

    output = tmp_path / "summary.md"
    finalize.write_markdown(manifest, output)

    text = output.read_text(encoding="utf-8")
    assert "Final Artifact Summary" in text
    assert "Stage 2 `yolov8s-cls`" in text
    assert "Weather Export" in text
    assert "stage1_detector_checkpoint: PASS" in text


def test_latest_csv_row_tolerates_legacy_spaced_csv(tmp_path):
    csv_path = tmp_path / "stage2_cross_dataset.csv"
    csv_path.write_text(
        "model         , dataset     , threshold, top1_accuracy, precision, recall, f1 , sample_count, support_free, support_occupied, confusion_matrix\n"
        "yolov8n_stage2, cnrpark_test,       0.5,        0.8983,     0.974, 0.8364, 0.9,        21746,         9849,            11897, \"[[9583, 266], [1946, 9951]]\"\n",
        encoding="utf-8",
    )

    row = finalize.latest_csv_row(csv_path)

    assert row == {
        "model": "yolov8n_stage2",
        "dataset": "cnrpark_test",
        "threshold": "0.5",
        "top1_accuracy": "0.8983",
        "precision": "0.974",
        "recall": "0.8364",
        "f1": "0.9",
        "sample_count": "21746",
        "support_free": "9849",
        "support_occupied": "11897",
        "confusion_matrix": "[[9583, 266], [1946, 9951]]",
    }


def test_best_csv_row_selects_highest_metric(tmp_path):
    csv_path = tmp_path / "stage2_threshold_sweep.csv"
    csv_path.write_text(
        "model,f1,threshold\n"
        "a,0.81,0.3\n"
        "a,0.92,0.1\n"
        "a,0.88,0.5\n",
        encoding="utf-8",
    )

    row = finalize.best_csv_row(csv_path, "f1")

    assert row == {"model": "a", "f1": "0.92", "threshold": "0.1"}


def test_collect_checkpoints_prefers_acpds_stage2_layout(tmp_path):
    checkpoint = tmp_path / "runs" / "acpds_cls" / "yolov8s_stage2" / "weights" / "best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"pt")

    manifest = finalize.collect_checkpoints(tmp_path / "runs", tmp_path / "artifacts")

    assert manifest["stage2"]["s"]["present"] is True
    assert manifest["stage2"]["s"]["path"].endswith("runs/acpds_cls/yolov8s_stage2/weights/best.pt")
