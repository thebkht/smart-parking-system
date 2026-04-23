import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import prepare_dataset, train


def make_image(path: Path, color: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 24), color=(color, color, color)).save(path)


def make_label(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_stratified_split_creates_all_splits():
    class_images = {
        "free": [Path(f"free_{i}.jpg") for i in range(10)],
        "occupied": [Path(f"occ_{i}.jpg") for i in range(10)],
    }
    splits = prepare_dataset.stratified_split(class_images, val_ratio=0.2, test_ratio=0.2, seed=7)
    assert set(splits.keys()) == {"train", "val", "test"}
    assert all(splits[split]["free"] for split in splits)
    assert all(splits[split]["occupied"] for split in splits)


def test_normalize_source_stem_strips_roboflow_suffix():
    assert prepare_dataset.normalize_source_stem("parking_lot_1_mp4-75_jpg.rf.97bf95f9bd26391575f2c08e5866c6bd.jpg") == "parking_lot_1_mp4-75_jpg"
    assert prepare_dataset.normalize_source_stem("sample.jpg") == "sample"


def test_assign_scene_splits_has_no_overlap():
    records = [
        {"scene_id": "scene_a", "normalized_stem": "a", "box_count": 1},
        {"scene_id": "scene_b", "normalized_stem": "b", "box_count": 1},
        {"scene_id": "scene_c", "normalized_stem": "c", "box_count": 1},
        {"scene_id": "scene_d", "normalized_stem": "d", "box_count": 1},
    ]
    splits = prepare_dataset.assign_scene_splits(records, val_ratio=0.2, test_ratio=0.2, seed=7)

    seen = {}
    for split, split_records in splits.items():
        for record in split_records:
            assert record["scene_id"] not in seen
            seen[record["scene_id"]] = split


def test_copy_images_handles_filename_collisions(tmp_path):
    src_a = tmp_path / "src_a" / "free" / "image.jpg"
    src_b = tmp_path / "src_b" / "free" / "image.jpg"
    make_image(src_a, 10)
    make_image(src_b, 20)

    collisions = prepare_dataset.copy_images(
        {"train": {"free": [src_a, src_b]}},
        tmp_path / "out",
    )

    written = list((tmp_path / "out" / "train" / "free").glob("*.jpg"))
    assert len(written) == 2
    assert collisions["train/free"] == 1


def test_sanity_check_writes_report(tmp_path):
    src = tmp_path / "patches" / "free" / "a.jpg"
    make_image(src, 10)
    src2 = tmp_path / "patches" / "occupied" / "b.jpg"
    make_image(src2, 20)

    prepare_dataset.sanity_check_stage2(
        {
            "train": {"free": [src], "occupied": [src2]},
            "val": {"free": [src], "occupied": [src2]},
            "test": {"free": [src], "occupied": [src2]},
        },
        all_images={"free": [src], "occupied": [src2]},
        collisions={},
        report_path=tmp_path / "stage2_data" / "dataset_report.json",
    )

    report = json.loads((tmp_path / "stage2_data" / "dataset_report.json").read_text())
    assert report["class_counts"] == {"free": 1, "occupied": 1}
    assert report["dimension_summary"]["free"]["sampled"] == 1


def test_weather_split_paths_requires_expected_layout(tmp_path):
    try:
        prepare_dataset.weather_split_paths(tmp_path)
    except SystemExit as exc:
        assert "Expected weather layout" in str(exc)
    else:
        raise AssertionError("expected weather_split_paths to fail")


def test_prepare_single_model_detection_preserves_two_classes(tmp_path):
    root = tmp_path / "pklot"
    make_image(root / "train" / "images" / "parking_lot_1_mp4-0_jpg.rf.aaaa.jpg", 20)
    make_label(
        root / "train" / "labels" / "parking_lot_1_mp4-0_jpg.rf.aaaa.txt",
        [
            "0 0.5 0.5 0.4 0.4",
            "1 0.2 0.2 0.1 0.1",
        ],
    )
    make_image(root / "valid" / "images" / "parking_lot_2_mp4-1_jpg.rf.bbbb.jpg", 20)
    make_label(root / "valid" / "labels" / "parking_lot_2_mp4-1_jpg.rf.bbbb.txt", ["1 0.5 0.5 0.4 0.4"])
    make_image(root / "test" / "images" / "parking_lot_3_mp4-2_jpg.rf.cccc.jpg", 20)
    make_label(root / "test" / "labels" / "parking_lot_3_mp4-2_jpg.rf.cccc.txt", ["0 0.5 0.5 0.4 0.4"])

    out_dir = tmp_path / "single_model_data"
    yaml_path = tmp_path / "single_model.yaml"
    prepare_dataset.prepare_single_model_detection(root, out_dir, yaml_path)

    all_labels = []
    for label_path in sorted((out_dir).glob("*/*/*.txt")):
        all_labels.extend(label_path.read_text(encoding="utf-8").splitlines())
    assert sorted(all_labels) == sorted([
        "0 0.500000 0.500000 0.400000 0.400000",
        "1 0.200000 0.200000 0.100000 0.100000",
        "1 0.500000 0.500000 0.400000 0.400000",
        "0 0.500000 0.500000 0.400000 0.400000",
    ])
    yaml_text = yaml_path.read_text(encoding="utf-8")
    assert "names:" in yaml_text
    assert "- free" in yaml_text
    assert "- occupied" in yaml_text
    report = json.loads((out_dir / prepare_dataset.DETECTION_REPORT).read_text(encoding="utf-8"))
    assert report["track"] == "single_model"
    assert report["empty_label_frames_excluded"] == 0
    assert report["leakage_checks"]["scene_leakage_detected"] is False


def test_iter_detection_boxes_converts_polygon_to_clipped_box(tmp_path):
    label = tmp_path / "sample.txt"
    make_label(
        label,
        [
            "1 0.10 0.20 0.40 0.20 0.45 0.60 0.05 0.70",
            "0 1.10 0.50 0.40 0.40",
        ],
    )

    rows = list(prepare_dataset.iter_detection_boxes(label))

    assert rows[0][0] == 1
    assert rows[0][1] == pytest.approx((0.25, 0.45, 0.4, 0.5))
    assert rows[0][2] == "polygon"
    assert rows[1][0] == 0
    assert rows[1][1] == pytest.approx((0.95, 0.5, 0.1, 0.4))
    assert rows[1][2] == "box"


def test_prepare_single_model_detection_converts_polygon_labels(tmp_path):
    root = tmp_path / "pklot"
    make_image(root / "train" / "images" / "parking_lot_1_mp4-0_jpg.rf.aaaa.jpg", 20)
    make_label(
        root / "train" / "labels" / "parking_lot_1_mp4-0_jpg.rf.aaaa.txt",
        [
            "0 0.10 0.20 0.40 0.20 0.40 0.60 0.10 0.60",
            "1 0.50 0.50 0.70 0.50 0.70 0.90 0.50 0.90",
        ],
    )
    make_image(root / "valid" / "images" / "parking_lot_2_mp4-1_jpg.rf.bbbb.jpg", 20)
    make_label(
        root / "valid" / "labels" / "parking_lot_2_mp4-1_jpg.rf.bbbb.txt",
        ["1 0.50 0.50 0.70 0.50 0.70 0.90 0.50 0.90"],
    )
    make_image(root / "test" / "images" / "parking_lot_3_mp4-2_jpg.rf.cccc.jpg", 20)
    make_label(
        root / "test" / "labels" / "parking_lot_3_mp4-2_jpg.rf.cccc.txt",
        ["0 0.10 0.20 0.40 0.20 0.40 0.60 0.10 0.60"],
    )

    out_dir = tmp_path / "single_model_data"
    yaml_path = tmp_path / "single_model.yaml"
    prepare_dataset.prepare_single_model_detection(root, out_dir, yaml_path)

    all_labels = []
    for label_path in sorted((out_dir).glob("*/*/*.txt")):
        all_labels.extend(label_path.read_text(encoding="utf-8").splitlines())
    assert sorted(all_labels) == sorted([
        "0 0.250000 0.400000 0.300000 0.400000",
        "1 0.600000 0.700000 0.200000 0.400000",
        "1 0.600000 0.700000 0.200000 0.400000",
        "0 0.250000 0.400000 0.300000 0.400000",
    ])
    report = json.loads((out_dir / prepare_dataset.DETECTION_REPORT).read_text(encoding="utf-8"))
    assert report["polygon_labels_converted"] == 4


def test_prepare_stage1_excludes_empty_label_frames(tmp_path):
    root = tmp_path / "pklot"
    make_image(root / "train" / "images" / "parking_lot_1_mp4-0_jpg.rf.aaaa.jpg", 20)
    make_label(root / "train" / "labels" / "parking_lot_1_mp4-0_jpg.rf.aaaa.txt", ["1 0.5 0.5 0.4 0.4"])
    make_image(root / "train" / "images" / "parking_lot_1_mp4-1_jpg.rf.bbbb.jpg", 20)
    make_label(root / "train" / "labels" / "parking_lot_1_mp4-1_jpg.rf.bbbb.txt", [])
    make_image(root / "valid" / "images" / "parking_lot_2_mp4-2_jpg.rf.cccc.jpg", 20)
    make_label(root / "valid" / "labels" / "parking_lot_2_mp4-2_jpg.rf.cccc.txt", ["0 0.5 0.5 0.4 0.4"])
    make_image(root / "test" / "images" / "parking_lot_3_mp4-3_jpg.rf.dddd.jpg", 20)
    make_label(root / "test" / "labels" / "parking_lot_3_mp4-3_jpg.rf.dddd.txt", ["0 0.5 0.5 0.4 0.4"])

    out_dir = tmp_path / "stage1_data"
    yaml_path = tmp_path / "stage1.yaml"
    prepare_dataset.prepare_stage1(root, out_dir, yaml_path)

    written = sorted((out_dir / "train" / "images").glob("*.jpg"))
    assert written
    assert all("parking_lot_1_mp4-1_jpg" not in path.name for path in written)
    report = json.loads((out_dir / prepare_dataset.DETECTION_REPORT).read_text(encoding="utf-8"))
    assert report["track"] == "stage1"
    assert report["empty_label_frames_excluded"] == 1
    assert report["leakage_checks"]["scene_leakage_detected"] is False


def test_collect_roboflow_patches_uses_polygon_boxes(tmp_path):
    root = tmp_path / "pklot"
    make_image(root / "train" / "images" / "parking_lot_1_mp4-0_jpg.rf.aaaa.jpg", 128)
    make_label(
        root / "train" / "labels" / "parking_lot_1_mp4-0_jpg.rf.aaaa.txt",
        ["1 0.25 0.25 0.50 0.25 0.50 0.75 0.25 0.75"],
    )
    make_image(root / "valid" / "images" / "parking_lot_2_mp4-1_jpg.rf.bbbb.jpg", 128)
    make_label(
        root / "valid" / "labels" / "parking_lot_2_mp4-1_jpg.rf.bbbb.txt",
        ["0 0.25 0.25 0.50 0.25 0.50 0.75 0.25 0.75"],
    )
    make_image(root / "test" / "images" / "parking_lot_3_mp4-2_jpg.rf.cccc.jpg", 128)
    make_label(
        root / "test" / "labels" / "parking_lot_3_mp4-2_jpg.rf.cccc.txt",
        ["1 0.25 0.25 0.50 0.25 0.50 0.75 0.25 0.75"],
    )

    patches = prepare_dataset.collect_roboflow_patches(root, tmp_path / "patches")

    assert len(patches["occupied"]) == 2
    with Image.open(patches["occupied"][0]) as patch:
        assert patch.size == (8, 12)


def test_prepare_stage2_inherits_scene_holdout(tmp_path):
    root = tmp_path / "pklot"
    make_image(root / "train" / "images" / "parking_lot_1_mp4-0_jpg.rf.aaaa.jpg", 100)
    make_label(root / "train" / "labels" / "parking_lot_1_mp4-0_jpg.rf.aaaa.txt", ["0 0.3 0.3 0.2 0.2", "1 0.7 0.7 0.2 0.2"])
    make_image(root / "valid" / "images" / "parking_lot_2_mp4-1_jpg.rf.bbbb.jpg", 100)
    make_label(root / "valid" / "labels" / "parking_lot_2_mp4-1_jpg.rf.bbbb.txt", ["0 0.3 0.3 0.2 0.2", "1 0.7 0.7 0.2 0.2"])
    make_image(root / "test" / "images" / "parking_lot_3_mp4-2_jpg.rf.cccc.jpg", 100)
    make_label(root / "test" / "labels" / "parking_lot_3_mp4-2_jpg.rf.cccc.txt", ["0 0.3 0.3 0.2 0.2", "1 0.7 0.7 0.2 0.2"])

    args = SimpleNamespace(
        pklot_dir=str(root),
        cnrpark_dir=None,
        patch_cache=str(tmp_path / "patches"),
        stage2_output=str(tmp_path / "stage2_data"),
        pklot_test_output=str(tmp_path / "pklot_test"),
        cnrpark_test_output=str(tmp_path / "cnrpark_test"),
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
    )
    prepare_dataset.prepare_stage2(args)

    report = json.loads((tmp_path / "stage2_data" / prepare_dataset.SANITY_REPORT).read_text(encoding="utf-8"))
    assert report["scene_holdout"]["source"] == "pklot_scene_holdout"
    assert report["scene_holdout"]["leakage_checks"]["scene_leakage_detected"] is False


def test_prepare_stage2_accepts_roi_annotations_dataset(tmp_path):
    root = tmp_path / "parking_rois"
    images = root / "images"
    make_image(images / "frame_a.jpg", 90)
    make_image(images / "frame_b.jpg", 140)
    make_image(images / "frame_c.jpg", 180)
    annotations = {
        "train": {
            "file_names": ["frame_a.jpg"],
            "rois_list": [[
                [[0.10, 0.10], [0.30, 0.10], [0.30, 0.40], [0.10, 0.40]],
                [[0.50, 0.20], [0.80, 0.20], [0.80, 0.60], [0.50, 0.60]],
            ]],
            "occupancy_list": [[False, True]],
        },
        "valid": {
            "file_names": ["frame_b.jpg"],
            "rois_list": [[
                [[0.15, 0.15], [0.35, 0.15], [0.35, 0.45], [0.15, 0.45]],
                [[0.55, 0.25], [0.85, 0.25], [0.85, 0.65], [0.55, 0.65]],
            ]],
            "occupancy_list": [[False, True]],
        },
        "test": {
            "file_names": ["frame_c.jpg"],
            "rois_list": [[
                [[0.20, 0.20], [0.40, 0.20], [0.40, 0.50], [0.20, 0.50]],
                [[0.60, 0.30], [0.90, 0.30], [0.90, 0.70], [0.60, 0.70]],
            ]],
            "occupancy_list": [[False, True]],
        },
    }
    (root / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")

    args = SimpleNamespace(
        pklot_dir=str(root),
        cnrpark_dir=None,
        patch_cache=str(tmp_path / "patches"),
        stage2_output=str(tmp_path / "stage2_data"),
        pklot_test_output=str(tmp_path / "pklot_test"),
        cnrpark_test_output=str(tmp_path / "cnrpark_test"),
        val_ratio=0.2,
        test_ratio=0.2,
        seed=7,
    )
    prepare_dataset.prepare_stage2(args)

    assert len(list((tmp_path / "stage2_data" / "train" / "free").glob("*.jpg"))) == 1
    assert len(list((tmp_path / "stage2_data" / "train" / "occupied").glob("*.jpg"))) == 1
    assert len(list((tmp_path / "stage2_data" / "val" / "free").glob("*.jpg"))) == 1
    assert len(list((tmp_path / "stage2_data" / "val" / "occupied").glob("*.jpg"))) == 1
    assert len(list((tmp_path / "stage2_data" / "test" / "free").glob("*.jpg"))) == 1
    assert len(list((tmp_path / "stage2_data" / "test" / "occupied").glob("*.jpg"))) == 1

    report = json.loads((tmp_path / "stage2_data" / prepare_dataset.SANITY_REPORT).read_text(encoding="utf-8"))
    assert report["split_strategy"] == "source_presplit"
    assert report["annotation_source"] == "annotations.json"
    assert report["invalid_polygons_skipped"] == 0
    assert report["splits"]["val"] == {"free": 1, "occupied": 1}


def test_collect_cnrpark_patches_reads_official_labels_layout(tmp_path):
    root = tmp_path / "cnr"
    sunny_free = root / "PATCHES" / "SUNNY" / "2015-11-22" / "camera6" / "S_2015-11-22_09.47_C06_205.jpg"
    rainy_busy = root / "PATCHES" / "RAINY" / "2015-11-23" / "camera1" / "R_2015-11-23_09.47_C01_099.jpg"
    make_image(sunny_free, 90)
    make_image(rainy_busy, 180)
    make_label(
        root / "LABELS" / "split.txt",
        [
            "PATCHES/SUNNY/2015-11-22/camera6/S_2015-11-22_09.47_C06_205.jpg 0",
            "PATCHES/RAINY/2015-11-23/camera1/R_2015-11-23_09.47_C01_099.jpg 1",
        ],
    )

    class_map, weather_map = prepare_dataset.collect_cnrpark_patches(root)

    assert class_map["free"] == [sunny_free]
    assert class_map["occupied"] == [rainy_busy]
    assert weather_map["sunny"]["free"] == [sunny_free]
    assert weather_map["rainy"]["occupied"] == [rainy_busy]


def test_copy_weather_flat_writes_per_weather_layout(tmp_path):
    src = tmp_path / "src"
    sunny_free = src / "PATCHES" / "SUNNY" / "2015-11-22" / "camera6" / "a.jpg"
    cloudy_occ = src / "PATCHES" / "OVERCAST" / "2015-11-22" / "camera6" / "b.jpg"
    make_image(sunny_free, 80)
    make_image(cloudy_occ, 120)

    collisions = prepare_dataset.copy_weather_flat(
        {
            "sunny": {"free": [sunny_free], "occupied": []},
            "cloudy": {"free": [], "occupied": [cloudy_occ]},
            "rainy": {"free": [], "occupied": []},
        },
        tmp_path / "weather",
        source_root=src,
    )

    assert collisions == {}
    assert (tmp_path / "weather" / "sunny" / "free").glob("*.jpg")
    assert len(list((tmp_path / "weather" / "sunny" / "free").glob("*.jpg"))) == 1
    assert len(list((tmp_path / "weather" / "cloudy" / "occupied").glob("*.jpg"))) == 1


def test_train_requires_explicit_mode(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train.py"])
    with pytest.raises(SystemExit) as exc:
        train.parse_args()
    assert exc.value.code == 2


def test_train_stage2_mode_resolution(monkeypatch, tmp_path):
    data_dir = tmp_path / "stage2_data"
    data_dir.mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(train, "STAGE2_DATA_DIR", str(data_dir))
    monkeypatch.setattr(sys, "argv", ["train.py", "--stage2"])
    args = train.parse_args()
    defaults = train.task_defaults(args)
    assert defaults["task"] == "classify"
    assert defaults["track"] == "stage2"
    assert defaults["data_path"] == str(data_dir)


def test_train_single_model_mode_resolution(monkeypatch, tmp_path):
    yaml_path = tmp_path / "single_model.yaml"
    yaml_path.write_text("path: /tmp\ntrain: train/images\nval: valid/images\ntest: test/images\nnc: 2\nnames: [free, occupied]\n", encoding="utf-8")
    monkeypatch.setattr(train, "SINGLE_MODEL_YAML", str(yaml_path))
    monkeypatch.setattr(sys, "argv", ["train.py", "--single-model"])
    args = train.parse_args()
    defaults = train.task_defaults(args)
    assert defaults["task"] == "detect"
    assert defaults["track"] == "single_model"
    assert defaults["project_dir"] == train.SINGLE_MODEL_PROJECT


def test_train_stage2_accuracy_defaults(monkeypatch, tmp_path):
    data_dir = tmp_path / "stage2_data"
    data_dir.mkdir()
    monkeypatch.setattr(train, "STAGE2_DATA_DIR", str(data_dir))
    monkeypatch.setattr(sys, "argv", ["train.py", "--stage2"])
    args = train.parse_args()
    defaults = train.task_defaults(args)
    assert defaults["lr0"] == train.STAGE2_LR
    assert defaults["patience"] == train.STAGE2_PATIENCE
    assert defaults["dropout"] == 0.1
    assert defaults["cos_lr"] is True


def test_train_stage1_allows_custom_checkpoint_and_run_paths(monkeypatch, tmp_path):
    yaml_path = tmp_path / "stage1.yaml"
    yaml_path.write_text("path: /tmp\ntrain: train/images\nval: val/images\ntest: test/images\nnc: 1\nnames: [space]\n", encoding="utf-8")
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    monkeypatch.setattr(train, "STAGE1_YAML", str(yaml_path))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--stage1",
            "--weights",
            str(weights),
            "--project",
            "runs/stage1_finetune",
            "--name",
            "my_camera",
            "--freeze",
            "10",
            "--optimizer",
            "SGD",
            "--no-amp",
            "--exist-ok",
        ],
    )

    args = train.parse_args()
    defaults = train.task_defaults(args)

    assert defaults["weights"] == str(weights)
    assert defaults["project_dir"] == "runs/stage1_finetune"
    assert defaults["run_name"] == "my_camera"
    assert args.freeze == 10
    assert args.optimizer == "SGD"
    assert args.amp is False
    assert args.exist_ok is True


def test_existing_checkpoint_prefers_best_then_last(tmp_path):
    best_ckpt, last_ckpt = train._checkpoint_paths(str(tmp_path / "runs"), "exp")
    last_ckpt.parent.mkdir(parents=True, exist_ok=True)

    assert train._existing_checkpoint(best_ckpt, last_ckpt) is None

    last_ckpt.write_bytes(b"last")
    assert train._existing_checkpoint(best_ckpt, last_ckpt) == last_ckpt

    best_ckpt.write_bytes(b"best")
    assert train._existing_checkpoint(best_ckpt, last_ckpt) == best_ckpt


def test_nan_recovery_patch_raises_clear_error_when_last_missing(tmp_path):
    train._patch_ultralytics_trainer_for_nan_checkpoints()

    fake_trainer = SimpleNamespace(
        loss=torch.tensor(float("nan")),
        fitness=float("nan"),
        best_fitness=0.0,
        start_epoch=0,
        last=tmp_path / "weights" / "last.pt",
        nan_recovery_attempts=0,
    )

    with pytest.raises(RuntimeError) as exc:
        train.BaseTrainer._handle_nan_recovery(fake_trainer, epoch=0)

    message = str(exc.value)
    assert "before a recoverable checkpoint was written" in message
    assert str(fake_trainer.last) in message
