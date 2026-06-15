import sys
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from edge.detect import (
    SmoothingBuffer,
    apply_perspective_transform,
    box_area,
    build_payload,
    classify_patch,
    dump_warp_samples,
    maybe_dump_warp_samples,
    fetch_rois_from_backend,
    geometry_to_box,
    handoff_to_builtin_camera,
    crop_patch,
    filter_stage1_box,
    get_spot_boxes,
    parse_args,
    open_camera_capture,
    reopen_camera_capture,
    resolve_camera_runtime,
    resolve_camera_source,
    roi_bounds,
    load_rois,
    load_spot_geometries,
    normalize_rois,
    load_perspective_transform,
    resolve_settings,
    run_pipeline,
    write_latest_frame,
    DEFAULT_LATEST_FRAME_PATH,
)


class FakeProbs:
    def __init__(self, top1: int, top1conf: float):
        self.top1 = top1
        self.top1conf = top1conf


class FakeResult:
    def __init__(self, label: str, confidence: float):
        self.names = {0: "free", 1: "occupied"}
        self.probs = FakeProbs(1 if label == "occupied" else 0, confidence)


class FakeYOLO:
    def __init__(self, labels):
        self.labels = iter(labels)

    def __call__(self, *_args, **_kwargs):
        label, confidence = next(self.labels)
        return [FakeResult(label, confidence)]


class FakeCapture:
    def __init__(self, opens: bool, reads: bool):
        self._opens = opens
        self._reads = reads
        self.released = False

    def isOpened(self):
        return self._opens

    def read(self):
        return self._reads, np.zeros((4, 4, 3), dtype=np.uint8)

    def set(self, *_args):
        return True

    def release(self):
        self.released = True


class FakeArray:
    def __init__(self, values):
        self._values = np.array(values, dtype=float)

    def cpu(self):
        return self

    def numpy(self):
        return self._values


class FakeDetectResult:
    def __init__(self, boxes):
        self.boxes = SimpleNamespace(xyxy=FakeArray(boxes))


class FakeDetectYOLO:
    def __init__(self, boxes):
        self._boxes = boxes

    def __call__(self, *_args, **_kwargs):
        return [FakeDetectResult(self._boxes)]


def test_normalize_rois_accepts_valid_boxes():
    rois = normalize_rois({"spot_1": [0, 1, 10, 20]})
    assert rois == {"spot_1": (0, 1, 10, 20)}


def test_load_rois_reads_config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("rois:\n  a: [1, 2, 3, 4]\n", encoding="utf-8")
    assert load_rois(config, backend_url=None) == {"a": (1, 2, 3, 4)}


def test_load_spot_geometries_reads_config_boxes_as_quads(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("rois:\n  a: [1, 2, 5, 6]\n", encoding="utf-8")

    geometries = load_spot_geometries(config, backend_url=None)

    assert np.array_equal(
        geometries["a"], np.array([[1, 2], [5, 2], [5, 6], [1, 6]], dtype=np.float32)
    )


def test_fetch_rois_from_backend_preserves_quad(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "spots": [
                    {
                        "spot_id": "spot_1",
                        "points": [[10, 10], [30, 8], [35, 28], [8, 24]],
                    }
                ]
            }

    monkeypatch.setattr(
        "edge.detect.requests.get", lambda *args, **kwargs: FakeResponse()
    )

    rois = fetch_rois_from_backend("http://127.0.0.1:8000")

    assert rois["spot_1"].shape == (4, 2)
    assert geometry_to_box(rois["spot_1"]) == (8, 8, 35, 28)
    assert not np.array_equal(
        rois["spot_1"], np.array([[8, 8], [35, 8], [35, 28], [8, 28]], dtype=np.float32)
    )


def test_fetch_rois_from_backend_skips_malformed_spots(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "spots": [
                    {
                        "spot_id": "spot_bad",
                        "points": [[10], [30, 8], [35, 28], [8, 24]],
                    },
                    {
                        "spot_id": "spot_good",
                        "points": [[40, 10], [60, 10], [60, 30], [40, 30]],
                    },
                ]
            }

    monkeypatch.setattr(
        "edge.detect.requests.get", lambda *args, **kwargs: FakeResponse()
    )

    rois = fetch_rois_from_backend("http://127.0.0.1:8000")

    assert "spot_bad" not in rois
    assert geometry_to_box(rois["spot_good"]) == (40, 10, 60, 30)


def test_load_perspective_transform_builds_output_size_from_config():
    transform = load_perspective_transform(
        {
            "preprocess": {
                "perspective": {
                    "source_points": [[0, 0], [9, 0], [9, 9], [0, 9]],
                    "output_size": [20, 10],
                }
            }
        }
    )

    assert transform is not None
    _matrix, output_size = transform
    assert output_size == (20, 10)


def test_apply_perspective_transform_returns_warped_frame():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    frame[0, 0] = (255, 255, 255)
    transform = load_perspective_transform(
        {
            "preprocess": {
                "perspective": {
                    "source_points": [[0, 0], [9, 0], [9, 9], [0, 9]],
                    "destination_points": [[1, 0], [9, 0], [9, 9], [1, 9]],
                    "output_size": [10, 10],
                }
            }
        }
    )

    warped = apply_perspective_transform(frame, transform)

    assert warped[0, 1].sum() > 0


def test_resolve_camera_source_accepts_numeric_indexes():
    assert resolve_camera_source("0") == 0
    assert resolve_camera_source("12") == 12


def test_open_camera_capture_uses_backend_when_provided(monkeypatch):
    calls = []

    def fake_video_capture(*args):
        calls.append(args)
        return FakeCapture(opens=True, reads=True)

    monkeypatch.setattr("edge.detect.cv2.VideoCapture", fake_video_capture)

    capture = open_camera_capture(1, 1200)

    assert capture.isOpened() is True
    assert calls == [(1, 1200)]


def test_reopen_camera_capture_releases_then_reopens(monkeypatch):
    calls = []
    current = FakeCapture(opens=True, reads=True)

    def fake_video_capture(*args):
        calls.append(args)
        return FakeCapture(opens=True, reads=True)

    monkeypatch.setattr("edge.detect.cv2.VideoCapture", fake_video_capture)

    reopened = reopen_camera_capture(3, 1200, current)

    assert current.released is True
    assert reopened.isOpened() is True
    assert calls == [(3, 1200)]


def test_resolve_camera_source_rejects_invalid_tokens():
    with pytest.raises(ValueError, match="Invalid --camera value"):
        resolve_camera_source("front-door")


def test_resolve_camera_source_rejects_iphone_outside_macos(monkeypatch):
    monkeypatch.setattr("edge.detect.platform.system", lambda: "Linux")
    with pytest.raises(ValueError, match="supported only on macOS"):
        resolve_camera_source("iphone")


def test_resolve_camera_source_requires_detected_iphone_camera(monkeypatch):
    monkeypatch.setattr("edge.detect.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "edge.detect.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout='{"SPCameraDataType": [{"_name": "FaceTime HD Camera"}]}'
        ),
    )
    with pytest.raises(ValueError, match="No iPhone camera detected"):
        resolve_camera_source("iphone")


def test_resolve_camera_source_uses_matching_iphone_camera(monkeypatch):
    monkeypatch.setattr("edge.detect.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "edge.detect.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout='{"SPCameraDataType": [{"_name": "John’s iPhone Camera"}]}'
        ),
    )
    monkeypatch.setattr("edge.detect._macos_camera_backend", lambda: 1200)

    attempted = []

    def fake_video_capture(index, backend):
        attempted.append((index, backend))
        return FakeCapture(opens=index in {2, 3}, reads=index in {2, 3})

    monkeypatch.setattr("edge.detect.cv2.VideoCapture", fake_video_capture)

    assert resolve_camera_source("iphone", probe_limit=4) == 3
    assert attempted == [(0, 1200), (1, 1200), (2, 1200), (3, 1200)]


def test_resolve_camera_source_fails_when_no_camera_index_opens(monkeypatch):
    monkeypatch.setattr("edge.detect.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "edge.detect.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout='{"SPCameraDataType": [{"_name": "Continuity Camera"}]}'
        ),
    )
    monkeypatch.setattr("edge.detect._macos_camera_backend", lambda: 1200)
    monkeypatch.setattr(
        "edge.detect.cv2.VideoCapture",
        lambda index, backend: FakeCapture(opens=False, reads=False),
    )

    with pytest.raises(
        RuntimeError, match="could not open any AVFoundation camera index"
    ):
        resolve_camera_source("iphone", probe_limit=3)


def test_resolve_camera_source_stops_after_miss_streak_once_candidates_exist(
    monkeypatch,
):
    monkeypatch.setattr("edge.detect.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "edge.detect.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(
            stdout='{"SPCameraDataType": [{"_name": "Continuity Camera"}]}'
        ),
    )
    monkeypatch.setattr("edge.detect._macos_camera_backend", lambda: 1200)

    attempted = []
    openings = {0: True, 1: True, 2: False, 3: False, 4: True}

    def fake_video_capture(index, backend):
        attempted.append((index, backend))
        opens = openings.get(index, False)
        return FakeCapture(opens=opens, reads=opens)

    monkeypatch.setattr("edge.detect.cv2.VideoCapture", fake_video_capture)

    assert resolve_camera_source("iphone", probe_limit=10) == 1
    assert attempted == [(0, 1200), (1, 1200), (2, 1200), (3, 1200)]


def test_resolve_camera_runtime_includes_builtin_fallback_for_iphone(monkeypatch):
    monkeypatch.setattr("edge.detect.platform.system", lambda: "Darwin")
    monkeypatch.setattr(
        "edge.detect._resolve_macos_iphone_camera", lambda probe_limit=10: (3, 0)
    )
    monkeypatch.setattr("edge.detect._macos_camera_backend", lambda: 1200)

    camera, backend, fallback_camera, fallback_backend, label = resolve_camera_runtime(
        "iphone"
    )

    assert camera == 3
    assert backend == 1200
    assert fallback_camera == 0
    assert fallback_backend == 1200
    assert label == "iphone"


def test_resolve_camera_runtime_handles_numeric_without_fallback():
    camera, backend, fallback_camera, fallback_backend, label = resolve_camera_runtime(
        "1"
    )

    assert camera == 1
    assert backend is None
    assert fallback_camera is None
    assert fallback_backend is None
    assert label == "1"


def test_handoff_to_builtin_camera_opens_and_releases_capture(monkeypatch):
    capture = FakeCapture(opens=True, reads=True)
    calls = []

    def fake_open_camera_capture(index, backend):
        calls.append((index, backend))
        return capture

    monkeypatch.setattr("edge.detect.open_camera_capture", fake_open_camera_capture)

    handoff_to_builtin_camera(0, 1200)

    assert calls == [(0, 1200)]
    assert capture.released is True


def test_crop_patch_returns_none_for_invalid_box():
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    assert crop_patch(frame, (10, 10, 5, 5)) is None


def test_classify_patch_thresholds_occupied_predictions():
    frame = np.ones((50, 50, 3), dtype=np.uint8)
    model = FakeYOLO([("occupied", 0.49), ("occupied", 0.82)])
    corners = np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float32)

    first_status, first_conf = classify_patch(frame, corners, model, "cpu", 0.5)
    second_status, second_conf = classify_patch(frame, corners, model, "cpu", 0.5)

    assert (first_status, round(first_conf, 2)) == ("free", 0.49)
    assert (second_status, round(second_conf, 2)) == ("occupied", 0.82)


def test_classify_patch_warps_to_128_for_model_input():
    frame = np.ones((40, 40, 3), dtype=np.uint8)
    seen = []

    class RecordingYOLO(FakeYOLO):
        def __call__(self, patch, **kwargs):
            seen.append(patch.shape)
            return super().__call__(patch, **kwargs)

    classify_patch(
        frame,
        np.array([[2, 3], [24, 1], [26, 25], [1, 24]], dtype=np.float32),
        RecordingYOLO([("occupied", 0.9)]),
        "cpu",
        0.5,
    )

    assert seen == [(128, 128, 3)]


def test_classify_patch_invalid_quad_falls_back_to_free():
    frame = np.ones((40, 40, 3), dtype=np.uint8)
    status, confidence = classify_patch(
        frame,
        np.array([[100, 100], [100, 100], [120, 120], [120, 120]], dtype=np.float32),
        FakeYOLO([("occupied", 0.9)]),
        "cpu",
        0.5,
    )
    assert status == "free"
    assert confidence == 0.0


def test_smoothing_buffer_majority_vote():
    buffer = SmoothingBuffer(window=3)
    buffer.update({"spot_1": "occupied"})
    buffer.update({"spot_1": "free"})
    buffer.update({"spot_1": "occupied"})
    assert buffer.get_status()["spot_1"] == "occupied"


def test_smoothing_buffer_tie_resolves_to_free():
    buffer = SmoothingBuffer(window=4)
    for value in ("occupied", "free", "occupied", "free"):
        buffer.update({"spot_1": value})
    assert buffer.get_status()["spot_1"] == "free"


def test_build_payload_uses_v3_schema():
    payload = build_payload({"spot_1": "free"}, {"spot_1": 0.8123})
    assert payload["spots"] == {"spot_1": "free"}
    assert payload["confidence"] == {"spot_1": 0.812}
    assert payload["timestamp"].endswith("Z")


def test_roi_bounds_returns_union_box():
    bounds = roi_bounds({"a": (10, 20, 30, 40), "b": (5, 15, 50, 25)})
    assert bounds == (5, 15, 50, 40)


def test_filter_stage1_box_rejects_small_detections():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    assert (
        filter_stage1_box(
            frame.shape,
            (10, 10, 20, 20),
            lot_mask=None,
            min_box_area=150,
            filter_mode="bounds",
        )
        is None
    )


def test_filter_stage1_box_rejects_boxes_outside_lot_mask():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    lot_mask = (50, 50, 250, 180)
    assert (
        filter_stage1_box(
            frame.shape,
            (0, 60, 80, 120),
            lot_mask=lot_mask,
            roi_boxes=None,
            min_box_area=100,
            filter_mode="bounds",
        )
        is None
    )


def test_filter_stage1_box_rejects_box_between_slots_even_inside_lot_bounds():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    roi_boxes = [(50, 50, 90, 140), (110, 50, 150, 140)]
    kept = filter_stage1_box(
        frame.shape,
        (92, 60, 108, 130),
        lot_mask=(50, 50, 150, 140),
        roi_boxes=roi_boxes,
        min_box_area=100,
        filter_mode="roi_center",
    )
    assert kept is None


def test_filter_stage1_box_bounds_mode_keeps_box_between_slots():
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    kept = filter_stage1_box(
        frame.shape,
        (92, 60, 108, 130),
        lot_mask=(50, 50, 150, 140),
        roi_boxes=[(50, 50, 90, 140), (110, 50, 150, 140)],
        min_box_area=100,
        filter_mode="bounds",
    )
    assert kept == (92, 60, 108, 130)


def test_box_area_computes_pixel_area():
    assert box_area((10, 20, 40, 70)) == 1500


def test_get_spot_boxes_returns_fixed_rois():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    fixed_rois = {
        "spot_1": (100, 100, 160, 180),
        "spot_2": (170, 100, 230, 180),
    }

    boxes = get_spot_boxes(
        frame=frame,
        fixed_rois=fixed_rois,
        stage1_model=None,
        device="cpu",
        use_stage1_detector=False,
    )

    assert boxes == fixed_rois


def test_get_spot_boxes_filters_stage1_detections():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    fixed_rois = {
        "spot_1": (100, 100, 160, 180),
        "spot_2": (170, 100, 230, 180),
    }
    stage1 = FakeDetectYOLO(
        [
            (105, 105, 155, 175),
            (0, 0, 20, 20),
            (162, 105, 168, 175),
            (250, 220, 360, 290),
        ]
    )

    boxes = get_spot_boxes(
        frame=frame,
        fixed_rois=fixed_rois,
        stage1_model=stage1,
        device="cpu",
        use_stage1_detector=True,
        use_sahi=False,
        min_box_area=1500,
        filter_mode="bounds",
    )

    assert boxes == {"spot_1": (105, 105, 155, 175)}


def test_post_enabled_by_default(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["detect.py", "--image", "samples/demo.jpg"])
    args = parse_args()
    resolved = resolve_settings(args, {})
    assert resolved.post is True
    assert resolved.stage1_filter_mode == "bounds"
    assert resolved.latest_frame_path == str(DEFAULT_LATEST_FRAME_PATH)


def test_no_post_flag_disables_backend_updates(monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["detect.py", "--image", "samples/demo.jpg", "--no-post"]
    )
    args = parse_args()
    resolved = resolve_settings(args, {})
    assert resolved.post is False


def test_run_pipeline_uses_shared_postprocess_path():
    frame = np.ones((60, 60, 3), dtype=np.uint8)
    args = type(
        "Args",
        (),
        {
            "device": "cpu",
            "stage1_detector": False,
            "stage1_sahi": True,
            "stage2_threshold": 0.5,
            "stage1_imgsz": 1280,
            "stage1_slice_size": 640,
            "stage1_overlap": 0.2,
            "stage1_min_box_area": 1500,
            "stage1_filter_mode": "bounds",
            "stage1_postprocess_type": "nmm",
            "stage1_match_threshold": 0.3,
        },
    )()

    payload, spot_geometries = run_pipeline(
        frame=frame,
        fixed_geometries={
            "spot_1": np.array([[0, 0], [20, 0], [20, 20], [0, 20]], dtype=np.float32),
            "spot_2": np.array(
                [[20, 0], [40, 0], [40, 20], [20, 20]], dtype=np.float32
            ),
        },
        stage1_model=None,
        stage2_model=FakeYOLO([("occupied", 0.9), ("free", 0.7)]),
        smooth_buf=SmoothingBuffer(window=1),
        args=args,
    )

    assert geometry_to_box(spot_geometries["spot_1"]) == (0, 0, 20, 20)
    assert payload["spots"] == {"spot_1": "occupied", "spot_2": "free"}
    assert payload["confidence"] == {"spot_1": 0.9, "spot_2": 0.7}


def test_dump_warp_samples_writes_side_by_side_images(tmp_path):
    frame = np.full((40, 40, 3), 200, dtype=np.uint8)
    dumped = dump_warp_samples(
        frame,
        {"spot_1": np.array([[5, 5], [20, 4], [22, 20], [4, 22]], dtype=np.float32)},
        tmp_path,
        source_name="demo.jpg",
        limit=3,
    )

    files = list(tmp_path.glob("*.jpg"))
    assert dumped == 1
    assert len(files) == 1


def test_maybe_dump_warp_samples_returns_false_when_nothing_written(tmp_path):
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    dumped = maybe_dump_warp_samples(
        frame,
        {"spot_1": np.array([[0, 0], [0, 0], [0, 0], [0, 0]], dtype=np.float32)},
        tmp_path,
        source_name="demo.jpg",
        limit=3,
    )

    assert dumped is False
    assert list(tmp_path.glob("*.jpg")) == []


def test_write_latest_frame_writes_jpeg_atomically(tmp_path):
    output_path = tmp_path / "latest.jpg"
    frame = np.zeros((16, 16, 3), dtype=np.uint8)

    assert write_latest_frame(frame, output_path) is True
    assert output_path.exists()
    assert output_path.with_suffix(".jpg.tmp").exists() is False


def test_write_latest_frame_can_downscale_for_stream(tmp_path):
    output_path = tmp_path / "latest.jpg"
    frame = np.zeros((40, 120, 3), dtype=np.uint8)

    assert write_latest_frame(frame, output_path, jpeg_quality=60, max_width=60) is True
    written = cv2.imread(str(output_path))

    assert written is not None
    assert written.shape[1] == 60
