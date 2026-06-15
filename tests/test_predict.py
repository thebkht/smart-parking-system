import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import predict


class FakeTensor:
    def __init__(self, values):
        self._values = values

    def cpu(self):
        return self

    def tolist(self):
        return list(self._values)


class FakeProbs:
    def __init__(self):
        self.top1 = 1
        self.top1conf = 0.81234
        self.data = [0.18766, 0.81234]


def test_prediction_mode_defaults_to_stage2():
    args = SimpleNamespace(stage1=False, stage2=False, single_model=False)
    assert predict.prediction_mode(args) == "stage2"


def test_prediction_mode_rejects_multiple_modes():
    args = SimpleNamespace(stage1=True, stage2=True, single_model=False)
    try:
        predict.prediction_mode(args)
    except SystemExit as exc:
        assert "Choose only one" in str(exc)
    else:
        raise AssertionError("expected prediction_mode to fail")


def test_default_imgsz_uses_stage_defaults():
    assert predict.default_imgsz("stage2", None) == predict.DEFAULT_STAGE2_IMGSZ
    assert predict.default_imgsz("stage1", None) == predict.DEFAULT_DETECT_IMGSZ
    assert predict.default_imgsz("single_model", 512) == 512


def test_classification_output_normalizes_probabilities():
    result = SimpleNamespace(
        names={0: "free", 1: "occupied"},
        probs=FakeProbs(),
    )

    payload = predict.classification_output(
        result, source="sample.jpg", weights="best.pt"
    )

    assert payload["task"] == "classify"
    assert payload["label"] == "occupied"
    assert payload["confidence"] == 0.8123
    assert payload["probabilities"] == {"free": 0.1877, "occupied": 0.8123}


def test_detection_output_normalizes_boxes():
    result = SimpleNamespace(
        names={0: "parking_spot"},
        boxes=SimpleNamespace(
            xyxy=FakeTensor([[10.123, 20.456, 30.789, 40.012]]),
            conf=FakeTensor([0.92345]),
            cls=FakeTensor([0.0]),
        ),
    )

    payload = predict.detection_output(
        result,
        source="frame.jpg",
        weights="best.pt",
        mode="stage1",
    )

    assert payload["task"] == "detect"
    assert payload["track"] == "stage1"
    assert payload["count"] == 1
    assert payload["boxes"] == [
        {
            "class_id": 0,
            "label": "parking_spot",
            "confidence": 0.9234,
            "xyxy": [10.12, 20.46, 30.79, 40.01],
        }
    ]
