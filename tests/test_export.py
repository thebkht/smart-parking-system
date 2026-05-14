import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import export


class FakeYOLO:
    def __init__(self, weights: str):
        self.weights = weights

    def export(self, *, format: str, imgsz: int, **_kwargs):
        out = Path(self.weights).parent / f"tmp_{format}_{imgsz}"
        if format == "onnx":
            path = out.with_suffix(".onnx")
            path.write_bytes(b"onnx")
            return str(path)
        package = out.with_suffix(".mlpackage")
        package.mkdir(parents=True, exist_ok=True)
        (package / "Manifest.json").write_text("{}", encoding="utf-8")
        return str(package)


def test_export_writes_summary_json(tmp_path, monkeypatch):
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"checkpoint")
    output_dir = tmp_path / "artifacts"
    summary_json = tmp_path / "summary.json"

    monkeypatch.setattr(export, "YOLO", FakeYOLO)

    class FakeQuantType:
        QInt8 = "QInt8"

    def fake_quantize_dynamic(src: str, dst: str, weight_type: str):
        assert weight_type == FakeQuantType.QInt8
        Path(dst).write_bytes(Path(src).read_bytes() + b"_int8")

    module = type(
        "FakeQuantModule",
        (),
        {"QuantType": FakeQuantType, "quantize_dynamic": staticmethod(fake_quantize_dynamic)},
    )
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export.py",
            "--weights",
            str(weights),
            "--output-dir",
            str(output_dir),
            "--imgsz",
            "128",
            "--summary-json",
            str(summary_json),
        ],
    )

    export.main()

    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["artifacts"]["best.pt"]["present"] is True
    assert payload["artifacts"]["best.onnx"]["present"] is True
    assert payload["artifacts"]["best_int8.onnx"]["present"] is True
