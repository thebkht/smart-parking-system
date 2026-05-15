import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from ml import localize


class FakeMatcher:
    def knnMatch(self, query_descriptors, reference_descriptors, k=2):
        count = int(reference_descriptors[0][0])
        pairs = []
        for index in range(count):
            pairs.append(
                [
                    SimpleNamespace(queryIdx=index, trainIdx=index, distance=0.1),
                    SimpleNamespace(queryIdx=index, trainIdx=index, distance=0.3),
                ]
            )
        return pairs


def write_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color=(120, 120, 120)).save(path)


def test_load_reference_manifest_supports_mapping(tmp_path):
    manifest = tmp_path / "refs.json"
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "a.jpg").write_bytes(b"jpg")
    manifest.write_text(json.dumps({"spot_1": "refs/a.jpg"}), encoding="utf-8")

    refs = localize.load_reference_manifest(manifest)

    assert refs[0].spot_id == "spot_1"
    assert refs[0].image_path.name == "a.jpg"


def test_discover_reference_directory_uses_subdirectories(tmp_path):
    write_image(tmp_path / "spot_7" / "a.jpg")

    refs = localize.discover_reference_directory(tmp_path)

    assert refs == [localize.ReferenceImage(spot_id="spot_7", image_path=(tmp_path / "spot_7" / "a.jpg").resolve())]


def test_evaluate_reference_rejects_homography_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(localize, "estimate_inliers", lambda *args, **kwargs: (False, 2))
    query_features = localize.ImageFeatures(
        image_path=tmp_path / "query.jpg",
        keypoints=[SimpleNamespace(pt=(float(i), float(i))) for i in range(8)],
        descriptors=np.ones((8, 1), dtype=np.float32),
    )
    reference_features = localize.ImageFeatures(
        image_path=tmp_path / "ref.jpg",
        keypoints=[SimpleNamespace(pt=(float(i), float(i))) for i in range(8)],
        descriptors=np.full((8, 1), 8, dtype=np.float32),
    )

    result = localize.evaluate_reference(
        FakeMatcher(),
        query_features,
        reference_features,
        ratio_threshold=0.75,
        min_matches=4,
        min_inliers=4,
        ransac_threshold=5.0,
    )

    assert result["passed"] is False
    assert result["failure_reason"] == "homography_rejected"


def test_main_writes_localization_json_output(tmp_path, monkeypatch):
    query = tmp_path / "query.jpg"
    ref_a = tmp_path / "refs" / "spot_1" / "a.jpg"
    ref_b = tmp_path / "refs" / "spot_2" / "b.jpg"
    output = tmp_path / "out.json"
    write_image(query)
    write_image(ref_a)
    write_image(ref_b)

    def fake_extract(_detector, image_path: Path):
        base_name = Path(image_path).name
        if base_name == "query.jpg":
            return localize.ImageFeatures(
                image_path=image_path,
                keypoints=[SimpleNamespace(pt=(float(i), float(i))) for i in range(12)],
                descriptors=np.ones((12, 1), dtype=np.float32),
            )
        match_count = 10 if "spot_1" in str(image_path) else 5
        return localize.ImageFeatures(
            image_path=image_path,
            keypoints=[SimpleNamespace(pt=(float(i), float(i))) for i in range(match_count)],
            descriptors=np.full((match_count, 1), match_count, dtype=np.float32),
        )

    monkeypatch.setattr(localize, "create_sift_detector", lambda: object())
    monkeypatch.setattr(localize, "build_matcher", lambda: FakeMatcher())
    monkeypatch.setattr(localize, "extract_image_features", fake_extract)
    monkeypatch.setattr(localize, "estimate_inliers", lambda *_args, **_kwargs: (True, 9))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "localize.py",
            "--query",
            str(query),
            "--references",
            str(tmp_path / "refs"),
            "--output",
            str(output),
            "--min-matches",
            "8",
        ],
    )

    localize.main()

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["spot_id"] == "spot_1"
    assert payload["failure_reason"] is None
    assert payload["candidates"][0]["spot_id"] == "spot_1"
