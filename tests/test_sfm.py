import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "ml"))

import sfm_layout


def _write_image(path, seed):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, size=(64, 64, 3)).astype("uint8")
    cv2.imwrite(str(path), img)


def test_generate_layout_writes_outputs(tmp_path):
    pytest.importorskip("cv2")
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    paths = []
    for i in range(2):
        p = img_dir / f"frame_{i}.png"
        _write_image(p, i)
        paths.append(p)

    out_dir = tmp_path / "out"
    layout = sfm_layout.generate_layout(paths, out_dir)

    assert (out_dir / "bev_map.png").exists()
    assert (out_dir / "layout.json").exists()
    assert layout["spots"]
    assert layout["canvas"]["width"] > 0
    assert layout["source_images"] == ["frame_0.png", "frame_1.png"]


def test_generate_layout_uses_acpds_annotations(tmp_path, monkeypatch):
    """Recognized ACPDS uploads yield a real ground-truth layout, not the grid."""
    pytest.importorskip("cv2")
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    frame = img_dir / "GOPRTEST.JPG"
    _write_image(frame, 7)  # 64x64

    ann = {
        "train": {
            "file_names": ["GOPRTEST.JPG"],
            "rois_list": [
                [
                    [[0.1, 0.1], [0.3, 0.1], [0.3, 0.4], [0.1, 0.4]],
                    [[0.5, 0.5], [0.7, 0.5], [0.7, 0.8], [0.5, 0.8]],
                ]
            ],
            "occupancy_list": [[True, False]],
        }
    }
    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(json.dumps(ann), encoding="utf-8")
    monkeypatch.setattr(sfm_layout, "ACPDS_ANNOTATIONS", ann_path)

    layout = sfm_layout.generate_layout([frame], tmp_path / "out")

    assert layout["spot_source"] == "acpds_annotations"
    assert len(layout["spots"]) == 2
    assert layout["canvas"] == {"width": 64, "height": 64}
    # Normalized corners are scaled to pixel space: 0.1 * 64 = 6.4.
    assert layout["spots"][0]["corners"][0] == [6.4, 6.4]
    assert (tmp_path / "out" / "bev_map.png").exists()


def test_generate_layout_falls_back_when_unrecognized(tmp_path, monkeypatch):
    """Images not present in the ACPDS annotations keep the placeholder grid."""
    pytest.importorskip("cv2")
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    frame = img_dir / "not_acpds.png"
    _write_image(frame, 3)

    ann_path = tmp_path / "annotations.json"
    ann_path.write_text(
        json.dumps({"train": {"file_names": [], "rois_list": [], "occupancy_list": []}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(sfm_layout, "ACPDS_ANNOTATIONS", ann_path)

    layout = sfm_layout.generate_layout([frame], tmp_path / "out")
    assert layout["spot_source"] == "placeholder_grid"


def test_generate_layout_no_images_raises(tmp_path):
    with pytest.raises(sfm_layout.SfMError):
        sfm_layout.generate_layout([], tmp_path / "out")


def test_generate_layout_unreadable_image_raises(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(sfm_layout.SfMError):
        sfm_layout.generate_layout([bad], tmp_path / "out")
