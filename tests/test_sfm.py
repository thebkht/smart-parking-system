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


def test_generate_layout_no_images_raises(tmp_path):
    with pytest.raises(sfm_layout.SfMError):
        sfm_layout.generate_layout([], tmp_path / "out")


def test_generate_layout_unreadable_image_raises(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    with pytest.raises(sfm_layout.SfMError):
        sfm_layout.generate_layout([bad], tmp_path / "out")
