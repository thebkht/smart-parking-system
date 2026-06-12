import sqlite3
import sys
import asyncio
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

import backend.main as backend_module
from backend.main import app


@pytest.fixture(autouse=True)
def fresh_db(monkeypatch):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    backend_module.init_db(conn)
    monkeypatch.setattr(backend_module, "_conn", conn)
    yield
    conn.close()


def sample_payload():
    return {
        "spots": {"spot_1": "occupied", "spot_2": "free"},
        "confidence": {"spot_1": 0.91, "spot_2": 0.22},
        "timestamp": "2026-04-21T00:00:00Z",
    }


def sample_layout():
    return {
        "spots": [
            {
                "spot_id": "spot_1",
                "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
                "label": "A1",
            },
            {
                "spot_id": "spot_2",
                "corners": [[20, 0], [30, 0], [30, 10], [20, 10]],
                "label": "A2",
            },
        ],
        "image_width": 100,
        "image_height": 80,
    }


def stored_layout_spots():
    return [
        {
            "spot_id": "spot_1",
            "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
            "corners": None,
            "label": "A1",
        },
        {
            "spot_id": "spot_2",
            "points": None,
            "corners": [[20.0, 0.0], [30.0, 0.0], [30.0, 10.0], [20.0, 10.0]],
            "label": "A2",
        },
    ]


def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_update_accepts_v3_payload():
    client = TestClient(app)
    response = client.post("/update", json=sample_payload())
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_update_persists_payload_as_is():
    client = TestClient(app)
    payload = sample_payload()
    client.post("/update", json=payload)
    row = backend_module.get_conn().execute(
        "SELECT payload FROM log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    assert row is not None
    assert backend_module.json.loads(row[0]) == payload


def test_status_returns_latest_payload_without_legacy_fields():
    client = TestClient(app)
    client.post("/update", json=sample_payload())
    latest = sample_payload()
    latest["spots"]["spot_1"] = "free"
    client.post("/update", json=latest)

    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == latest


def test_status_empty_returns_empty_object():
    client = TestClient(app)
    response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {}


def test_history_wraps_payloads_with_recorded_at():
    client = TestClient(app)
    client.post("/update", json=sample_payload())
    response = client.get("/history")
    assert response.status_code == 200
    body = response.json()
    assert list(body.keys()) == ["items"]
    assert body["items"][0]["payload"] == sample_payload()
    assert body["items"][0]["recorded_at"].endswith("Z")


def test_invalid_legacy_payload_is_rejected():
    client = TestClient(app)
    response = client.post("/update", json={"spot_1": "occupied", "fps": 10.0})
    assert response.status_code == 422


def test_generate_mjpeg_stream_wraps_latest_frame(tmp_path):
    frame_path = tmp_path / "latest.jpg"
    frame_bytes = b"fake-jpeg"
    frame_path.write_bytes(frame_bytes)

    async def consume():
        stream = backend_module.generate_mjpeg_stream(frame_path=frame_path, poll_interval=0.0)
        return await asyncio.wait_for(stream.__anext__(), timeout=1)

    chunk = asyncio.run(consume())

    assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
    assert chunk.endswith(frame_bytes + b"\r\n")


def test_save_map_rejects_non_quad_points():
    client = TestClient(app)
    response = client.post(
        "/map",
        json={
            "spots": [
                {
                    "spot_id": "spot_1",
                    "points": [[0, 0], [10, 0], [10, 10]],
                }
            ]
        },
    )

    assert response.status_code == 422
    assert "exactly 4 [x, y] points" in response.json()["detail"]


def test_save_map_rejects_non_pair_points():
    client = TestClient(app)
    response = client.post(
        "/map",
        json={
            "spots": [
                {
                    "spot_id": "spot_1",
                    "corners": [[0, 0], [10, 0], [10], [0, 10]],
                }
            ]
        },
    )

    assert response.status_code == 422
    assert "entries must be [x, y] pairs" in response.json()["detail"]


def test_save_map_persists_layout_and_spot_references():
    client = TestClient(app)
    response = client.post("/map", json=sample_layout())

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "spots_saved": 2}

    layout_response = client.get("/map")
    assert layout_response.status_code == 200
    body = layout_response.json()
    assert body["spots"] == stored_layout_spots()
    assert body["image_width"] == 100
    assert body["image_height"] == 80
    assert body["updated_at"].endswith("Z")

    rows = backend_module.get_conn().execute(
        "SELECT spot_id, label, points FROM spot_references ORDER BY spot_id"
    ).fetchall()
    assert rows == [
        ("spot_1", "A1", "[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]"),
        ("spot_2", "A2", "[[20.0, 0.0], [30.0, 0.0], [30.0, 10.0], [20.0, 10.0]]"),
    ]


def test_save_map_round_trips_layout_metadata():
    client = TestClient(app)
    layout = sample_layout()
    layout["canvas"] = {"width": 2560, "height": 1440}
    layout["background_image"] = "bev_map.png"
    layout["spot_source"] = "placeholder_grid"
    layout["source_images"] = ["lot-1.jpg", "lot-2.jpg"]

    response = client.post("/map", json=layout)
    assert response.status_code == 200

    body = client.get("/map").json()
    assert body["canvas"] == {"width": 2560, "height": 1440}
    assert body["background_image"] == "bev_map.png"
    assert body["spot_source"] == "placeholder_grid"
    assert body["source_images"] == ["lot-1.jpg", "lot-2.jpg"]


def test_map_background_serves_png(monkeypatch, tmp_path):
    bev_path = tmp_path / "bev_map.png"
    bev_bytes = b"\x89PNG\r\n\x1a\nfake"
    bev_path.write_bytes(bev_bytes)
    monkeypatch.setattr(backend_module, "BEV_BACKGROUND_PATH", bev_path)

    client = TestClient(app)
    response = client.get("/map/background")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content == bev_bytes


def test_map_background_missing_returns_404(monkeypatch, tmp_path):
    monkeypatch.setattr(
        backend_module, "BEV_BACKGROUND_PATH", tmp_path / "missing.png"
    )
    client = TestClient(app)
    response = client.get("/map/background")
    assert response.status_code == 404


def test_layout_aliases_return_latest_layout():
    client = TestClient(app)

    save_response = client.post("/layout", json=sample_layout())
    assert save_response.status_code == 200
    assert save_response.json() == {"status": "ok", "spots_saved": 2}

    get_response = client.get("/layout")
    assert get_response.status_code == 200
    body = get_response.json()
    assert body["spots"] == stored_layout_spots()
    assert body["updated_at"].endswith("Z")


def test_multipart_layout_alias_returns_stored_layout():
    client = TestClient(app)
    client.post("/map", json=sample_layout())

    response = client.post(
        "/layout",
        files=[
            ("images", ("lot-1.jpg", b"fake-image-1", "image/jpeg")),
            ("images", ("lot-2.jpg", b"fake-image-2", "image/jpeg")),
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["spots"] == stored_layout_spots()
    assert body["updated_at"].endswith("Z")


def test_sessions_returns_recent_updates_and_filter():
    client = TestClient(app)
    client.post("/update", json=sample_payload())

    response = client.get("/sessions")
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert body["sessions"][0]["spot_id"] in {"spot_1", "spot_2"}
    assert body["sessions"][0]["recorded_at"].endswith("Z")

    filtered = client.get("/sessions", params={"spot_id": "spot_1"})
    assert filtered.status_code == 200
    assert filtered.json()["count"] == 1
    assert filtered.json()["sessions"][0]["spot_id"] == "spot_1"


def test_park_and_find_happy_path(monkeypatch):
    fake_localize = types.ModuleType("localize")

    def localize_query(*args, **kwargs):
        return {"spot_id": "spot_1", "score": 14.031, "elapsed_ms": 312.5}

    fake_localize.localize_query = localize_query
    monkeypatch.setitem(sys.modules, "localize", fake_localize)

    client = TestClient(app)
    client.post("/map", json=sample_layout())

    park_response = client.post(
        "/park",
        files={"photo": ("driver.jpg", b"fake-driver-image", "image/jpeg")},
    )
    assert park_response.status_code == 200
    park_body = park_response.json()
    assert park_body["session_id"] == 1
    assert park_body["spot_id"] == "spot_1"
    assert park_body["similarity_score"] == 14.031
    assert park_body["elapsed_ms"] == 312.5
    assert park_body["localized"] is True

    find_response = client.get(f"/find/{park_body['session_id']}")
    assert find_response.status_code == 200
    assert find_response.json() == {
        "session_id": 1,
        "spot_id": "spot_1",
        "corners": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
        "similarity_score": 14.031,
        "recorded_at": find_response.json()["recorded_at"],
    }
    assert find_response.json()["recorded_at"].endswith("Z")
