import sqlite3
import sys
import asyncio
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
