#!/usr/bin/env python3
"""End-to-end smoke test for the full PRD path.

Exercises the canonical product flow in-process (no running server needed)
against the real FastAPI app and the real SIFT localizer:

    owner setup (POST /map)
      -> map persistence (GET /map)
      -> edge updates (POST /update)
      -> live map (GET /status)
      -> park (POST /park, real photo + real localization)
      -> find (GET /find/{session_id})

Uses an isolated in-memory database so it never touches parking.db. Prints a
PASS/FAIL line per stage and exits non-zero on the first failure.

Usage:
    python scripts/smoke_test.py
    make smoke-test
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402

import backend.main as backend_module  # noqa: E402
from backend.main import app  # noqa: E402

QUERY_PHOTO = REPO_ROOT / "samples" / "photo_2026-04-23 21.29.16.jpeg"

LAYOUT = {
    "spots": [
        {
            "spot_id": "spot_1",
            "points": [[0, 0], [10, 0], [10, 10], [0, 10]],
            "label": "A1",
        },
        {
            "spot_id": "spot_2",
            "points": [[20, 0], [30, 0], [30, 10], [20, 10]],
            "label": "A2",
        },
        {
            "spot_id": "spot_3",
            "points": [[40, 0], [50, 0], [50, 10], [40, 10]],
            "label": "A3",
        },
    ],
    "image_width": 100,
    "image_height": 80,
}

OCCUPANCY = {
    "spots": {"spot_1": "occupied", "spot_2": "free", "spot_3": "occupied"},
    "confidence": {"spot_1": 0.91, "spot_2": 0.22, "spot_3": 0.88},
    "timestamp": "2026-04-21T00:00:00Z",
}


class SmokeFailure(RuntimeError):
    pass


def _check(label: str, condition: bool, detail: str = "") -> None:
    mark = "PASS" if condition else "FAIL"
    line = f"[{mark}] {label}"
    if detail:
        line += f" — {detail}"
    print(line)
    if not condition:
        raise SmokeFailure(label)


def run() -> None:
    # Isolated in-memory DB so the smoke test never mutates parking.db.
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    backend_module.init_db(conn)
    backend_module._conn = conn
    client = TestClient(app)

    # 1. Owner setup: publish a quadrilateral layout.
    r = client.post("/map", json=LAYOUT)
    _check("owner setup: POST /map", r.status_code == 200, f"status={r.status_code}")
    _check(
        "owner setup: spots saved",
        r.json().get("spots_saved") == len(LAYOUT["spots"]),
        f"saved={r.json().get('spots_saved')}",
    )

    # 2. Map persistence: read the layout back.
    r = client.get("/map")
    _check("map persistence: GET /map", r.status_code == 200, f"status={r.status_code}")
    spot_ids = {s["spot_id"] for s in r.json().get("spots", [])}
    _check(
        "map persistence: spot_ids round-trip",
        spot_ids == {"spot_1", "spot_2", "spot_3"},
        f"got={sorted(spot_ids)}",
    )

    # 3. Edge updates: post an occupancy payload.
    r = client.post("/update", json=OCCUPANCY)
    _check("edge update: POST /update", r.status_code == 200, f"status={r.status_code}")

    # 4. Live map: read current occupancy.
    r = client.get("/status")
    _check("live map: GET /status", r.status_code == 200, f"status={r.status_code}")
    status_spots = r.json().get("spots", {})
    _check(
        "live map: status reflects update",
        status_spots == OCCUPANCY["spots"],
        f"got={status_spots}",
    )

    # 5. Park: upload a real driver photo, run real SIFT localization.
    _check("park: query photo present", QUERY_PHOTO.exists(), str(QUERY_PHOTO))
    with QUERY_PHOTO.open("rb") as fh:
        r = client.post("/park", files={"photo": (QUERY_PHOTO.name, fh, "image/jpeg")})
    _check("park: POST /park", r.status_code == 200, f"status={r.status_code}")
    park = r.json()
    session_id = park.get("session_id")
    _check(
        "park: session created", isinstance(session_id, int), f"session_id={session_id}"
    )
    _check(
        "park: localized to a known spot",
        park.get("spot_id") in spot_ids,
        f"spot_id={park.get('spot_id')} score={park.get('similarity_score')}",
    )

    # 6. Find: resolve the session back to a spot + corners.
    r = client.get(f"/find/{session_id}")
    _check("find: GET /find/{id}", r.status_code == 200, f"status={r.status_code}")
    body = r.json()
    _check(
        "find: returns matching spot_id",
        body.get("spot_id") == park.get("spot_id"),
        f"spot_id={body.get('spot_id')}",
    )
    corners = body.get("corners")
    _check(
        "find: returns 4 corners",
        isinstance(corners, list) and len(corners) == 4,
        f"corners={corners}",
    )

    # 7. Find: unknown session is a clean 404.
    r = client.get("/find/999999")
    _check(
        "find: unknown session -> 404", r.status_code == 404, f"status={r.status_code}"
    )

    conn.close()


def main() -> int:
    print("Running end-to-end PRD smoke test (in-process, in-memory DB)...\n")
    try:
        run()
    except SmokeFailure as exc:
        print(f"\nSMOKE TEST FAILED at: {exc}")
        return 1
    print("\nSMOKE TEST PASSED — full PRD path works end to end.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
