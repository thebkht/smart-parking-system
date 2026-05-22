from __future__ import annotations

import json
import sqlite3
import shutil
import tempfile
from pathlib import Path as FilePath
from asyncio import sleep
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()
DB = Path("parking.db")
LATEST_FRAME_PATH = Path("logs/latest_frame.jpg")
_conn: sqlite3.Connection | None = None


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ParkingUpdate(BaseModel):
    spots: Dict[str, Literal["occupied", "free"]] = Field(default_factory=dict)
    confidence: Dict[str, float] = Field(default_factory=dict)
    timestamp: str


class HistoryItem(BaseModel):
    payload: ParkingUpdate
    recorded_at: str


class HistoryResponse(BaseModel):
    items: List[HistoryItem]


class SpotPolygon(BaseModel):
    spot_id: str
    points: Optional[List[List[float]]] = None   # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    corners: Optional[List[List[float]]] = None  # alias used by ML pipeline
    label: Optional[str] = None

    def get_points(self) -> List[List[float]]:
        """Return points regardless of whether corners or points was used."""
        if self.points is not None:
            return validate_quad_points(self.points, self.spot_id)
        if self.corners is not None:
            return validate_quad_points(self.corners, self.spot_id)
        raise ValueError(f"Spot {self.spot_id!r} has neither points nor corners.")


class LayoutPayload(BaseModel):
    spots: List[SpotPolygon]
    image_width: Optional[int] = None
    image_height: Optional[int] = None


class ParkSession(BaseModel):
    spot_id: str
    status: Literal["occupied", "free"]
    timestamp: str
    confidence: Optional[float] = None


def validate_quad_points(raw_points: List[List[float]], spot_id: str) -> List[List[float]]:
    if not isinstance(raw_points, list) or len(raw_points) != 4:
        raise ValueError(f"Spot {spot_id!r} must contain exactly 4 [x, y] points.")

    normalized: List[List[float]] = []
    for point in raw_points:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"Spot {spot_id!r} entries must be [x, y] pairs.")
        x, y = point
        normalized.append([float(x), float(y)])
    return normalized


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def init_db(conn: sqlite3.Connection | None = None) -> sqlite3.Connection:
    global _conn
    if conn is None:
        conn = sqlite3.connect(DB, check_same_thread=False)

    # Original table
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            payload   TEXT    NOT NULL,
            recorded  TEXT    NOT NULL
        )
        """
    )

    # NEW: parking lot layout (quad polygons)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS layout (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            data         TEXT    NOT NULL,
            updated_at   TEXT    NOT NULL
        )
        """
    )

    # NEW: spot reference metadata
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS spot_references (
            spot_id      TEXT    PRIMARY KEY,
            label        TEXT,
            points       TEXT    NOT NULL,
            created_at   TEXT    NOT NULL
        )
        """
    )

    # NEW: parking sessions (car arrives / leaves)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS park_sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            spot_id      TEXT    NOT NULL,
            status       TEXT    NOT NULL,
            confidence   REAL,
            recorded_at  TEXT    NOT NULL
        )
        """
    )

    conn.commit()
    _conn = conn
    return conn


def get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = init_db()
    return _conn


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# MJPEG stream helper
# ---------------------------------------------------------------------------

async def generate_mjpeg_stream(
    frame_path: Path = LATEST_FRAME_PATH,
    poll_interval: float = 0.1,
    resend_interval: float = 0.25,
):
    last_signature: tuple[int, int] | None = None
    last_frame_bytes: bytes | None = None
    last_sent_at = 0.0
    while True:
        try:
            stat = frame_path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if signature != last_signature:
                frame_bytes = frame_path.read_bytes()
                if frame_bytes:
                    last_signature = signature
                    last_frame_bytes = frame_bytes
                    last_sent_at = 0.0
            if last_frame_bytes:
                now = datetime.now(timezone.utc).timestamp()
                if last_sent_at == 0.0 or now - last_sent_at >= resend_interval:
                    last_sent_at = now
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n"
                        + last_frame_bytes
                        + b"\r\n"
                    )
        except FileNotFoundError:
            pass
        await sleep(poll_interval)


init_db()


# ---------------------------------------------------------------------------
# Endpoints — original 5
# ---------------------------------------------------------------------------

@app.post("/update")
async def update(payload: ParkingUpdate) -> dict:
    conn = get_conn()
    # Log the update
    conn.execute(
        "INSERT INTO log (payload, recorded) VALUES (?, ?)",
        (payload.model_dump_json(), utc_now_iso()),
    )
    # Also write a park_session row for each spot
    now = utc_now_iso()
    for spot_id, status in payload.spots.items():
        confidence = payload.confidence.get(spot_id)
        conn.execute(
            "INSERT INTO park_sessions (spot_id, status, confidence, recorded_at) VALUES (?, ?, ?, ?)",
            (spot_id, status, confidence, now),
        )
    conn.commit()
    return {"status": "ok"}


@app.get("/status")
async def status() -> dict:
    conn = get_conn()
    row = conn.execute(
        "SELECT payload FROM log ORDER BY id DESC LIMIT 1"
    ).fetchone()
    return json.loads(row[0]) if row else {}


@app.get("/history")
async def history(limit: int = 100) -> HistoryResponse:
    conn = get_conn()
    rows = conn.execute(
        "SELECT payload, recorded FROM log ORDER BY id DESC LIMIT ?",
        (limit,),
    ).fetchall()
    items = [
        HistoryItem(
            payload=ParkingUpdate(**json.loads(payload)),
            recorded_at=recorded,
        )
        for payload, recorded in rows
    ]
    return HistoryResponse(items=items)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff",
            "Access-Control-Allow-Origin": "*",
        },
    )


# ---------------------------------------------------------------------------
# NEW Endpoints — /map and /sessions
# ---------------------------------------------------------------------------

@app.post("/map")
async def save_map(layout: LayoutPayload) -> dict:
    """Save the parking lot layout (quad polygons for each spot)."""
    # Validate ALL spots first before touching the database
    for spot in layout.spots:
        try:
            spot.get_points()
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    conn = get_conn()
    now = utc_now_iso()

    conn.execute(
        "INSERT INTO layout (data, updated_at) VALUES (?, ?)",
        (layout.model_dump_json(), now),
    )
    for spot in layout.spots:
        conn.execute(
            """
            INSERT INTO spot_references (spot_id, label, points, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(spot_id) DO UPDATE SET
                label      = excluded.label,
                points     = excluded.points,
                created_at = excluded.created_at
            """,
            (spot.spot_id, spot.label, json.dumps(spot.get_points()), now),
        )
    conn.commit()
    return {"status": "ok", "spots_saved": len(layout.spots)}

@app.get("/map")
async def get_map() -> dict:
    """Return the latest parking lot layout with quad polygons."""
    conn = get_conn()
    row = conn.execute(
        "SELECT data, updated_at FROM layout ORDER BY id DESC LIMIT 1"
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail="No layout found. POST /map first.")

    data = json.loads(row[0])
    data["updated_at"] = row[1]
    return data


@app.get("/sessions")
async def get_sessions(spot_id: Optional[str] = None, limit: int = 100) -> dict:
    """Return recent parking sessions, optionally filtered by spot_id."""
    conn = get_conn()
    if spot_id:
        rows = conn.execute(
            "SELECT spot_id, status, confidence, recorded_at FROM park_sessions WHERE spot_id=? ORDER BY id DESC LIMIT ?",
            (spot_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT spot_id, status, confidence, recorded_at FROM park_sessions ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()

    sessions = [
        {"spot_id": r[0], "status": r[1], "confidence": r[2], "recorded_at": r[3]}
        for r in rows
    ]
    return {"sessions": sessions, "count": len(sessions)}

# ---------------------------------------------------------------------------
# Find My Car endpoints
# ---------------------------------------------------------------------------

from fastapi import UploadFile, File

@app.post("/park")
async def park_car(photo: UploadFile = File(...)) -> dict:
    """Accept a driver photo, run SIFT localization, insert park_session row."""
    import sys
    sys.path.insert(0, str(Path("ml")))
    from localize import localize_query

    # Save uploaded photo to a temp file
    suffix = Path(photo.filename).suffix if photo.filename else ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        contents = await photo.read()
        tmp_path.write_bytes(contents)

    try:
        refs_path = Path("samples/localization_refs")
        if not refs_path.exists():
            raise HTTPException(status_code=503, detail="No reference images found.")

        result = localize_query(
            tmp_path,
            refs_path,
            ratio_threshold=0.75,
            min_matches=8,
            min_inliers=6,
            ransac_threshold=5.0,
            top_k=3,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    spot_id = result.get("spot_id")
    score = float(result.get("score", 0.0))

    conn = get_conn()
    now = utc_now_iso()
    cursor = conn.execute(
        """
        INSERT INTO park_sessions (spot_id, status, confidence, recorded_at)
        VALUES (?, 'occupied', ?, ?)
        """,
        (spot_id, score, now),
    )
    conn.commit()
    session_id = cursor.lastrowid

    return {
        "session_id": session_id,
        "spot_id": spot_id,
        "similarity_score": score,
        "elapsed_ms": result.get("elapsed_ms"),
        "localized": spot_id is not None,
    }


@app.get("/find/{session_id}")
async def find_car(session_id: int) -> dict:
    """Look up a park session and return spot_id + corner coordinates."""
    conn = get_conn()

    # Get session
    row = conn.execute(
        "SELECT spot_id, confidence, recorded_at FROM park_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    spot_id, confidence, recorded_at = row

    # Get corners from spot_references
    spot_row = conn.execute(
        "SELECT points FROM spot_references WHERE spot_id = ?",
        (spot_id,),
    ).fetchone()

    corners = json.loads(spot_row[0]) if spot_row else None

    return {
        "session_id": session_id,
        "spot_id": spot_id,
        "corners": corners,
        "similarity_score": confidence,
        "recorded_at": recorded_at,
    }


# Canonical name is /map — add /layout as alias for frontend compatibility
@app.post("/layout")
async def save_layout_alias(layout: LayoutPayload) -> dict:
    """Alias for POST /map — kept for frontend compatibility."""
    return await save_map(layout)


  
