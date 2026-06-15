from __future__ import annotations

import json
import os
import secrets
import sqlite3
import shutil
import tempfile
from pathlib import Path as FilePath
from asyncio import sleep
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request, UploadFile, File
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

# Auth is opt-in so the class demo keeps working without tokens. Set
# AUTH_ENABLED=1 to require bearer tokens on owner-mutating routes and to scope
# Find My Car sessions to their owner.
AUTH_ENABLED = os.getenv("AUTH_ENABLED", "").strip().lower() in {"1", "true", "yes"}

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print("VALIDATION ERROR:", exc.errors())
    errors = []
    for error in exc.errors():
        clean = {}
        for k, v in error.items():
            try:
                json.dumps(v)
                clean[k] = v
            except (TypeError, UnicodeDecodeError):
                clean[k] = "<non-serializable>"
        errors.append(clean)
    return JSONResponse(status_code=422, content={"detail": errors})


DB = Path("parking.db")
LATEST_FRAME_PATH = Path("logs/latest_frame.jpg")
BEV_BACKGROUND_PATH = Path("artifacts/layout_sample/bev_map.png")
# Layout output dir shares the parent of the served BEV background so that a
# freshly generated map is reachable via GET /map/background.
LAYOUT_OUTPUT_DIR = BEV_BACKGROUND_PATH.parent
OWNER_UPLOAD_ROOT = Path("artifacts/owner_uploads")
SPOT_REFERENCE_ROOT = Path("artifacts/spot_references")
_conn: sqlite3.Connection | None = None


FALLBACK_REFERENCE_PATH = Path("samples/localization_refs")


def _run_sfm(paths: List[Path], output_dir: Path) -> dict:
    """Run the SfM/BEV pipeline in-process. Isolated for easy test monkeypatching."""
    import sys

    sys.path.insert(0, str(Path("ml")))
    from sfm_layout import generate_layout

    return generate_layout(paths, output_dir)


def _has_reference_images(root: Path) -> bool:
    if not root.exists():
        return False
    for sub in root.iterdir():
        if sub.is_dir() and any(
            child.suffix.lower() in {".jpg", ".jpeg", ".png"} for child in sub.iterdir()
        ):
            return True
    return False


def resolve_references_path() -> Path | None:
    """Prefer owner-managed per-spot references; fall back to the bundled samples."""
    if _has_reference_images(SPOT_REFERENCE_ROOT):
        return SPOT_REFERENCE_ROOT
    if FALLBACK_REFERENCE_PATH.exists():
        return FALLBACK_REFERENCE_PATH
    return None


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
    # Layout metadata produced by the SfM pipeline (round-tripped so the apps
    # can scale the map to the real canvas and fetch the BEV background).
    canvas: Optional[Dict[str, int]] = None
    background_image: Optional[str] = None
    spot_source: Optional[str] = None
    source_images: Optional[List[str]] = None


class ParkSession(BaseModel):
    spot_id: str
    status: Literal["occupied", "free"]
    timestamp: str
    confidence: Optional[float] = None


class SpotLabelUpdate(BaseModel):
    label: str


class RegisterPayload(BaseModel):
    username: str


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

    # NEW: per-spot reference photos for Find My Car localization
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS spot_reference_images (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            spot_id      TEXT    NOT NULL,
            path         TEXT    NOT NULL,
            created_at   TEXT    NOT NULL
        )
        """
    )

    # NEW: users (lightweight bearer-token auth + session ownership)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            username     TEXT    NOT NULL UNIQUE,
            token        TEXT    NOT NULL UNIQUE,
            created_at   TEXT    NOT NULL
        )
        """
    )

    # Migration: attach session ownership to park_sessions if not present yet.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(park_sessions)").fetchall()}
    if "user_id" not in cols:
        conn.execute("ALTER TABLE park_sessions ADD COLUMN user_id INTEGER")

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
# Auth helpers (active only when AUTH_ENABLED)
# ---------------------------------------------------------------------------

def _token_from_header(authorization: Optional[str]) -> Optional[str]:
    if not authorization or not authorization.lower().startswith("bearer "):
        return None
    return authorization.split(" ", 1)[1].strip() or None


def _user_by_token(token: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT id, username FROM users WHERE token=?", (token,)
    ).fetchone()
    return {"id": row[0], "username": row[1]} if row else None


async def require_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Require a valid bearer token on owner-mutating routes when auth is on.

    When AUTH_ENABLED is false this is a no-op (returns None) so the demo and
    existing clients keep working without tokens.
    """
    if not AUTH_ENABLED:
        return None
    token = _token_from_header(authorization)
    if token is None:
        raise HTTPException(status_code=401, detail="Missing bearer token.")
    user = _user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token.")
    return user


async def optional_user(authorization: Optional[str] = Header(None)) -> Optional[dict]:
    """Resolve the caller if a valid token is present, but never reject."""
    token = _token_from_header(authorization)
    if token is None:
        return None
    return _user_by_token(token)


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
    return {"status": "ok", "auth_enabled": AUTH_ENABLED}


@app.post("/auth/register")
async def auth_register(payload: RegisterPayload) -> dict:
    """Issue a bearer token for an owner. Lightweight (no password) by design."""
    conn = get_conn()
    token = secrets.token_urlsafe(32)
    now = utc_now_iso()
    try:
        cursor = conn.execute(
            "INSERT INTO users (username, token, created_at) VALUES (?, ?, ?)",
            (payload.username, token, now),
        )
        conn.commit()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Username already registered.") from exc
    return {"user_id": cursor.lastrowid, "username": payload.username, "token": token}


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
async def save_map(
    layout: LayoutPayload, user: Optional[dict] = Depends(require_user)
) -> dict:
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


@app.get("/map/background")
async def get_map_background() -> FileResponse:
    """Serve the BEV background image referenced by the stored layout."""
    if not BEV_BACKGROUND_PATH.exists():
        raise HTTPException(status_code=404, detail="No background image available.")
    return FileResponse(BEV_BACKGROUND_PATH, media_type="image/png")


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
async def park_car(
    photo: UploadFile = File(...), user: Optional[dict] = Depends(optional_user)
) -> dict:
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
        refs_path = resolve_references_path()
        if refs_path is None:
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
        INSERT INTO park_sessions (spot_id, status, confidence, recorded_at, user_id)
        VALUES (?, 'occupied', ?, ?, ?)
        """,
        (spot_id, score, now, user["id"] if user else None),
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
async def find_car(
    session_id: int, user: Optional[dict] = Depends(optional_user)
) -> dict:
    """Look up a park session and return spot_id + corner coordinates."""
    conn = get_conn()

    # Get session
    row = conn.execute(
        "SELECT spot_id, confidence, recorded_at, user_id FROM park_sessions WHERE id = ?",
        (session_id,),
    ).fetchone()

    if not row:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

    spot_id, confidence, recorded_at, session_user_id = row

    # When auth is on, an owned session is only visible to its owner.
    if AUTH_ENABLED and session_user_id is not None:
        if user is None or user["id"] != session_user_id:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found.")

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


# ---------------------------------------------------------------------------
# Find My Car — per-spot reference photo management
# ---------------------------------------------------------------------------

@app.patch("/spots/{spot_id}")
async def update_spot_label(
    spot_id: str,
    payload: SpotLabelUpdate,
    user: Optional[dict] = Depends(require_user),
) -> dict:
    """Owner correction: rename a spot's label after layout generation.

    Updates ``spot_references`` and rewrites the latest ``layout`` row so the
    new label is reflected by ``GET /map``. Geometry is left unchanged.
    """
    conn = get_conn()
    now = utc_now_iso()

    row = conn.execute(
        "SELECT spot_id FROM spot_references WHERE spot_id=?", (spot_id,)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Spot {spot_id} not found.")

    conn.execute(
        "UPDATE spot_references SET label=? WHERE spot_id=?", (payload.label, spot_id)
    )

    layout_row = conn.execute(
        "SELECT id, data FROM layout ORDER BY id DESC LIMIT 1"
    ).fetchone()
    if layout_row:
        layout_id, data_json = layout_row
        data = json.loads(data_json)
        for spot in data.get("spots", []):
            if spot.get("spot_id") == spot_id:
                spot["label"] = payload.label
        conn.execute(
            "UPDATE layout SET data=?, updated_at=? WHERE id=?",
            (json.dumps(data), now, layout_id),
        )

    conn.commit()
    return {"spot_id": spot_id, "label": payload.label}


@app.post("/spots/{spot_id}/references")
async def add_spot_references(
    spot_id: str,
    photos: List[UploadFile] = File(...),
    user: Optional[dict] = Depends(require_user),
) -> dict:
    """Store one or more reference photos for a spot (used by Find My Car SIFT matching)."""
    spot_dir = SPOT_REFERENCE_ROOT / spot_id
    spot_dir.mkdir(parents=True, exist_ok=True)

    conn = get_conn()
    now = utc_now_iso()
    saved: List[str] = []
    for photo in photos:
        suffix = Path(photo.filename).suffix if photo.filename else ".jpg"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        dest = spot_dir / f"{stamp}{suffix}"
        dest.write_bytes(await photo.read())
        conn.execute(
            "INSERT INTO spot_reference_images (spot_id, path, created_at) VALUES (?, ?, ?)",
            (spot_id, str(dest), now),
        )
        saved.append(str(dest))
    conn.commit()
    return {"spot_id": spot_id, "saved": len(saved), "paths": saved}


@app.get("/spots/{spot_id}/references")
async def list_spot_references(spot_id: str) -> dict:
    """List stored reference photos for a spot."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT path, created_at FROM spot_reference_images WHERE spot_id=? ORDER BY id DESC",
        (spot_id,),
    ).fetchall()
    return {
        "spot_id": spot_id,
        "references": [{"path": r[0], "created_at": r[1]} for r in rows],
        "count": len(rows),
    }


# ---------------------------------------------------------------------------
# /layout alias — frontend compatibility
# ---------------------------------------------------------------------------
@app.post("/layout")
async def save_layout_alias(
    request: Request, user: Optional[dict] = Depends(require_user)
) -> dict:
    """Owner setup entry point.

    Multipart (image uploads): run SfM server-side to build the BEV map + spot
    polygons, persist the layout, and return it. If SfM cannot produce a usable
    layout, respond 422 so the app can fall back to manual polygon submission via
    ``POST /map``.

    JSON body: treated as a precomputed ``LayoutPayload`` (the manual fallback),
    forwarded to ``POST /map``.
    """
    content_type = request.headers.get("content-type", "")

    if "multipart/" in content_type:
        form = await request.form()
        uploads = [
            item
            for item in form.getlist("images")
            if hasattr(item, "filename") and item.filename
        ]
        if not uploads:
            raise HTTPException(status_code=422, detail="No images uploaded for SfM.")

        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        upload_dir = OWNER_UPLOAD_ROOT / stamp
        upload_dir.mkdir(parents=True, exist_ok=True)
        saved_paths: List[Path] = []
        for upload in uploads:
            dest = upload_dir / Path(upload.filename).name
            dest.write_bytes(await upload.read())
            saved_paths.append(dest)

        try:
            layout_dict = _run_sfm(saved_paths, LAYOUT_OUTPUT_DIR)
        except Exception as exc:  # SfMError or any CV failure → manual fallback
            raise HTTPException(
                status_code=422,
                detail=f"SfM failed — submit polygons manually via POST /map. ({exc})",
            ) from exc

        try:
            payload = LayoutPayload.model_validate(layout_dict)
        except Exception as exc:
            raise HTTPException(
                status_code=422,
                detail=f"SfM produced an invalid layout — submit polygons manually via POST /map. ({exc})",
            ) from exc

        await save_map(payload, user=user)
        return await get_map()

    try:
        body = await request.json()
        layout = LayoutPayload.model_validate(body)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid layout payload: {exc}") from exc

    return await save_map(layout, user=user)


@app.get("/layout")
async def get_layout_alias() -> dict:
    """Alias for GET /map."""
    return await get_map()
