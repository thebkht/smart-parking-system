# Backend Layer

The backend is a FastAPI persistence and coordination layer for the Smart Parking System edge pipeline. It stores occupancy updates, serves the parking lot layout to edge nodes at startup, and supports the Find My Car feature.

## Running the Backend

```bash
source .venv/bin/activate
make backend
```

API docs available at: `http://127.0.0.1:8000/docs`

`make backend` binds uvicorn to `0.0.0.0:8000` so Expo/mobile devices on the
same Wi-Fi can reach the API through the host machine's LAN IP. To run local-only, use:

```bash
make backend BACKEND_HOST=127.0.0.1
```

---

## Endpoints

### Occupancy

#### `POST /update`
Receives an occupancy payload from the edge node and persists it to the `log` table. Also writes a row per spot into `park_sessions`.

**Request body:**
```json
{
  "spots": {
    "spot_1": "free",
    "spot_2": "occupied"
  },
  "confidence": {
    "spot_1": 0.91,
    "spot_2": 0.84
  },
  "timestamp": "2026-04-21T00:00:00Z"
}
```

**Response:**
```json
{ "status": "ok" }
```

---

#### `GET /status`
Returns the latest occupancy snapshot. Frontend should read `response.spots` for occupancy data.

**Response shape (final, confirmed):**
```json
{
  "spots": {
    "spot_1": "free",
    "spot_2": "occupied"
  },
  "confidence": {
    "spot_1": 0.91,
    "spot_2": 0.84
  },
  "timestamp": "2026-04-21T00:00:00Z"
}
```

> **Frontend note (@mirzayv):** Read occupancy as `response.spots`, confidence as `response.confidence`. This shape is final and will not change.

---

#### `GET /history?limit=100`
Returns recent occupancy snapshots in reverse chronological order.

**Response:**
```json
{
  "items": [
    {
      "payload": {
        "spots": { "spot_1": "free" },
        "confidence": { "spot_1": 0.91 },
        "timestamp": "2026-04-21T00:00:00Z"
      },
      "recorded_at": "2026-04-21T00:00:01Z"
    }
  ]
}
```

---

### Parking Lot Layout

> **Canonical route is `/map`**. The `/layout` route is an alias kept for frontend compatibility — both routes accept the same payload and return the same response.

#### `POST /map` (alias: `POST /layout`)
Saves the parking lot layout. Accepts spot polygons using either `points` or `corners` field — both are supported.

**Request body:**
```json
{
  "spots": [
    {
      "spot_id": "spot_1",
      "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
      "label": "A1"
    }
  ],
  "image_width": 1920,
  "image_height": 1080,
  "canvas": {"width": 2560, "height": 1440},
  "background_image": "bev_map.png",
  "spot_source": "placeholder_grid",
  "source_images": ["img_001.jpg"]
}
```

> **Note:** `corners` is accepted as an alias for `points` (used by ML pipeline output). `canvas`, `background_image`, `spot_source`, and `source_images` are optional metadata fields that are round-tripped verbatim by `GET /map`.

**Response:**
```json
{ "status": "ok", "spots_saved": 15 }
```

#### `POST /layout` with image uploads (owner setup → server-side SfM)

When `POST /layout` receives `multipart/form-data` with one or more `images`
files, the backend runs the SfM pipeline (`ml/sfm_layout.generate_layout`)
in-process: it builds a bird's-eye-view canvas, extracts spot polygons, persists
the layout (same as `POST /map`), writes the BEV to `artifacts/layout_sample/bev_map.png`
(served by `GET /map/background`), and returns the stored layout.

**Request:** `multipart/form-data` with one or more `images` fields.

**Fallback:** if SfM cannot produce a usable layout (too few/unreadable photos),
the route responds `422`. The app then falls back to **manual polygon
submission** via `POST /map` with a precomputed `LayoutPayload` (JSON body) — this
JSON path is the documented owner-setup fallback.

---

#### `GET /map` (alias: `GET /layout`)
Returns the latest parking lot layout. Called by `edge/detect.py` at startup to load spot polygons, and by both apps on launch to render the lot map.

**Response:**
```json
{
  "spots": [
    {
      "spot_id": "spot_1",
      "points": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
      "label": "A1"
    }
  ],
  "image_width": 1920,
  "image_height": 1080,
  "canvas": {"width": 2560, "height": 1440},
  "background_image": "bev_map.png",
  "spot_source": "placeholder_grid",
  "source_images": ["img_001.jpg"],
  "updated_at": "2026-04-21T00:00:00Z"
}
```

> Returns `404` if no layout has been posted yet.

---

#### `GET /map/background`
Serves `artifacts/layout_sample/bev_map.png` as a PNG image for use as the map background in both apps.

> Returns `404` if the file is not present.

---

### Spot Correction & References

#### `PATCH /spots/{spot_id}`
Owner correction step: rename a spot's label after layout generation. Updates
`spot_references` and rewrites the latest `layout` row so `GET /map` reflects the
new label. Geometry is left unchanged.

**Request body:** `{ "label": "VIP-1" }`

**Response:** `{ "spot_id": "spot_1", "label": "VIP-1" }`

> Returns `404` if the spot is unknown.

#### `POST /spots/{spot_id}/references`
Stores one or more reference photos for a spot under
`artifacts/spot_references/<spot_id>/` and records them in `spot_reference_images`.
These per-spot references are what Find My Car (`POST /park`) matches against.

**Request:** `multipart/form-data` with one or more `photos` fields.

**Response:** `{ "spot_id": "spot_1", "saved": 2, "paths": [...] }`

#### `GET /spots/{spot_id}/references`
Lists stored reference photos for a spot:
`{ "spot_id": "spot_1", "references": [{"path": ..., "created_at": ...}], "count": 2 }`

---

### Parking Sessions

#### `GET /sessions?spot_id=spot_1&limit=100`
Returns recent parking session records. Optionally filter by `spot_id`.

**Response:**
```json
{
  "sessions": [
    {
      "spot_id": "spot_1",
      "status": "occupied",
      "confidence": 0.91,
      "recorded_at": "2026-04-21T00:00:00Z"
    }
  ],
  "count": 1
}
```

---

### Find My Car (mobile only)

#### `POST /park`
Accepts a driver photo, runs SIFT feature matching against the owner-managed
per-spot references in `artifacts/spot_references/` (falling back to the bundled
`samples/localization_refs/` when none have been uploaded), inserts a row into
`park_sessions`, and returns a `session_id` for later lookup.

Used by: **Mobile** — Find My Car screen.

**Request:** `multipart/form-data` with field `photo` (image file).

**Response:**
```json
{
  "session_id": 42,
  "spot_id": "spot_7",
  "similarity_score": 14.031,
  "elapsed_ms": 312.5,
  "localized": true
}
```

> `localized: false` if SIFT matching failed to find a confident match.

---

#### `GET /find/{session_id}`
Looks up a parking session by ID and returns the spot location with corner coordinates for map display.

Used by: **Mobile** — Find My Car screen.

**Response:**
```json
{
  "session_id": 42,
  "spot_id": "spot_7",
  "corners": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],
  "similarity_score": 14.031,
  "recorded_at": "2026-04-21T00:00:00Z"
}
```

> Returns `404` if session not found.

---

### Utilities

#### `GET /health`
Service health check (also reports whether auth is enforced).

```json
{ "status": "ok", "auth_enabled": false }
```

#### `GET /stream`
Serves `logs/latest_frame.jpg` as a multipart MJPEG stream for browser or VLC clients.

```html
<img src="http://127.0.0.1:8000/stream" alt="Parking stream">
```

---

### Authentication (optional)

Auth is **opt-in** so the demo runs token-free by default. Set `AUTH_ENABLED=1`
when starting the backend to require bearer tokens on owner-mutating routes
(`POST /map`, `POST /layout`, `PATCH /spots/{id}`, `POST /spots/{id}/references`)
and to scope Find My Car sessions to their owner.

#### `POST /auth/register`
Issues a bearer token for an owner (lightweight, no password).

**Request body:** `{ "username": "owner" }`
**Response:** `{ "user_id": 1, "username": "owner", "token": "..." }`

Send the token as `Authorization: Bearer <token>` on protected routes. With auth
on, `POST /park` stamps the session's owner and `GET /find/{session_id}` returns
`404` for sessions owned by a different user. Read-only routes (`GET /status`,
`GET /map`) stay public.

---

## Database Schema

SQLite database at `parking.db`:

| Table | Purpose |
|---|---|
| `log` | Full occupancy payload snapshots from edge |
| `layout` | Parking lot layout snapshots (quad polygons) |
| `spot_references` | Per-spot polygon coordinates + label |
| `spot_reference_images` | Per-spot Find My Car reference photo paths |
| `park_sessions` | Per-spot occupancy history + Find My Car sessions (with optional `user_id`) |
| `users` | Owner accounts + bearer tokens (when auth is enabled) |

---

## Bandwidth

The edge node POSTs compact JSON (~280 bytes) every 2 seconds instead of streaming raw video.

| Method | Bandwidth |
|---|---|
| JSON POST (this system) | 0.3 KB/s |
| H.264 720p stream | 200 KB/s |
| H.264 1080p stream | 500 KB/s |

**Result: 99.9% bandwidth reduction vs raw video streaming.**
