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
Accepts a driver photo, runs SIFT feature matching against stored reference images in `samples/localization_refs/`, inserts a row into `park_sessions`, and returns a `session_id` for later lookup.

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
Service health check.

```json
{ "status": "ok" }
```

#### `GET /stream`
Serves `logs/latest_frame.jpg` as a multipart MJPEG stream for browser or VLC clients.

```html
<img src="http://127.0.0.1:8000/stream" alt="Parking stream">
```

---

## Database Schema

SQLite database at `parking.db` with four tables:

| Table | Purpose |
|---|---|
| `log` | Full occupancy payload snapshots from edge |
| `layout` | Parking lot layout snapshots (quad polygons) |
| `spot_references` | Per-spot polygon coordinates |
| `park_sessions` | Per-spot occupancy history + Find My Car sessions |

---

## Bandwidth

The edge node POSTs compact JSON (~280 bytes) every 2 seconds instead of streaming raw video.

| Method | Bandwidth |
|---|---|
| JSON POST (this system) | 0.3 KB/s |
| H.264 720p stream | 200 KB/s |
| H.264 1080p stream | 500 KB/s |

**Result: 99.9% bandwidth reduction vs raw video streaming.**
