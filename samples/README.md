# Sample Images

Demo assets selected from the **ACPDS** dataset (`datasets/acpds/`). The whole
demo is anchored on **one physical lot** — a large plaza-style lot with painted,
numbered bays — so the three product flows tell one coherent story: an owner
sets up *this* lot, the live map shows *this* lot over time, and a user finds
their car in *this* lot.

The lot appears in six ACPDS frames across two sessions:

- daytime: `GOPR6541`, `GOPR6542`, `GOPR6543`
- night / wet: `GOPR0089`, `GOPR0090`, `GOPR0093`

Geometry/occupancy come from `datasets/acpds/annotations.json`.

> Viewpoint caveat: ACPDS has no fixed-camera video — the action camera pans
> between shots, so these frames share the lot but not a pixel-locked
> viewpoint. The layout polygons from Owner Setup won't line up frame-to-frame
> in Live Occupancy; treat the sequence as the same lot over time, not a single
> static camera feed.

> `localization_refs/` is **not** part of this selection — it is the tracked
> Find-My-Car evaluation workspace wired into `backend/main.py`
> (`FALLBACK_REFERENCE_PATH`) and the test suite, and is left untouched.

## owner_setup/

Six **daytime** overlapping frames of the lot from different angles — the web
Owner Setup SfM pipeline needs ≥4 overlapping photos to reconstruct a
bird's-eye layout. Kept daytime-only so feature matching isn't broken by the
day/night lighting gap:

- `GOPR6536.JPG`, `GOPR6537.JPG`, `GOPR6538.JPG` — fuller occupancy, varied angles
- `GOPR6541.JPG`, `GOPR6542.JPG`, `GOPR6543.JPG` — wider/emptier layout views

## live_occupancy/

The same lot as an ordered day→night sequence (`frame_01..06`), names keep
provenance:

| frame | source | session | occupied |
|-------|--------|---------|----------|
| 01 | GOPR6541 | day | 3/83 |
| 02 | GOPR6542 | day | 6/95 |
| 03 | GOPR6543 | day | 2/76 |
| 04 | GOPR0089 | night | 3/35 |
| 05 | GOPR0090 | night | 5/61 |
| 06 | GOPR0093 | night | 2/42 |

## find_my_car/

Per-spot close-ups cropped from the daytime frames (annotation quad + padding),
simulating a user's "where did I park" photo — each is a single clearly
recognizable parked car:

- `my_car_NN_<frame>_spotNN.jpg` — `NN` is display order; `<frame>` and `spotNN`
  give the ACPDS source frame and spot index.
