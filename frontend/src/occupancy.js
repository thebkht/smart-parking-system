// ---------------------------------------------------------------------------
// Occupancy contract helpers — shared by the Live Occupancy map screen.
//
// The backend `GET /status` response is `{ spots, confidence, timestamp }`.
// `parseStatus` extracts the per-spot state map (tolerating a bare map for
// backward compatibility). `countOccupancy` scopes free/occupied counts to the
// published layout's spot IDs so the counters can never exceed the total — the
// `/status` payload may carry stale spot IDs from earlier edge runs.
// ---------------------------------------------------------------------------

export function parseStatus(data) {
  if (!data || typeof data !== "object") return null;
  if (data.spots && typeof data.spots === "object") return data.spots;
  return data;
}

export function countOccupancy(layoutSpots, statusMap) {
  const total = Array.isArray(layoutSpots) ? layoutSpots.length : 0;
  if (!statusMap || typeof statusMap !== "object") {
    return { free: 0, occupied: 0, total };
  }
  let free = 0;
  let occupied = 0;
  for (const spot of layoutSpots ?? []) {
    const state = statusMap[spot.spot_id];
    if (state === "free") free += 1;
    else if (state === "occupied") occupied += 1;
  }
  return { free, occupied, total };
}
