// Live Occupancy contract: parseStatus reads `response.spots`, and
// countOccupancy scopes counts to the published layout's spot IDs so the
// free/occupied cards can never exceed the total (the bug that showed
// "55 free + 127 occupied vs 12 total").
import { describe, it, expect } from "vitest";
import { parseStatus, countOccupancy } from "../occupancy";

describe("parseStatus (GET /status contract)", () => {
  it("extracts the spots map from { spots, confidence, timestamp }", () => {
    const data = {
      spots: { spot_1: "free", spot_2: "occupied" },
      confidence: { spot_1: 0.9 },
      timestamp: "2026-04-21T00:00:00Z",
    };
    expect(parseStatus(data)).toEqual({ spot_1: "free", spot_2: "occupied" });
  });

  it("tolerates a bare spot map (backward compatibility)", () => {
    expect(parseStatus({ spot_1: "free" })).toEqual({ spot_1: "free" });
  });

  it("returns null for invalid payloads", () => {
    expect(parseStatus(null)).toBeNull();
    expect(parseStatus(undefined)).toBeNull();
  });
});

describe("countOccupancy (counter scoping)", () => {
  const layoutSpots = [{ spot_id: "spot_1" }, { spot_id: "spot_2" }, { spot_id: "spot_3" }];

  it("counts only spots present in the layout", () => {
    const status = { spot_1: "free", spot_2: "occupied", spot_3: "free" };
    expect(countOccupancy(layoutSpots, status)).toEqual({ free: 2, occupied: 1, total: 3 });
  });

  it("ignores stale spot IDs not in the layout", () => {
    const status = {
      spot_1: "occupied",
      stale_a: "occupied",
      stale_b: "free",
      stale_c: "occupied",
    };
    const { free, occupied, total } = countOccupancy(layoutSpots, status);
    expect(total).toBe(3);
    expect(free + occupied).toBeLessThanOrEqual(total);
    expect(occupied).toBe(1);
  });

  it("treats missing/unknown spot states as neither free nor occupied", () => {
    const status = { spot_1: "free" }; // spot_2, spot_3 unknown
    expect(countOccupancy(layoutSpots, status)).toEqual({ free: 1, occupied: 0, total: 3 });
  });

  it("returns zeroes with a correct total when status is null", () => {
    expect(countOccupancy(layoutSpots, null)).toEqual({ free: 0, occupied: 0, total: 3 });
  });
});
