// Owner Setup + map-render contract: normalizeLayout turns backend layout
// payloads (GET /map / POST /layout response) into clean spot polygons.
import { describe, it, expect } from "vitest";
import { normalizeLayout } from "../layout";

describe("normalizeLayout (Owner Setup / map contract)", () => {
  it("returns null for non-layout payloads", () => {
    expect(normalizeLayout(null)).toBeNull();
    expect(normalizeLayout({})).toBeNull();
    expect(normalizeLayout({ spots: "nope" })).toBeNull();
  });

  it("accepts both `points` and `corners` per spot", () => {
    const out = normalizeLayout({
      spots: [
        { spot_id: "spot_1", points: [[0, 0], [10, 0], [10, 10], [0, 10]], label: "A1" },
        { spot_id: "spot_2", corners: [[20, 0], [30, 0], [30, 10], [20, 10]] },
      ],
      image_width: 100,
      image_height: 80,
    });
    expect(out.spots).toHaveLength(2);
    expect(out.spots[0]).toEqual({
      spot_id: "spot_1",
      label: "A1",
      corners: [[0, 0], [10, 0], [10, 10], [0, 10]],
    });
    expect(out.spots[1].corners[0]).toEqual([20, 0]);
    expect(out.canvas).toEqual({ width: 100, height: 80 });
  });

  it("drops spots without exactly four corners", () => {
    const out = normalizeLayout({
      spots: [
        { spot_id: "ok", points: [[0, 0], [1, 0], [1, 1], [0, 1]] },
        { spot_id: "bad", points: [[0, 0], [1, 1]] },
      ],
    });
    expect(out.spots.map((s) => s.spot_id)).toEqual(["ok"]);
  });

  it("derives canvas size from corner extents when dimensions are absent", () => {
    const out = normalizeLayout({
      spots: [{ spot_id: "s", points: [[0, 0], [200, 0], [200, 100], [0, 100]] }],
    });
    expect(out.canvas.width).toBeGreaterThanOrEqual(200);
    expect(out.canvas.height).toBeGreaterThanOrEqual(100);
  });

  it("only resolves the background image when an apiBase is given", () => {
    const payload = { spots: [], background_image: "bev.png", image_width: 10, image_height: 10 };
    expect(normalizeLayout(payload).background_image).toBeNull();
    expect(normalizeLayout(payload, { apiBase: "http://x" }).background_image).toBe(
      "http://x/map/background",
    );
  });
});
