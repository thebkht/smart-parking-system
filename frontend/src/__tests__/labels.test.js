// Owner-setup label-correction contract: spotLabel derives the display label
// and applyLabelEdit applies a PATCH /spots/{id} edit back into the layout.
import { describe, it, expect } from "vitest";
import { spotLabel, applyLabelEdit } from "../labels";

describe("spotLabel", () => {
  it("prefers an explicit label", () => {
    expect(spotLabel({ spot_id: "spot_3", label: "VIP" })).toBe("VIP");
  });

  it("derives P-style label from spot_id when unlabeled", () => {
    expect(spotLabel({ spot_id: "spot_3" })).toBe("P3");
  });

  it("is null-safe", () => {
    expect(spotLabel(null)).toBe("");
    expect(spotLabel({})).toBe("");
  });
});

describe("applyLabelEdit", () => {
  const layout = {
    canvas: { width: 10, height: 10 },
    spots: [
      { spot_id: "spot_1", label: "A1", corners: [] },
      { spot_id: "spot_2", label: "A2", corners: [] },
    ],
  };

  it("updates only the targeted spot's label", () => {
    const out = applyLabelEdit(layout, "spot_1", "VIP-1");
    expect(out.spots[0].label).toBe("VIP-1");
    expect(out.spots[1].label).toBe("A2");
  });

  it("does not mutate the input layout", () => {
    applyLabelEdit(layout, "spot_1", "Changed");
    expect(layout.spots[0].label).toBe("A1");
  });

  it("is null-safe", () => {
    expect(applyLabelEdit(null, "spot_1", "x")).toBeNull();
    expect(applyLabelEdit({ spots: "nope" }, "spot_1", "x")).toEqual({ spots: "nope" });
  });
});
