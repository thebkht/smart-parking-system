// Find My Car contract: POST /park -> session_id; GET /find/{id} -> spot_id,
// confidence (only kept when finite and in [0,1]), similarity_score.
import { describe, it, expect } from "vitest";
import { parseParkResponse, parseFindResponse } from "../findMyCar";

describe("parseParkResponse (POST /park contract)", () => {
  it("extracts session_id and localization state", () => {
    expect(
      parseParkResponse({ session_id: 7, spot_id: "spot_3", similarity_score: 13.1, localized: true }),
    ).toEqual({ sessionId: 7, spotId: "spot_3", localized: true });
  });

  it("infers localized=false when no spot matched", () => {
    expect(parseParkResponse({ session_id: 9, spot_id: null }).localized).toBe(false);
  });

  it("is null-safe", () => {
    expect(parseParkResponse(null)).toEqual({ sessionId: null });
  });
});

describe("parseFindResponse (GET /find/{id} contract)", () => {
  it("returns spot_id and a valid confidence/similarity", () => {
    expect(
      parseFindResponse({ spot_id: "spot_2", confidence: 0.84, similarity_score: 14.0 }),
    ).toEqual({ spotId: "spot_2", confidence: 0.84, similarityScore: 14.0 });
  });

  it("rejects out-of-range or non-numeric confidence", () => {
    expect(parseFindResponse({ spot_id: "s", confidence: 42 }).confidence).toBeNull();
    expect(parseFindResponse({ spot_id: "s", confidence: null }).confidence).toBeNull();
    expect(parseFindResponse({ spot_id: "s" }).confidence).toBeNull();
  });

  it("keeps similarity score only when finite", () => {
    expect(parseFindResponse({ spot_id: "s", similarity_score: 861 }).similarityScore).toBe(861);
    expect(parseFindResponse({ spot_id: "s", similarity_score: null }).similarityScore).toBeNull();
  });

  it("is null-safe", () => {
    expect(parseFindResponse(undefined)).toEqual({
      spotId: null,
      confidence: null,
      similarityScore: null,
    });
  });
});
