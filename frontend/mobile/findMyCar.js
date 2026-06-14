// ---------------------------------------------------------------------------
// Find My Car contract helpers — shared by FindMyCarScreen.
//
// `POST /park` returns `{ session_id, spot_id, similarity_score, ... }`.
// `GET /find/{session_id}` returns `{ spot_id, corners, similarity_score,
// confidence?, recorded_at }`. These helpers parse those responses defensively:
// confidence is only kept when it is a finite number in [0, 1]; similarity is
// kept when finite. Both fall back to null so the UI can show "unknown".
// ---------------------------------------------------------------------------

export function parseParkResponse(data) {
  if (!data || typeof data !== "object") return { sessionId: null };
  return {
    sessionId: data.session_id ?? null,
    spotId: data.spot_id ?? null,
    localized: Boolean(data.localized ?? data.spot_id != null),
  };
}

export function parseFindResponse(data) {
  if (!data || typeof data !== "object") {
    return { spotId: null, confidence: null, similarityScore: null };
  }
  const confidence = data.confidence == null ? NaN : Number(data.confidence);
  const similarity =
    data.similarity_score == null ? NaN : Number(data.similarity_score);
  return {
    spotId: data.spot_id ?? null,
    confidence:
      Number.isFinite(confidence) && confidence >= 0 && confidence <= 1
        ? confidence
        : null,
    similarityScore: Number.isFinite(similarity) ? similarity : null,
  };
}
