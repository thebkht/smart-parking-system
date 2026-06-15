// Owner-setup label-correction contract: derive a display label for a spot and
// apply an edited label back into a normalized layout (used by SpotLabelsEditor
// and mirrored by the LeafletMap tooltip / backend PATCH /spots/{id}).

export function spotLabel(spot) {
  if (!spot) return "";
  if (spot.label) return spot.label;
  return spot.spot_id ? spot.spot_id.replace("spot_", "P") : "";
}

export function applyLabelEdit(layout, spotId, label) {
  if (!layout || !Array.isArray(layout.spots)) return layout;
  return {
    ...layout,
    spots: layout.spots.map((s) =>
      s.spot_id === spotId ? { ...s, label } : s,
    ),
  };
}
