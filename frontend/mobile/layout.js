// ---------------------------------------------------------------------------
// Layout normalizer — shared contract between backend payloads and map views.
//
// Backend layouts may carry `points` or `corners` per spot and may or may not
// include `canvas` dimensions. This normalizes everything to:
//   { canvas: {width, height}, background_image, spot_source, source_images,
//     spots: [{ spot_id, label, corners }] }
// so LeafletMap (web) and LotMap (mobile) always receive clean data.
// ---------------------------------------------------------------------------

export function normalizeLayout(data, { apiBase } = {}) {
  if (!data || !Array.isArray(data.spots)) return null;

  const spots = data.spots
    .map((spot) => {
      const corners = spot.corners ?? spot.points;
      if (!Array.isArray(corners) || corners.length !== 4) return null;
      return {
        spot_id: spot.spot_id,
        label: spot.label ?? null,
        corners: corners.map(([x, y]) => [Number(x), Number(y)]),
      };
    })
    .filter(Boolean);

  let width = data.canvas?.width ?? data.image_width ?? null;
  let height = data.canvas?.height ?? data.image_height ?? null;
  if (!width || !height) {
    // Derive from corner extents with a 5% margin so spots never clip.
    let maxX = 0;
    let maxY = 0;
    spots.forEach(({ corners }) =>
      corners.forEach(([x, y]) => {
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }),
    );
    width = width || Math.ceil(maxX * 1.05) || 600;
    height = height || Math.ceil(maxY * 1.05) || 400;
  }

  return {
    canvas: { width, height },
    // Resolve the BEV background to the backend endpoint that serves it.
    // Only set when an apiBase is provided (i.e. layout came from the backend).
    background_image:
      data.background_image && apiBase ? `${apiBase}/map/background` : null,
    spot_source: data.spot_source ?? null,
    source_images: data.source_images ?? [],
    spots,
  };
}
