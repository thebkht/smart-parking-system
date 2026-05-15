import { useEffect, useRef } from "react";
import PropTypes from "prop-types";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

// ---------------------------------------------------------------------------
// PropTypes (mirrors App.jsx shapes)
// ---------------------------------------------------------------------------
const spotShape = PropTypes.shape({
  spot_id: PropTypes.string.isRequired,
  corners: PropTypes.arrayOf(PropTypes.arrayOf(PropTypes.number)).isRequired,
});

const layoutShape = PropTypes.shape({
  canvas: PropTypes.shape({
    width: PropTypes.number,
    height: PropTypes.number,
  }),
  background_image: PropTypes.string,
  spots: PropTypes.arrayOf(spotShape),
});

// ---------------------------------------------------------------------------
// Color helpers — same palette as the old BEVMap SVG component
// ---------------------------------------------------------------------------
function spotStyle(occ, isHighlit, isDimmed) {
  let fillColor = "#e7e5e4";
  let color = "#d4d0cb";

  if (occ === "free") {
    fillColor = "#1a7a4a";
    color = "#1a7a4a";
  } else if (occ === "occupied") {
    fillColor = "#c0392b";
    color = "#c0392b";
  }

  if (isHighlit) {
    fillColor = "#e89600";
    color = "#e89600";
  }

  return {
    fillColor,
    color,
    weight: isHighlit ? 3 : 1.5,
    fillOpacity: isDimmed ? 0.05 : 0.13,
    opacity: isDimmed ? 0.25 : 1,
  };
}

// ---------------------------------------------------------------------------
// LeafletMap
// Drop-in replacement for BEVMap. Accepts identical props so App.jsx needs
// zero changes — just swap <BEVMap … /> → <LeafletMap … />.
// ---------------------------------------------------------------------------
export default function LeafletMap({
  layout,
  status,
  highlightSpot,
  onSpotClick,
  dimUnhighlighted,
}) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  // spot_id → L.polygon instance
  const polygonsRef = useRef({});
  // spot_id → L.tooltip instance
  const tooltipsRef = useRef({});

  // -------------------------------------------------------------------------
  // 1. Init map once layout arrives
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!layout || !containerRef.current) return;

    const { width, height } = layout.canvas;

    // Destroy previous instance if layout changes
    if (mapRef.current) {
      mapRef.current.remove();
      mapRef.current = null;
      polygonsRef.current = {};
      tooltipsRef.current = {};
    }

    // CRS.Simple maps pixel coords directly.
    // Leaflet uses [lat, lng] = [y, x] in image space.
    const map = L.map(containerRef.current, {
      crs: L.CRS.Simple,
      zoomSnap: 0.1,
      zoomControl: false,
      attributionControl: false,
      doubleClickZoom: false,
      scrollWheelZoom: false,
      dragging: false,
    });
    mapRef.current = map;

    // Bounds: Leaflet CRS.Simple has y=0 at bottom-left.
    // We flip: image (0,0) top-left → Leaflet [height, 0].
    const bounds = [
      [0, 0],
      [height, width],
    ];

    map.fitBounds(bounds, { padding: [0, 0] });

    // Optional background image overlay
    if (layout.background_image) {
      L.imageOverlay(layout.background_image, bounds).addTo(map);
    } else {
      // Subtle parking-lot tarmac fill as fallback
      map.getContainer().style.background = "#f5f5f4";
    }

    // Draw all spot polygons
    layout.spots.forEach((spot) => {
      // Convert pixel [x, y] → Leaflet [flipped-y, x]
      const latlngs = spot.corners.map(([x, y]) => [height - y, x]);

      const poly = L.polygon(latlngs, {
        ...spotStyle(
          status?.[spot.spot_id],
          highlightSpot === spot.spot_id,
          dimUnhighlighted && highlightSpot && highlightSpot !== spot.spot_id,
        ),
        // smooth rendering
        smoothFactor: 1,
      }).addTo(map);

      // Label tooltip (permanent, no interaction)
      const label = spot.spot_id.replace("spot_", "P");
      const tooltip = L.tooltip({
        permanent: true,
        direction: "center",
        className: "leaflet-spot-label",
        interactive: false,
      })
        .setContent(label)
        .setLatLng(poly.getBounds().getCenter());

      poly.bindTooltip(tooltip);

      // Click handler
      if (onSpotClick) {
        poly.on("click", () => onSpotClick(spot.spot_id));
        poly.on("mouseover", () => {
          poly.setStyle({ weight: 2.5, fillOpacity: 0.25 });
        });
        poly.on("mouseout", () => {
          // re-derive style so hover doesn't override status color
          const isDimmed =
            dimUnhighlighted &&
            !!highlightSpot &&
            highlightSpot !== spot.spot_id;
          poly.setStyle(
            spotStyle(
              status?.[spot.spot_id],
              highlightSpot === spot.spot_id,
              isDimmed,
            ),
          );
        });
      }

      polygonsRef.current[spot.spot_id] = poly;
      tooltipsRef.current[spot.spot_id] = tooltip;
    });

    // Cleanup on unmount
    return () => {
      map.remove();
      mapRef.current = null;
      polygonsRef.current = {};
      tooltipsRef.current = {};
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [layout]);

  // -------------------------------------------------------------------------
  // 2. Update polygon styles when status / highlight changes (no re-init)
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (!layout) return;
    layout.spots.forEach((spot) => {
      const poly = polygonsRef.current[spot.spot_id];
      if (!poly) return;
      const isDimmed =
        dimUnhighlighted && !!highlightSpot && highlightSpot !== spot.spot_id;
      poly.setStyle(
        spotStyle(
          status?.[spot.spot_id],
          highlightSpot === spot.spot_id,
          isDimmed,
        ),
      );
    });
  }, [status, highlightSpot, dimUnhighlighted, layout]);

  // -------------------------------------------------------------------------
  // 3. Render
  // -------------------------------------------------------------------------
  if (!layout) return null;

  const { width, height } = layout.canvas;
  // Match the SVG's aspect ratio so the container sizes itself correctly
  const paddingBottom = `${(height / width) * 100}%`;

  return (
    <>
      {/* Inject label styles once — Leaflet tooltip resets strip everything */}
      <style>{`
        .leaflet-spot-label {
          background: transparent !important;
          border: none !important;
          box-shadow: none !important;
          padding: 0 !important;
          font-family: monospace;
          font-size: 11px;
          font-weight: 600;
          color: #78716c;
          pointer-events: none;
          white-space: nowrap;
        }
        .leaflet-spot-label::before {
          display: none !important;
        }
      `}</style>

      {/* Aspect-ratio wrapper — mirrors the SVG's width="100%" behaviour */}
      <div
        className="block rounded-xl border border-stone-200 overflow-hidden"
        style={{ position: "relative", width: "100%", paddingBottom }}
        aria-label="Parking lot map"
      >
        <div
          ref={containerRef}
          style={{
            position: "absolute",
            inset: 0,
            background: "#f5f5f4",
          }}
        />
      </div>
    </>
  );
}

LeafletMap.propTypes = {
  layout: layoutShape,
  status: PropTypes.objectOf(PropTypes.string),
  highlightSpot: PropTypes.string,
  onSpotClick: PropTypes.func,
  dimUnhighlighted: PropTypes.bool,
};
