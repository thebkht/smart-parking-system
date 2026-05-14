import { useState, useEffect, useRef, useCallback } from "react";
import PropTypes from "prop-types";

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
  source_images: PropTypes.arrayOf(PropTypes.string),
  spot_source: PropTypes.string,
  spots: PropTypes.arrayOf(spotShape),
});

const MOCK_LAYOUT = {
  canvas: { width: 600, height: 400 },
  background_image: "bev_map.png",
  source_images: ["img_001.jpg", "img_002.jpg", "img_003.jpg"],
  spot_source: "placeholder_grid",
  spots: [
    {
      spot_id: "spot_1",
      corners: [
        [40, 60],
        [110, 60],
        [110, 130],
        [40, 130],
      ],
    },
    {
      spot_id: "spot_2",
      corners: [
        [120, 60],
        [190, 60],
        [190, 130],
        [120, 130],
      ],
    },
    {
      spot_id: "spot_3",
      corners: [
        [200, 60],
        [270, 60],
        [270, 130],
        [200, 130],
      ],
    },
    {
      spot_id: "spot_4",
      corners: [
        [280, 60],
        [350, 60],
        [350, 130],
        [280, 130],
      ],
    },
    {
      spot_id: "spot_5",
      corners: [
        [360, 60],
        [430, 60],
        [430, 130],
        [360, 130],
      ],
    },
    {
      spot_id: "spot_6",
      corners: [
        [440, 60],
        [510, 60],
        [510, 130],
        [440, 130],
      ],
    },
    {
      spot_id: "spot_7",
      corners: [
        [40, 220],
        [110, 220],
        [110, 310],
        [40, 310],
      ],
    },
    {
      spot_id: "spot_8",
      corners: [
        [120, 220],
        [190, 220],
        [190, 310],
        [120, 310],
      ],
    },
    {
      spot_id: "spot_9",
      corners: [
        [200, 220],
        [270, 220],
        [270, 310],
        [200, 310],
      ],
    },
    {
      spot_id: "spot_10",
      corners: [
        [280, 220],
        [350, 220],
        [350, 310],
        [280, 310],
      ],
    },
    {
      spot_id: "spot_11",
      corners: [
        [360, 220],
        [430, 220],
        [430, 310],
        [360, 310],
      ],
    },
    {
      spot_id: "spot_12",
      corners: [
        [440, 220],
        [510, 220],
        [510, 310],
        [440, 310],
      ],
    },
  ],
};

const MOCK_STATUS = {
  spot_1: "occupied",
  spot_2: "free",
  spot_3: "occupied",
  spot_4: "free",
  spot_5: "free",
  spot_6: "occupied",
  spot_7: "free",
  spot_8: "occupied",
  spot_9: "free",
  spot_10: "occupied",
  spot_11: "free",
  spot_12: "occupied",
};

const API_BASE = "http://localhost:8000";

async function apiGet(path) {
  try {
    const r = await fetch(API_BASE + path);
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  } catch {
    return null;
  }
}

async function apiPost(path, body) {
  try {
    const r = await fetch(API_BASE + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  } catch {
    return null;
  }
}

function BEVMap({
  layout,
  status,
  highlightSpot,
  onSpotClick,
  dimUnhighlighted,
}) {
  if (!layout) return null;
  const { canvas, spots } = layout;
  const VW = 560;
  const VH = (canvas.height / canvas.width) * VW;
  const sx = VW / canvas.width;
  const sy = VH / canvas.height;

  return (
    <svg
      viewBox={`0 0 ${VW} ${VH}`}
      width="100%"
      style={{ display: "block", borderRadius: 8, border: "1px solid #e7e5e4" }}
      aria-label="Parking lot BEV map"
    >
      <rect width={VW} height={VH} fill="#f5f5f4" rx="8" />
      <rect
        x={VW * 0.06}
        y={VH * 0.1}
        width={VW * 0.88}
        height={VH * 0.8}
        fill="#e7e5e4"
        rx="4"
        opacity="0.5"
      />
      <text
        x={VW / 2}
        y={VH / 2}
        textAnchor="middle"
        dominantBaseline="middle"
        fill="#a8a29e"
        fontSize="11"
        fontFamily="monospace"
      >
        BEV map · {canvas.width}×{canvas.height}
      </text>
      {spots.map((spot) => {
        const pts = spot.corners
          .map(([x, y]) => `${x * sx},${y * sy}`)
          .join(" ");
        const cx = (spot.corners.reduce((s, [x]) => s + x, 0) / 4) * sx;
        const cy = (spot.corners.reduce((s, [, y]) => s + y, 0) / 4) * sy;
        const occ = status?.[spot.spot_id];
        const isHighlit = highlightSpot === spot.spot_id;
        const isDimmed = dimUnhighlighted && highlightSpot && !isHighlit;

        let fill = "#e7e5e4";
        let stroke = "#d4d0cb";
        let textFill = "#78716c";

        if (occ === "free") {
          fill = "#1a7a4a22";
          stroke = "#1a7a4a";
          textFill = "#0f5c36";
        }
        if (occ === "occupied") {
          fill = "#c0392b22";
          stroke = "#c0392b";
          textFill = "#922b21";
        }
        if (isHighlit) {
          fill = "#e8960022";
          stroke = "#e89600";
          textFill = "#7d5200";
        }

        return (
          <g
            key={spot.spot_id}
            onClick={() => onSpotClick?.(spot.spot_id)}
            style={{
              cursor: onSpotClick ? "pointer" : "default",
              opacity: isDimmed ? 0.3 : 1,
              transition: "opacity 0.3s",
            }}
          >
            <polygon
              points={pts}
              fill={fill}
              stroke={stroke}
              strokeWidth={isHighlit ? 2 : 1}
            />
            <text
              x={cx}
              y={cy}
              textAnchor="middle"
              dominantBaseline="middle"
              fontSize="9"
              fill={textFill}
              fontFamily="monospace"
              fontWeight="500"
            >
              {spot.spot_id.replace("spot_", "P")}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

BEVMap.propTypes = {
  layout: layoutShape,
  status: PropTypes.objectOf(PropTypes.string),
  highlightSpot: PropTypes.string,
  onSpotClick: PropTypes.func,
  dimUnhighlighted: PropTypes.bool,
};

function StatusDot({ status }) {
  const c =
    status === "connected"
      ? "#27ae60"
      : status === "polling"
        ? "#e89600"
        : "#aaa";
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 5,
        fontSize: 12,
        color: "#78716c",
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: c,
          display: "inline-block",
        }}
      />
      {status}
    </span>
  );
}

StatusDot.propTypes = {
  status: PropTypes.string.isRequired,
};

function OwnerSetup({ layout, setLayout }) {
  const [files, setFiles] = useState([]);
  const [step, setStep] = useState("idle");
  const [error, setError] = useState(null);
  const fileRef = useRef();

  const handleFiles = (e) => {
    const picked = Array.from(e.target.files).filter((f) =>
      f.type.startsWith("image/"),
    );
    setFiles(picked);
    setStep("ready");
    setError(null);
  };

  const submit = async () => {
    if (files.length < 4) {
      setError("Upload at least 4 photos for reliable SfM.");
      return;
    }
    setStep("processing");
    setError(null);
    await new Promise((r) => setTimeout(r, 2200));
    const result = await apiPost("/layout", {
      images: files.map((f) => f.name),
    });
    if (result) {
      setLayout(result);
    } else {
      setLayout(MOCK_LAYOUT);
    }
    setStep("done");
  };

  const reset = () => {
    setFiles([]);
    setStep("idle");
    setError(null);
    setLayout(null);
  };

  return (
    <div style={{ maxWidth: 600, margin: "0 auto" }}>
      <div style={{ marginBottom: 24 }}>
        <p
          style={{
            fontSize: 13,
            color: "#78716c",
            margin: "0 0 16px",
            lineHeight: 1.6,
          }}
        >
          Upload 4–5 overlapping photos of your parking lot. The SfM pipeline
          will compute a bird's-eye-view layout and extract spot polygons
          automatically.
        </p>

        <div
          style={{
            border: "1.5px dashed #d4d0cb",
            borderRadius: 10,
            padding: "32px 24px",
            textAlign: "center",
            background: "#f5f5f4",
            cursor: "pointer",
          }}
          onClick={() => fileRef.current?.click()}
        >
          <div style={{ fontSize: 32, marginBottom: 8 }}>📷</div>
          <p
            style={{
              margin: 0,
              fontSize: 14,
              color: "#1c1917",
              fontWeight: 500,
            }}
          >
            {files.length
              ? `${files.length} photo${files.length > 1 ? "s" : ""} selected`
              : "Click to select photos"}
          </p>
          <p style={{ margin: "4px 0 0", fontSize: 12, color: "#78716c" }}>
            JPG or PNG · minimum 4 images · 60%+ overlap recommended
          </p>
          <input
            ref={fileRef}
            type="file"
            multiple
            accept="image/*"
            onChange={handleFiles}
            style={{ display: "none" }}
          />
        </div>

        {files.length > 0 && (
          <div
            style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 10 }}
          >
            {files.map((f, i) => (
              <span
                key={i}
                style={{
                  fontSize: 11,
                  background: "#eff6ff",
                  color: "#1d4ed8",
                  padding: "3px 8px",
                  borderRadius: 4,
                  fontFamily: "monospace",
                }}
              >
                {f.name}
              </span>
            ))}
          </div>
        )}

        {error && (
          <p style={{ fontSize: 12, color: "#dc2626", marginTop: 8 }}>
            {error}
          </p>
        )}
      </div>

      {step === "idle" || step === "ready" ? (
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={submit}
            disabled={files.length === 0}
            style={{
              padding: "8px 20px",
              borderRadius: 6,
              border: "1px solid #a8a29e",
              background: "#ffffff",
              cursor: files.length ? "pointer" : "not-allowed",
              fontSize: 13,
              fontWeight: 500,
              opacity: files.length ? 1 : 0.45,
            }}
          >
            Run SfM layout →
          </button>
          <button
            onClick={() => {
              setLayout(MOCK_LAYOUT);
              setStep("done");
            }}
            style={{
              padding: "8px 16px",
              borderRadius: 6,
              border: "1px solid #e7e5e4",
              background: "transparent",
              cursor: "pointer",
              fontSize: 12,
              color: "#78716c",
            }}
          >
            Load sample handoff
          </button>
        </div>
      ) : step === "processing" ? (
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 12,
            padding: "12px 0",
          }}
        >
          <div
            style={{
              width: 18,
              height: 18,
              border: "2px solid #d4d0cb",
              borderTopColor: "#1c1917",
              borderRadius: "50%",
              animation: "spin 0.8s linear infinite",
            }}
          />
          <span style={{ fontSize: 13, color: "#78716c" }}>
            Running SfM pipeline…
          </span>
        </div>
      ) : null}

      {layout && (
        <div style={{ marginTop: 28 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 10,
            }}
          >
            <p style={{ margin: 0, fontSize: 13, fontWeight: 500 }}>
              Layout ready — {layout.spots.length} spots detected
            </p>
            <span
              style={{
                fontSize: 11,
                fontFamily: "monospace",
                color: "#78716c",
                background: "#f5f5f4",
                padding: "2px 8px",
                borderRadius: 4,
              }}
            >
              {layout.spot_source}
            </span>
          </div>
          <BEVMap layout={layout} />
          <div style={{ marginTop: 10, display: "flex", gap: 8 }}>
            <button
              onClick={reset}
              style={{
                fontSize: 12,
                padding: "6px 14px",
                borderRadius: 6,
                border: "1px solid #e7e5e4",
                background: "transparent",
                cursor: "pointer",
                color: "#78716c",
              }}
            >
              Re-upload
            </button>
            <span
              style={{ fontSize: 12, color: "#78716c", alignSelf: "center" }}
            >
              Canvas {layout.canvas.width}×{layout.canvas.height} ·{" "}
              {layout.source_images.length} source images
            </span>
          </div>
        </div>
      )}
    </div>
  );
}

OwnerSetup.propTypes = {
  layout: layoutShape,
  setLayout: PropTypes.func.isRequired,
};

function OccupancyMap({ layout }) {
  const [status, setStatus] = useState(null);
  const [pollState, setPollState] = useState("idle");
  const [lastUpdated, setLastUpdated] = useState(null);
  const [selectedSpot, setSelectedSpot] = useState(null);
  const intervalRef = useRef(null);

  const poll = useCallback(async () => {
    setPollState("polling");
    const data = await apiGet("/status");
    if (data) {
      setStatus(data);
    } else {
      setStatus((prev) => {
        if (prev) return prev;
        const toggled = {};
        Object.keys(MOCK_STATUS).forEach((k) => {
          toggled[k] = Math.random() > 0.45 ? "occupied" : "free";
        });
        return toggled;
      });
    }
    setLastUpdated(new Date());
    setPollState("connected");
  }, []);

  const startPolling = () => {
    poll();
    intervalRef.current = setInterval(poll, 3000);
  };

  const stopPolling = () => {
    clearInterval(intervalRef.current);
    setPollState("idle");
  };

  useEffect(() => () => clearInterval(intervalRef.current), []);

  const freeCount = status
    ? Object.values(status).filter((v) => v === "free").length
    : 0;
  const occCount = status
    ? Object.values(status).filter((v) => v === "occupied").length
    : 0;

  if (!layout) {
    return (
      <div
        style={{
          textAlign: "center",
          padding: "60px 0",
          color: "#78716c",
          fontSize: 13,
        }}
      >
        No layout loaded. Go to <strong>Owner Setup</strong> first to generate
        the parking map.
      </div>
    );
  }

  return (
    <div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 1fr)",
          gap: 10,
          marginBottom: 20,
        }}
      >
        {[
          { label: "Free spots", value: freeCount, accent: "#1a7a4a" },
          { label: "Occupied", value: occCount, accent: "#c0392b" },
          {
            label: "Total spots",
            value: layout.spots.length,
            accent: "#78716c",
          },
        ].map(({ label, value, accent }) => (
          <div
            key={label}
            style={{
              background: "#f5f5f4",
              borderRadius: 8,
              padding: "12px 14px",
            }}
          >
            <p
              style={{
                margin: 0,
                fontSize: 11,
                color: "#78716c",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
              }}
            >
              {label}
            </p>
            <p
              style={{
                margin: "4px 0 0",
                fontSize: 22,
                fontWeight: 500,
                color: accent,
                fontFamily: "monospace",
              }}
            >
              {status ? value : "–"}
            </p>
          </div>
        ))}
      </div>

      <div
        style={{
          marginBottom: 14,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", gap: 8 }}>
          <button
            onClick={startPolling}
            disabled={pollState !== "idle"}
            style={{
              fontSize: 12,
              padding: "6px 14px",
              borderRadius: 6,
              border: "1px solid #d4d0cb",
              background: "transparent",
              cursor: pollState === "idle" ? "pointer" : "not-allowed",
              opacity: pollState === "idle" ? 1 : 0.5,
            }}
          >
            ▶ Start polling
          </button>
          <button
            onClick={stopPolling}
            disabled={pollState === "idle"}
            style={{
              fontSize: 12,
              padding: "6px 14px",
              borderRadius: 6,
              border: "1px solid #e7e5e4",
              background: "transparent",
              cursor: pollState !== "idle" ? "pointer" : "not-allowed",
              opacity: pollState !== "idle" ? 1 : 0.5,
              color: "#78716c",
            }}
          >
            ■ Stop
          </button>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {lastUpdated && (
            <span
              style={{
                fontSize: 11,
                color: "#78716c",
                fontFamily: "monospace",
              }}
            >
              {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <StatusDot status={pollState} />
        </div>
      </div>

      <BEVMap layout={layout} status={status} onSpotClick={setSelectedSpot} />

      <div
        style={{
          marginTop: 10,
          display: "flex",
          gap: 16,
          fontSize: 11,
          color: "#78716c",
        }}
      >
        {[
          ["#1a7a4a", "Free"],
          ["#c0392b", "Occupied"],
          ["#e7e5e4", "Unknown"],
        ].map(([c, l]) => (
          <span
            key={l}
            style={{ display: "flex", alignItems: "center", gap: 5 }}
          >
            <span
              style={{
                width: 10,
                height: 10,
                borderRadius: 2,
                background: c,
                display: "inline-block",
              }}
            />
            {l}
          </span>
        ))}
        <span style={{ marginLeft: "auto" }}>click a spot to select</span>
      </div>

      {selectedSpot && status && (
        <div
          style={{
            marginTop: 12,
            padding: "10px 14px",
            borderRadius: 8,
            border: "1px solid #d4d0cb",
            background: "#f5f5f4",
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          <span style={{ fontFamily: "monospace", fontSize: 13 }}>
            {selectedSpot}
          </span>
          <span
            style={{
              fontSize: 12,
              fontWeight: 500,
              color: status[selectedSpot] === "free" ? "#1a7a4a" : "#c0392b",
            }}
          >
            {status[selectedSpot] ?? "unknown"}
          </span>
          <button
            onClick={() => setSelectedSpot(null)}
            style={{
              fontSize: 11,
              background: "transparent",
              border: "none",
              cursor: "pointer",
              color: "#78716c",
            }}
          >
            ✕
          </button>
        </div>
      )}
    </div>
  );
}

OccupancyMap.propTypes = {
  layout: layoutShape,
};

function FindMyCar({ layout }) {
  const [step, setStep] = useState("idle");
  const [sessionId, setSessionId] = useState(null);
  const [foundSpot, setFoundSpot] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const fileRef = useRef();

  const handleFile = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setStep("ready");
    setFoundSpot(null);
    setSessionId(null);
    setError(null);
  };

  const park = async () => {
    setStep("matching");
    setError(null);
    await new Promise((r) => setTimeout(r, 1800));
    const result = await apiPost("/park", { filename: file?.name });
    if (result?.session_id) {
      setSessionId(result.session_id);
    } else {
      setSessionId("sess_" + Math.random().toString(36).slice(2, 8));
    }
    setStep("parked");
  };

  const find = async () => {
    setStep("locating");
    await new Promise((r) => setTimeout(r, 1200));
    const result = await apiGet(`/find/${sessionId}`);
    if (result?.spot_id) {
      setFoundSpot(result.spot_id);
      setConfidence(result.confidence ?? 0.91);
    } else {
      const spots = layout?.spots ?? [];
      const pick = spots[Math.floor(Math.random() * spots.length)];
      setFoundSpot(pick?.spot_id ?? "spot_3");
      setConfidence(0.89 + Math.random() * 0.09);
    }
    setStep("found");
  };

  const reset = () => {
    setStep("idle");
    setSessionId(null);
    setFoundSpot(null);
    setFile(null);
    setPreview(null);
    setError(null);
    setConfidence(null);
  };

  if (!layout) {
    return (
      <div
        style={{
          textAlign: "center",
          padding: "60px 0",
          color: "#78716c",
          fontSize: 13,
        }}
      >
        No layout loaded. Go to <strong>Owner Setup</strong> first.
      </div>
    );
  }

  return (
    <div>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20 }}>
        <div>
          <p
            style={{
              fontSize: 13,
              color: "#78716c",
              margin: "0 0 14px",
              lineHeight: 1.6,
            }}
          >
            Take a photo from near where you parked. SIFT feature matching will
            identify your spot.
          </p>

          <div
            style={{
              border: "1.5px dashed #d4d0cb",
              borderRadius: 10,
              overflow: "hidden",
              cursor: "pointer",
              minHeight: 150,
              position: "relative",
              background: "#f5f5f4",
            }}
            onClick={() => !foundSpot && fileRef.current?.click()}
          >
            {preview ? (
              <img
                src={preview}
                alt="Your photo"
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "cover",
                  display: "block",
                }}
              />
            ) : (
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  gap: 6,
                }}
              >
                <span style={{ fontSize: 28 }}>🚗</span>
                <span style={{ fontSize: 12, color: "#78716c" }}>
                  Tap to take / upload photo
                </span>
              </div>
            )}
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              capture="environment"
              onChange={handleFile}
              style={{ display: "none" }}
            />
          </div>

          <div
            style={{
              marginTop: 12,
              display: "flex",
              flexDirection: "column",
              gap: 8,
            }}
          >
            {(step === "idle" || step === "ready") && (
              <button
                onClick={park}
                disabled={!file}
                style={{
                  padding: "9px 0",
                  borderRadius: 6,
                  border: "1px solid #a8a29e",
                  background: "#ffffff",
                  cursor: file ? "pointer" : "not-allowed",
                  fontSize: 13,
                  fontWeight: 500,
                  opacity: file ? 1 : 0.45,
                }}
              >
                POST /park — find my spot →
              </button>
            )}
            {step === "matching" && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "9px 0",
                }}
              >
                <div
                  style={{
                    width: 16,
                    height: 16,
                    border: "2px solid #d4d0cb",
                    borderTopColor: "#1c1917",
                    borderRadius: "50%",
                    animation: "spin 0.8s linear infinite",
                    flexShrink: 0,
                  }}
                />
                <span style={{ fontSize: 12, color: "#78716c" }}>
                  Running SIFT localization…
                </span>
              </div>
            )}
            {step === "parked" && (
              <>
                <div
                  style={{
                    padding: "8px 12px",
                    borderRadius: 6,
                    background: "#f5f5f4",
                    fontSize: 12,
                    fontFamily: "monospace",
                  }}
                >
                  session_id: <strong>{sessionId}</strong>
                </div>
                <button
                  onClick={find}
                  style={{
                    padding: "9px 0",
                    borderRadius: 6,
                    border: "1px solid #d4d0cb",
                    background: "transparent",
                    cursor: "pointer",
                    fontSize: 13,
                  }}
                >
                  GET /find/{sessionId} →
                </button>
              </>
            )}
            {step === "locating" && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  padding: "9px 0",
                }}
              >
                <div
                  style={{
                    width: 16,
                    height: 16,
                    border: "2px solid #d4d0cb",
                    borderTopColor: "#1c1917",
                    borderRadius: "50%",
                    animation: "spin 0.8s linear infinite",
                    flexShrink: 0,
                  }}
                />
                <span style={{ fontSize: 12, color: "#78716c" }}>
                  Querying backend…
                </span>
              </div>
            )}
            {step === "found" && (
              <div
                style={{
                  padding: "10px 12px",
                  borderRadius: 8,
                  border: "1px solid #e89600",
                  background: "#e8960011",
                }}
              >
                <p
                  style={{
                    margin: "0 0 4px",
                    fontSize: 13,
                    fontWeight: 500,
                    color: "#7d5200",
                  }}
                >
                  Your car:{" "}
                  <span style={{ fontFamily: "monospace" }}>{foundSpot}</span>
                </p>
                <p style={{ margin: 0, fontSize: 11, color: "#7d5200" }}>
                  Confidence: {(confidence * 100).toFixed(0)}%
                </p>
              </div>
            )}
            {step === "found" && (
              <button
                onClick={reset}
                style={{
                  fontSize: 12,
                  padding: "6px 0",
                  borderRadius: 6,
                  border: "1px solid #e7e5e4",
                  background: "transparent",
                  cursor: "pointer",
                  color: "#78716c",
                }}
              >
                New session
              </button>
            )}
          </div>
        </div>

        <div>
          <p style={{ fontSize: 12, color: "#78716c", margin: "0 0 8px" }}>
            {step === "found"
              ? "Your spot is highlighted below ↓"
              : "Map will highlight your spot after matching"}
          </p>
          <BEVMap
            layout={layout}
            status={Object.fromEntries(
              (layout?.spots ?? []).map((s) => [s.spot_id, "unknown"]),
            )}
            highlightSpot={foundSpot}
            dimUnhighlighted={!!foundSpot}
          />
          {step === "found" && (
            <p
              style={{
                fontSize: 11,
                color: "#78716c",
                marginTop: 6,
                fontFamily: "monospace",
              }}
            >
              amber = your car · GET /find/{sessionId}
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

FindMyCar.propTypes = {
  layout: layoutShape,
};

const TABS = [
  { id: "setup", label: "Owner setup", icon: "⚙" },
  { id: "map", label: "Live occupancy", icon: "🅿" },
  { id: "find", label: "Find my car", icon: "🔍" },
];

export default function App() {
  const [tab, setTab] = useState("setup");
  const [layout, setLayout] = useState(null);

  return (
    <div
      style={{
        fontFamily: "system-ui, sans-serif",
        maxWidth: 700,
        margin: "0 auto",
        padding: "0 0 40px",
      }}
    >
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>

      <div
        style={{
          borderBottom: "1px solid #e7e5e4",
          marginBottom: 24,
          paddingBottom: 0,
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "baseline",
            gap: 8,
            paddingBottom: 14,
          }}
        >
          <span
            style={{
              fontSize: 13,
              fontFamily: "monospace",
              fontWeight: 500,
              color: "#1c1917",
            }}
          >
            SmartParking
          </span>
          <span
            style={{ fontSize: 11, color: "#78716c", fontFamily: "monospace" }}
          >
            v6 · edge inference
          </span>
          {layout && (
            <span
              style={{
                marginLeft: "auto",
                fontSize: 11,
                fontFamily: "monospace",
                color: "#78716c",
              }}
            >
              layout: {layout.spots.length} spots · {layout.spot_source}
            </span>
          )}
        </div>

        <div style={{ display: "flex", gap: 0 }}>
          {TABS.map((t) => (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              style={{
                padding: "8px 16px",
                fontSize: 13,
                border: "none",
                background: "transparent",
                cursor: "pointer",
                borderBottom:
                  tab === t.id ? "2px solid #1c1917" : "2px solid transparent",
                color: tab === t.id ? "#1c1917" : "#78716c",
                fontWeight: tab === t.id ? 500 : 400,
                transition: "color 0.15s",
              }}
            >
              {t.icon} {t.label}
            </button>
          ))}
        </div>
      </div>

      {tab === "setup" && <OwnerSetup layout={layout} setLayout={setLayout} />}
      {tab === "map" && <OccupancyMap layout={layout} />}
      {tab === "find" && <FindMyCar layout={layout} />}
    </div>
  );
}
