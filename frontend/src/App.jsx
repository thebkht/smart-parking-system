import { useState, useEffect, useRef, useCallback } from "react";
import PropTypes from "prop-types";
import {
  CameraIcon,
  GearIcon,
  MagnifyingGlassIcon,
  PlayIcon,
  StopIcon,
  Cross2Icon,
  ArrowRightIcon,
  ArrowDownIcon,
  ImageIcon,
  TokensIcon,
  UpdateIcon,
} from "@radix-ui/react-icons";

// ---------------------------------------------------------------------------
// PropTypes shapes
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
  source_images: PropTypes.arrayOf(PropTypes.string),
  spot_source: PropTypes.string,
  spots: PropTypes.arrayOf(spotShape),
});

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// API helpers
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// BEVMap
// ---------------------------------------------------------------------------
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
      className="block rounded-lg border border-stone-200"
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

// ---------------------------------------------------------------------------
// StatusDot
// ---------------------------------------------------------------------------
function StatusDot({ status }) {
  const dotColor =
    status === "connected"
      ? "bg-green-500"
      : status === "polling"
        ? "bg-amber-500"
        : "bg-stone-400";

  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-stone-500">
      <span className={`inline-block size-1.75 rounded-full ${dotColor}`} />
      {status}
    </span>
  );
}

StatusDot.propTypes = { status: PropTypes.string.isRequired };

// ---------------------------------------------------------------------------
// Spinner — Radix UpdateIcon with Tailwind spin
// ---------------------------------------------------------------------------
function Spinner() {
  return (
    <UpdateIcon className="w-4 h-4 text-stone-500 animate-spin shrink-0" />
  );
}

// ---------------------------------------------------------------------------
// OwnerSetup
// ---------------------------------------------------------------------------
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
    setLayout(result ?? MOCK_LAYOUT);
    setStep("done");
  };

  const reset = () => {
    setFiles([]);
    setStep("idle");
    setError(null);
    setLayout(null);
  };

  return (
    <div className="max-w-xl mx-auto">
      <div className="mb-6">
        <p className="text-[13px] text-stone-500 mb-4 leading-relaxed">
          Upload 4–5 overlapping photos of your parking lot. The SfM pipeline
          will compute a bird&apos;s-eye-view layout and extract spot polygons
          automatically.
        </p>

        {/* Drop zone */}
        <div
          className="border-2 border-dashed border-stone-300 rounded-xl p-8 text-center bg-stone-50
                     cursor-pointer hover:border-stone-400 transition-colors"
          onClick={() => fileRef.current?.click()}
        >
          <div className="flex justify-center mb-2">
            <CameraIcon className="w-8 h-8 text-stone-400" />
          </div>
          <p className="text-sm font-medium text-stone-800">
            {files.length
              ? `${files.length} photo${files.length > 1 ? "s" : ""} selected`
              : "Click to select photos"}
          </p>
          <p className="text-xs text-stone-400 mt-1">
            JPG or PNG · minimum 4 images · 60%+ overlap recommended
          </p>
          <input
            ref={fileRef}
            type="file"
            multiple
            accept="image/*"
            onChange={handleFiles}
            className="hidden"
          />
        </div>

        {/* File chips */}
        {files.length > 0 && (
          <div className="flex flex-wrap gap-1.5 mt-2.5">
            {files.map((f, i) => (
              <span
                key={i}
                className="text-[11px] bg-blue-50 text-blue-700 px-2 py-0.5 rounded font-mono"
              >
                {f.name}
              </span>
            ))}
          </div>
        )}

        {error && <p className="text-xs text-red-600 mt-2">{error}</p>}
      </div>

      {/* Actions */}
      {(step === "idle" || step === "ready") && (
        <div className="flex gap-2">
          <button
            onClick={submit}
            disabled={files.length === 0}
            className="inline-flex items-center gap-1.5 px-5 py-2 rounded-md border border-stone-400
                       bg-white text-[13px] font-medium hover:bg-stone-50 transition-colors
                       disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            Run SfM layout
            <ArrowRightIcon className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => {
              setLayout(MOCK_LAYOUT);
              setStep("done");
            }}
            className="px-4 py-2 rounded-md border border-stone-200 bg-transparent text-xs
                       text-stone-500 hover:bg-stone-50 transition-colors cursor-pointer"
          >
            Load sample handoff
          </button>
        </div>
      )}

      {step === "processing" && (
        <div className="flex items-center gap-3 py-3">
          <Spinner />
          <span className="text-[13px] text-stone-500">
            Running SfM pipeline…
          </span>
        </div>
      )}

      {/* Layout preview */}
      {layout && (
        <div className="mt-7">
          <div className="flex justify-between items-center mb-2.5">
            <p className="text-[13px] font-medium text-stone-800">
              Layout ready — {layout.spots.length} spots detected
            </p>
            <span className="text-[11px] font-mono text-stone-500 bg-stone-100 px-2 py-0.5 rounded">
              {layout.spot_source}
            </span>
          </div>
          <BEVMap layout={layout} />
          <div className="flex gap-2 mt-2.5 items-center">
            <button
              onClick={reset}
              className="text-xs px-3.5 py-1.5 rounded-md border border-stone-200 bg-transparent
                         text-stone-500 hover:bg-stone-50 transition-colors cursor-pointer"
            >
              Re-upload
            </button>
            <span className="text-xs text-stone-500">
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

// ---------------------------------------------------------------------------
// OccupancyMap
// ---------------------------------------------------------------------------
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
      <div className="text-center py-16 text-stone-500 text-[13px]">
        No layout loaded. Go to <strong>Owner Setup</strong> first to generate
        the parking map.
      </div>
    );
  }

  const stats = [
    { label: "Free spots", value: freeCount, color: "text-green-700" },
    { label: "Occupied", value: occCount, color: "text-red-700" },
    {
      label: "Total spots",
      value: layout.spots.length,
      color: "text-stone-500",
    },
  ];

  return (
    <div>
      {/* Stats cards */}
      <div className="grid grid-cols-3 gap-2.5 mb-5">
        {stats.map(({ label, value, color }) => (
          <div key={label} className="bg-stone-100 rounded-lg px-3.5 py-3">
            <p className="text-[11px] text-stone-400 uppercase tracking-wide">
              {label}
            </p>
            <p className={`mt-1 text-[22px] font-medium font-mono ${color}`}>
              {status ? value : "–"}
            </p>
          </div>
        ))}
      </div>

      {/* Controls */}
      <div className="flex items-center justify-between mb-3.5">
        <div className="flex gap-2">
          <button
            onClick={startPolling}
            disabled={pollState !== "idle"}
            className="inline-flex items-center gap-1.5 text-xs px-3.5 py-1.5 rounded-md
                       border border-stone-300 bg-transparent hover:bg-stone-50 transition-colors
                       cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <PlayIcon className="w-3 h-3" />
            Start polling
          </button>
          <button
            onClick={stopPolling}
            disabled={pollState === "idle"}
            className="inline-flex items-center gap-1.5 text-xs px-3.5 py-1.5 rounded-md
                       border border-stone-200 bg-transparent text-stone-500 hover:bg-stone-50
                       transition-colors cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <StopIcon className="w-3 h-3" />
            Stop
          </button>
        </div>
        <div className="flex items-center gap-3">
          {lastUpdated && (
            <span className="text-[11px] text-stone-400 font-mono">
              {lastUpdated.toLocaleTimeString()}
            </span>
          )}
          <StatusDot status={pollState} />
        </div>
      </div>

      <BEVMap layout={layout} status={status} onSpotClick={setSelectedSpot} />

      {/* Legend */}
      <div className="flex gap-4 mt-2.5 text-[11px] text-stone-500 items-center">
        {[
          ["#1a7a4a", "Free"],
          ["#c0392b", "Occupied"],
          ["#e7e5e4", "Unknown"],
        ].map(([c, l]) => (
          <span key={l} className="flex items-center gap-1.5">
            <span
              className="inline-block size-2.5 rounded-xs"
              style={{ background: c }}
            />
            {l}
          </span>
        ))}
        <span className="ml-auto">click a spot to select</span>
      </div>

      {/* Selected spot panel */}
      {selectedSpot && status && (
        <div
          className="mt-3 px-3.5 py-2.5 rounded-lg border border-stone-300 bg-stone-50
                        flex justify-between items-center"
        >
          <span className="font-mono text-[13px]">{selectedSpot}</span>
          <span
            className={`text-xs font-medium ${
              status[selectedSpot] === "free"
                ? "text-green-700"
                : "text-red-700"
            }`}
          >
            {status[selectedSpot] ?? "unknown"}
          </span>
          <button
            onClick={() => setSelectedSpot(null)}
            className="flex items-center justify-center p-0.5 rounded text-stone-400
                       hover:text-stone-600 bg-transparent border-none cursor-pointer"
            aria-label="Dismiss"
          >
            <Cross2Icon className="w-3.5 h-3.5" />
          </button>
        </div>
      )}
    </div>
  );
}

OccupancyMap.propTypes = { layout: layoutShape };

// ---------------------------------------------------------------------------
// FindMyCar
// ---------------------------------------------------------------------------
function FindMyCar({ layout }) {
  const [step, setStep] = useState("idle");
  const [sessionId, setSessionId] = useState(null);
  const [foundSpot, setFoundSpot] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null); // eslint-disable-line no-unused-vars
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
    setSessionId(
      result?.session_id ?? "sess_" + Math.random().toString(36).slice(2, 8),
    );
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
      <div className="text-center py-16 text-stone-500 text-[13px]">
        No layout loaded. Go to <strong>Owner Setup</strong> first.
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 gap-5">
      {/* Left column */}
      <div>
        <p className="text-[13px] text-stone-500 mb-3.5 leading-relaxed">
          Take a photo from near where you parked. SIFT feature matching will
          identify your spot.
        </p>

        {/* Photo input */}
        <div
          className="border-2 border-dashed border-stone-300 rounded-xl overflow-hidden cursor-pointer
                     min-h-37.5 relative bg-stone-50 hover:border-stone-400 transition-colors"
          onClick={() => !foundSpot && fileRef.current?.click()}
        >
          {preview ? (
            <img
              src={preview}
              alt="Your photo"
              className="w-full h-full object-cover block"
            />
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-2">
              <TokensIcon className="w-8 h-8 text-stone-400" />
              <span className="text-xs text-stone-500">
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
            className="hidden"
          />
        </div>

        {/* Step buttons */}
        <div className="mt-3 flex flex-col gap-2">
          {(step === "idle" || step === "ready") && (
            <button
              onClick={park}
              disabled={!file}
              className="inline-flex items-center justify-center gap-1.5 py-2.5 rounded-md
                         border border-stone-400 bg-white text-[13px] font-medium hover:bg-stone-50
                         transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
            >
              POST /park — find my spot
              <ArrowRightIcon className="w-3.5 h-3.5" />
            </button>
          )}

          {step === "matching" && (
            <div className="flex items-center gap-2.5 py-2.5">
              <Spinner />
              <span className="text-xs text-stone-500">
                Running SIFT localization…
              </span>
            </div>
          )}

          {step === "parked" && (
            <>
              <div className="px-3 py-2 rounded-md bg-stone-100 text-xs font-mono">
                session_id: <strong>{sessionId}</strong>
              </div>
              <button
                onClick={find}
                className="inline-flex items-center justify-center gap-1.5 py-2.5 rounded-md
                           border border-stone-300 bg-transparent text-[13px] hover:bg-stone-50
                           transition-colors cursor-pointer"
              >
                GET /find/{sessionId}
                <ArrowRightIcon className="w-3.5 h-3.5" />
              </button>
            </>
          )}

          {step === "locating" && (
            <div className="flex items-center gap-2.5 py-2.5">
              <Spinner />
              <span className="text-xs text-stone-500">Querying backend…</span>
            </div>
          )}

          {step === "found" && (
            <>
              <div className="px-3 py-2.5 rounded-lg border border-amber-400 bg-amber-50">
                <p className="text-[13px] font-medium text-amber-800 mb-1">
                  Your car: <span className="font-mono">{foundSpot}</span>
                </p>
                <p className="text-[11px] text-amber-700">
                  Confidence: {(confidence * 100).toFixed(0)}%
                </p>
              </div>
              <button
                onClick={reset}
                className="text-xs py-1.5 rounded-md border border-stone-200 bg-transparent
                           text-stone-500 hover:bg-stone-50 transition-colors cursor-pointer"
              >
                New session
              </button>
            </>
          )}
        </div>
      </div>

      {/* Right column — map */}
      <div>
        <p className="text-xs text-stone-500 mb-2 flex items-center gap-1">
          {step === "found" ? (
            <>
              Your spot is highlighted below{" "}
              <ArrowDownIcon className="w-3 h-3" />
            </>
          ) : (
            <>
              <ImageIcon className="w-3 h-3" /> Map will highlight your spot
              after matching
            </>
          )}
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
          <p className="text-[11px] text-stone-400 mt-1.5 font-mono">
            amber = your car · GET /find/{sessionId}
          </p>
        )}
      </div>
    </div>
  );
}

FindMyCar.propTypes = { layout: layoutShape };

// ---------------------------------------------------------------------------
// Root App
// ---------------------------------------------------------------------------
const TABS = [
  { id: "setup", label: "Owner setup", Icon: GearIcon },
  { id: "map", label: "Live occupancy", Icon: UpdateIcon },
  { id: "find", label: "Find my car", Icon: MagnifyingGlassIcon },
];

export default function App() {
  const [tab, setTab] = useState("setup");
  const [layout, setLayout] = useState(null);

  return (
    <div className="font-sans max-w-175 mx-auto pb-10">
      {/* Header */}
      <div className="border-b border-stone-200 mb-6">
        <div className="flex items-baseline gap-2 pb-3.5">
          <span className="text-[13px] font-mono font-medium text-stone-900">
            SmartParking
          </span>
          <span className="text-[11px] text-stone-400 font-mono">
            v6 · edge inference
          </span>
          {layout && (
            <span className="ml-auto text-[11px] font-mono text-stone-400">
              layout: {layout.spots.length} spots · {layout.spot_source}
            </span>
          )}
        </div>

        {/* Tab bar */}
        <div className="flex">
          {TABS.map(({ id, label, Icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={[
                "inline-flex items-center gap-1.5 px-4 py-2 text-[13px] border-b-2 transition-colors",
                "cursor-pointer bg-transparent border-x-0 border-t-0",
                tab === id
                  ? "border-stone-900 text-stone-900 font-medium"
                  : "border-transparent text-stone-400 hover:text-stone-600",
              ].join(" ")}
            >
              <Icon className="w-3.5 h-3.5" />
              {label}
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
