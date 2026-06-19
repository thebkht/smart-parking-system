#!/usr/bin/env bash
# run_demo.sh — Launch the full demo stack in one command:
#   1. backend  (FastAPI / uvicorn      → http://127.0.0.1:8000)
#   2. web       (Vite dev server        → http://localhost:5173)
#   3. mobile    (Expo / Metro bundler   → http://localhost:8081, QR for Expo Go)
# It also seeds live occupancy once from the curated ACPDS sample sequence so the
# apps have data to render. Press Ctrl+C to stop everything.
#
# Usage:
#   ./run_demo.sh                        # seed from curated samples (samples/live_occupancy)
#   ./run_demo.sh --image /path/to/img   # seed from your own image
#   ./run_demo.sh --video /path/to.mp4   # seed from your own video
#   ./run_demo.sh --camera 0             # seed from a live webcam
#   ./run_demo.sh --no-seed              # skip edge seeding, just run the 3 services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Activate virtualenv ──────────────────────────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
else
    echo "ERROR: .venv not found. Run: make install"
    exit 1
fi

# ── Parse args: pull out --no-seed, pass the rest to the edge seeder ─────────
SEED=true
EXTRA_ARGS=()
USE_SAMPLE=true
for arg in "$@"; do
    case "$arg" in
        --no-seed) SEED=false ;;
        --image|--video|--camera) USE_SAMPLE=false; EXTRA_ARGS+=("$arg") ;;
        *) EXTRA_ARGS+=("$arg") ;;
    esac
done

if $SEED && $USE_SAMPLE; then
    # Curated ACPDS demo set: the live-occupancy sequence (same lot over time).
    # edge/detect.py treats a directory as a sorted image sequence.
    SAMPLE_FRAMES="samples/live_occupancy"
    if [ -d "$SAMPLE_FRAMES" ] && ls "$SAMPLE_FRAMES"/*.JPG >/dev/null 2>&1; then
        EXTRA_ARGS=("--image" "$SAMPLE_FRAMES" "${EXTRA_ARGS[@]}")
    else
        echo "WARNING: demo samples not found in $SAMPLE_FRAMES; skipping seed."
        SEED=false
    fi
fi

mkdir -p logs

# ── Track child processes and tear them all down on exit ─────────────────────
PIDS=()
cleanup() {
    trap - EXIT INT TERM
    echo ""
    echo "→ Shutting down demo stack..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            pkill -TERM -P "$pid" 2>/dev/null || true
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
    wait 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

# ── 1. Backend ───────────────────────────────────────────────────────────────
# Bind to 0.0.0.0 so the mobile app (Expo Go on a phone/simulator) can reach the
# backend over the LAN, not just loopback. Expo's api.js derives http://<lan-ip>:8000.
echo "→ Starting backend on http://0.0.0.0:8000 (LAN-reachable) ..."
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --log-level warning &
PIDS+=($!)

echo "→ Waiting for backend..."
for i in $(seq 1 20); do
    if curl -sf http://127.0.0.1:8000/health > /dev/null 2>&1; then
        echo "→ Backend ready."
        break
    fi
    sleep 0.5
    if [ "$i" -eq 20 ]; then
        echo "ERROR: Backend did not start in time."
        exit 1
    fi
done

# ── 2. Seed live occupancy (non-fatal: the app stack runs regardless) ────────
if $SEED; then
    echo ""
    echo "→ Seeding live occupancy via edge inference..."
    echo "─────────────────────────────────────────────"
    set +e
    python edge/detect.py \
        --post \
        --save-annotated logs/demo_annotated.jpg \
        "${EXTRA_ARGS[@]}"
    if [ $? -ne 0 ]; then
        echo "WARNING: edge seeding failed (e.g. missing Stage-2 weights)."
        echo "         The apps will still run; configure /map + occupancy manually."
    fi
    set -e
    echo "─────────────────────────────────────────────"
fi

# ── 3. Web frontend (Vite) ───────────────────────────────────────────────────
if [ -d "frontend/node_modules" ]; then
    echo "→ Starting web frontend on http://localhost:5173 ..."
    ( cd frontend && npm run dev ) &
    PIDS+=($!)
else
    echo "WARNING: frontend/node_modules missing — run 'npm install' in frontend/. Skipping web."
fi

# ── 4. Mobile (Expo / Metro) ─────────────────────────────────────────────────
if [ -d "frontend/mobile/node_modules" ]; then
    echo "→ Starting mobile (Expo) on http://localhost:8081 ..."
    ( cd frontend/mobile && npm run start ) &
    PIDS+=($!)
else
    echo "WARNING: frontend/mobile/node_modules missing — run 'npm install' in frontend/mobile/. Skipping mobile."
fi

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "═════════════════════════════════════════════"
echo " Demo stack running. Press Ctrl+C to stop."
echo "   Backend:  http://127.0.0.1:8000  (/health /status /map /history)"
echo "   Web:      http://localhost:5173"
echo "   Mobile:   http://localhost:8081  (scan the Expo QR with Expo Go)"
echo "═════════════════════════════════════════════"
echo ""

# Keep running until a service exits or the user hits Ctrl+C.
wait
