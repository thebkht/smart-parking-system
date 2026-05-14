#!/usr/bin/env bash
# run_demo.sh — Start backend + run edge inference in one command.
# Usage:
#   ./run_demo.sh                        # loops the bundled demo video
#   ./run_demo.sh --image /path/to/img   # uses your own image
#   ./run_demo.sh --video /path/to.mp4   # uses your own video
#   ./run_demo.sh --camera 0             # live webcam mode
#   ./run_demo.sh --camera iphone        # macOS Continuity Camera / iPhone camera

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

# ── Prepare sample source if no explicit image / video / camera arg given ────
EXTRA_ARGS=("$@")
USE_SAMPLE=true
for arg in "$@"; do
    if [[ "$arg" == "--image" || "$arg" == "--video" || "$arg" == "--camera" ]]; then
        USE_SAMPLE=false
        break
    fi
done

if $USE_SAMPLE; then
    mkdir -p samples
    SAMPLE_VIDEO="samples/12125602_3840_2160_30fps.mp4"
    SAMPLE_IMAGE="samples/demo.jpg"

    if [ -f "$SAMPLE_VIDEO" ]; then
        EXTRA_ARGS=("--video" "$SAMPLE_VIDEO" "--loop-video" "${EXTRA_ARGS[@]}")
    else
        if [ ! -f "$SAMPLE_IMAGE" ]; then
        echo "→ Copying bundled sample image..."
        python - <<'EOF'
import shutil
from ultralytics.utils import ASSETS
shutil.copy(ASSETS / "bus.jpg", "samples/demo.jpg")
EOF
        fi

        EXTRA_ARGS=("--image" "$SAMPLE_IMAGE" "${EXTRA_ARGS[@]}")
    fi
fi

# ── Create logs dir ───────────────────────────────────────────────────────────
mkdir -p logs

# ── Start backend in background ───────────────────────────────────────────────
echo "→ Starting backend on http://127.0.0.1:8000 ..."
uvicorn backend.main:app --host 127.0.0.1 --port 8000 --log-level warning &
BACKEND_PID=$!

# Ensure backend is killed when this script exits
trap 'echo ""; echo "→ Stopping backend (PID $BACKEND_PID)..."; kill "$BACKEND_PID" 2>/dev/null; wait "$BACKEND_PID" 2>/dev/null; echo "Done."' EXIT

# Wait for backend to be ready
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

# ── Run inference ─────────────────────────────────────────────────────────────
echo ""
echo "→ Running inference..."
echo "─────────────────────────────────────────────"
python edge/detect.py \
    --post \
    --save-annotated logs/demo_annotated.jpg \
    "${EXTRA_ARGS[@]}"

echo "─────────────────────────────────────────────"
echo ""
echo "→ Latest annotated frame:   logs/latest_frame.jpg"
echo "→ Log written to:           logs/parking_log_$(date +%Y-%m-%d).json"
echo ""
echo "→ Backend endpoints (while running):"
echo "   GET  http://127.0.0.1:8000/health"
echo "   GET  http://127.0.0.1:8000/status"
echo "   GET  http://127.0.0.1:8000/history"
