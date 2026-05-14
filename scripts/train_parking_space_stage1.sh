#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SOURCE_DATASET="${SOURCE_DATASET:-datasets/parking-space-detection}"
YOLO_DATASET="${YOLO_DATASET:-${SOURCE_DATASET}/yolo-stage1}"
DATA_YAML="${DATA_YAML:-${YOLO_DATASET}/dataset.yaml}"
WEIGHTS="${WEIGHTS:-runs/stage1_det/yolov8s_stage1/weights/best.pt}"
PROJECT="${PROJECT:-runs/stage1_finetune}"
NAME="${NAME:-parking_space_detection}"
DEVICE="${DEVICE:-mps}"
EPOCHS="${EPOCHS:-50}"
IMGSZ="${IMGSZ:-640}"
BATCH="${BATCH:-8}"
LR="${LR:-0.001}"
FREEZE="${FREEZE:-10}"
SEED="${SEED:-42}"

if [[ ! -f "$WEIGHTS" ]]; then
  echo "Missing weights: $WEIGHTS" >&2
  exit 1
fi

echo "[1/2] Preparing one-class YOLO dataset at $YOLO_DATASET"
python3 tools/prepare_parking_space_detection_yolo.py \
  --source "$SOURCE_DATASET" \
  --output "$YOLO_DATASET" \
  --collapse-classes \
  --seed "$SEED"

echo "[2/2] Fine-tuning Stage 1 detector from $WEIGHTS"
python3 ml/train.py \
  --stage1 \
  --data "$DATA_YAML" \
  --weights "$WEIGHTS" \
  --epochs "$EPOCHS" \
  --imgsz "$IMGSZ" \
  --freeze "$FREEZE" \
  --lr "$LR" \
  --batch "$BATCH" \
  --device "$DEVICE" \
  --project "$PROJECT" \
  --name "$NAME" \
  --exist-ok \
  --no-amp
