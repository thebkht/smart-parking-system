#!/usr/bin/env bash
# Download the promoted Stage 2 classifier weights from a GitHub Release.
#
# Weights are published as Release assets rather than committed to git history
# (see the README "Model weights" section and MODEL_LICENSE.md). This script
# fetches them into acpds_cls/weights/ where the edge runtime and ml/ scripts
# expect them (Makefile: STAGE2_WEIGHTS ?= acpds_cls/weights/best.pt).
#
# Usage:
#   scripts/fetch_weights.sh                 # fetch best.pt for the default tag
#   RELEASE_TAG=v1.0.0 scripts/fetch_weights.sh
#   ASSETS="best.pt best.onnx best_int8.onnx" scripts/fetch_weights.sh
set -euo pipefail

REPO="${REPO:-thebkht/smart-parking-system}"
RELEASE_TAG="${RELEASE_TAG:-v1.0.0}"
ASSETS="${ASSETS:-best.pt}"
DEST="${DEST:-acpds_cls/weights}"

mkdir -p "$DEST"

for asset in $ASSETS; do
  url="https://github.com/${REPO}/releases/download/${RELEASE_TAG}/${asset}"
  out="${DEST}/${asset}"
  echo "Fetching ${asset} from ${RELEASE_TAG} -> ${out}"
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "$out" "$url"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "$out" "$url"
  else
    echo "error: need curl or wget to download weights" >&2
    exit 1
  fi
done

echo "Done. Weights in ${DEST}/:"
ls -lh "$DEST"
