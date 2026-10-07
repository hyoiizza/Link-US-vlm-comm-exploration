#!/usr/bin/env bash
# Build a TensorRT engine for THIS device from the ONNX model.
# Engines are tied to the GPU + TensorRT version, so rebuild on each Jetson.
#
# Usage: scripts/build_engine.sh [fp16|int8] [onnx_path]
set -euo pipefail

PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PRECISION="${1:-fp16}"
ONNX="${2:-$PKG_DIR/models/best.onnx}"
OUT_DIR="$PKG_DIR/models/optimized"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
mkdir -p "$OUT_DIR"

case "$PRECISION" in
  fp16) FLAGS=(--fp16) ;;
  # INT8 without a calibration cache uses dummy scales -> only valid for a Q/DQ (pre-quantized) ONNX.
  int8) FLAGS=(--fp16 --int8) ;;
  *) echo "unknown precision: $PRECISION" >&2; exit 1 ;;
esac

ENGINE="$OUT_DIR/best_${PRECISION}.engine"
"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" "${FLAGS[@]}" \
  --memPoolSize=workspace:1024 --skipInference
echo "Saved: $ENGINE"
