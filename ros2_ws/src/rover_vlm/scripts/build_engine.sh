#!/usr/bin/env bash
# Build the TensorRT image-encoder engine for THIS device from the exported ONNX.
# Engines are tied to the GPU + TensorRT version, so rebuild on each Jetson.
#
# Usage: scripts/build_engine.sh [onnx_path] [meta_json] [max_batch]
#   onnx_path  image encoder, input pixel_values [B, 3, S, S]
#              (default models/siglip2_base_patch16_224.onnx from export_siglip2.py)
#   meta_json  robot metadata from export_siglip2.py or prepare_meta.py
#              (default: the ONNX path with .json); the engine is saved next to it
#              as <meta name>_fp16.engine
#   max_batch  crops per keyframe (vlm.yaml n_crops), default 3
set -euo pipefail

PKG_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ONNX="${1:-$PKG_DIR/models/siglip2_base_patch16_224.onnx}"
JSON="${2:-${ONNX%.onnx}.json}"
MAX_BATCH="${3:-3}"
ENGINE="${JSON%.json}_fp16.engine"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

SIZE="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['image_size'])" "$JSON")"
SHAPE="pixel_values:%dx3x${SIZE}x${SIZE}"

"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENGINE" --fp16 \
  --minShapes="$(printf "$SHAPE" 1)" \
  --optShapes="$(printf "$SHAPE" "$MAX_BATCH")" \
  --maxShapes="$(printf "$SHAPE" "$MAX_BATCH")" \
  --memPoolSize=workspace:"${WORKSPACE_MB:-128}" \
  --builderOptimizationLevel="${OPT_LEVEL:-1}" --maxAuxStreams=0 --skipInference
# The Orin Nano shares its 8 GB with the desktop: at the default optimization level (3)
# the builder ran out of memory (NvMap error 12) with VS Code open. Level 1, a small
# workspace and no auxiliary streams fit; "Skipping tactic ... allocation failed" lines
# in the log are tactics that did not fit, not a failed build.
echo "Saved: $ENGINE"
echo "Check accuracy and latency with: scripts/validate_model.py --engine $ENGINE --meta $JSON --images <dir>"
