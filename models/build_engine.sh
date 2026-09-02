#!/usr/bin/env bash
# Run this locally on each machine to build a TensorRT engine matched to
# your own GPU. Never commit the resulting .engine file - it is gitignored
# on purpose, since an engine built on one GPU architecture will not
# reliably run on another.
#
# Usage: ./build_engine.sh <path-to-model.onnx> <output-engine-name>

set -e

ONNX_PATH="${1:-model.onnx}"
ENGINE_NAME="${2:-model.engine}"

trtexec \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${ENGINE_NAME}" \
    --fp16

echo "Built ${ENGINE_NAME} from ${ONNX_PATH} for this machine's GPU only."
