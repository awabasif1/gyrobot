#!/usr/bin/env bash
# Build a TensorRT engine from the committed ONNX model, for THIS machine's GPU.
#
# Run this once per machine, and again after any driver or TensorRT upgrade.
# The resulting .engine file is gitignored on purpose: it is tied to the GPU
# architecture, the TensorRT version AND the driver it was built against.
# Copying one between the two machines on this project produces a
# deserialisation failure that looks like a corrupt file.
#
# Usage:
#   ./build_engine.sh [model.onnx] [model.engine] [extra trtexec args...]
#
# Examples:
#   ./build_engine.sh                                  # fp16, batch 1
#   ./build_engine.sh model.onnx m8.engine --shapes=images:8x3x640x640
#   ./build_engine.sh model.onnx m.engine --best       # let TRT pick per layer

set -euo pipefail

ONNX_PATH="${1:-model.onnx}"
ENGINE_NAME="${2:-model.engine}"
# Consume the two positional arguments if they were given, so "$@" below holds
# only the extra trtexec flags. Written this way because a bare `shift 2` fails
# when fewer than two arguments were passed, and `set -e` would abort on it.
if [ $# -gt 2 ]; then shift 2; else shift $#; fi

if ! command -v trtexec >/dev/null 2>&1; then
    cat >&2 <<'EOF'
error: trtexec not found on PATH.

trtexec ships with the standalone TensorRT SDK (not with the CUDA Toolkit, and
not with the pip 'tensorrt' wheel). After installing the SDK, add its bin
directory to PATH, e.g.:

    export PATH=$PATH:/usr/src/tensorrt/bin           # .deb install
    export PATH=$PATH:/opt/TensorRT-10.x/bin          # tarball install

Record the version you used in docs/ENVIRONMENT.md.
EOF
    exit 1
fi

if [ ! -f "$ONNX_PATH" ]; then
    echo "error: $ONNX_PATH not found." >&2
    echo "  Produce it first:  python3 export_onnx.py" >&2
    exit 1
fi

echo "building for this machine's GPU:"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,compute_cap,driver_version \
               --format=csv,noheader || true
else
    echo "  (nvidia-smi not found - is the driver installed?)"
fi
echo

# --fp16 lets TensorRT run layers in half precision where that is numerically
# safe. It is the single biggest latency win available here and costs almost
# nothing in detection quality for this kind of model. Note it does NOT change
# the engine's input/output tensors - those stay fp32, which is what
# src/trt_infer.cpp expects. (Adding --inputIOFormats=fp16:chw WOULD change
# them, and the pipeline rejects such an engine with an explanatory error.)
#
# For an INT8 engine you also need a calibration dataset; that is a worthwhile
# experiment once fp16 is working and measured, not before.
set -x
trtexec \
    --onnx="${ONNX_PATH}" \
    --saveEngine="${ENGINE_NAME}" \
    --fp16 \
    "$@"
set +x

echo
echo "built ${ENGINE_NAME} for this machine's GPU only - do not commit it."
echo "verify it loads:  ../build/reflex_cv --engine ${ENGINE_NAME} --no-send --help"
