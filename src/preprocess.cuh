#pragma once
// ---------------------------------------------------------------------------
// GPU preprocessing stage: camera frame -> model input tensor.
//
// One kernel does all four transforms the detector needs, in a single pass
// over the pixels:
//   1. letterbox resize (bilinear, aspect-preserving, padded)   - geometry
//   2. BGR -> RGB                                               - channel order
//   3. /255 normalisation                                       - value range
//   4. HWC -> CHW, batched                                      - memory layout
//
// Doing them as one fused kernel rather than four passes is the whole point:
// each of those steps on its own is memory-bound, so four passes would move
// the frame through GPU memory four times to do a few arithmetic ops. Fusing
// makes the stage cost roughly one read plus one write.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include "preprocess_math.cuh"

namespace reflex {

// The launch configuration, decided once on the host at startup rather than
// per frame. Printed at startup so the choice is visible in logs, and reported
// in the README latency table alongside the measured numbers.
struct PreprocPlan {
    bool   use_shared = true;   // false => fell back to the global-memory kernel
    int    tile_w = 0, tile_h = 0;
    int    win_w = 0, win_h = 0; // source-pixel window each block caches
    size_t shared_bytes = 0;     // dynamic shared memory requested per block
    size_t shared_limit = 0;     // what the device actually allows per block
    dim3   grid{1, 1, 1};
    dim3   block{1, 1, 1};
};

// Works out the grid/block/shared-memory configuration for a given letterbox
// geometry and batch size. Queries the device for its real shared-memory
// limit, so a machine with a smaller limit transparently gets the
// global-memory kernel instead of a launch failure.
PreprocPlan plan_preprocess(const LetterboxParams& p, int batch, int device = 0);

// Enqueues the preprocessing kernel on `stream`. Nothing synchronises here -
// the caller owns the stream and decides when to wait.
//
//   d_src_bgr   device buffer, batch * src_h * src_w * 3 bytes, packed BGR8
//               (i.e. exactly what an OpenCV CV_8UC3 frame looks like)
//   d_dst_nchw  device buffer, batch * 3 * dst_h * dst_w floats
//   norm_scale  multiplier applied to raw 0-255 values; 1/255 for YOLOv8
//   pad_value   value written into the padding strip, already in OUTPUT units
//               (so 114.0f/255.0f, not 114.0f)
void launch_preprocess(const uint8_t* d_src_bgr,
                       float* d_dst_nchw,
                       const LetterboxParams& p,
                       int batch,
                       const PreprocPlan& plan,
                       float norm_scale,
                       float pad_value,
                       cudaStream_t stream);

} // namespace reflex
