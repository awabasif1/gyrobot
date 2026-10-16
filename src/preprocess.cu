#include "preprocess.cuh"
#include "cuda_utils.cuh"

namespace reflex {

// ===========================================================================
// Launch geometry, and why it is these numbers
// ===========================================================================
//
// Block = 32 x 8 = 256 threads.
//
// Why 32 wide: the output tensor is NCHW float32, so within one channel plane
// threads with consecutive x hold consecutive addresses. A 32-lane warp
// therefore writes exactly 32 * 4 = 128 contiguous bytes - one full cache line
// transaction, no partial sectors. Had the tile been 16 wide, each warp would
// have straddled two output rows and issued two 64-byte stores per channel
// instead of one 128-byte store, for no gain anywhere else.
//
// Why 8 tall: 32 * 8 = 256 threads per block. An sm_86 SM retires at most 1536
// resident threads, so exactly 6 of these blocks saturate an SM. 6 blocks also
// stay inside the 100 KB of shared memory an sm_86 SM has as long as each
// block asks for under ~16 KB, which holds for every sane camera-to-model
// scale (see plan_preprocess below, and the fallback when it does not).
//
// Why not one thread per output pixel with a flat 1-D grid: the shared-memory
// variant needs a 2-D neighbourhood of source pixels per block, which only
// makes sense if the block itself is a 2-D tile of the output.
static constexpr int kTileW = 32;
static constexpr int kTileH = 8;

// ===========================================================================
// Variant A: shared-memory tile cache (the fast path)
// ===========================================================================
//
// Neighbouring output pixels read overlapping source pixels - at scale ~1.0
// every source pixel is touched by up to four different output pixels, and
// more as the image is downscaled further. Left in global memory those are
// repeat trips through L2. So each block first cooperatively copies the source
// window it needs into shared memory, then every thread interpolates out of
// shared memory instead.
//
// The window is sized on the host and passed in (win_w/win_h), identical for
// every block, so there is no way for the kernel's indexing to disagree with
// the host's allocation.
__global__ void preprocess_shared_kernel(const uint8_t* __restrict__ src,
                                         float* __restrict__ dst,
                                         const LetterboxParams p,
                                         const int win_w, const int win_h,
                                         const float norm_scale,
                                         const float pad_value) {
    extern __shared__ uchar4 s_tile[];

    const int n       = blockIdx.z;
    const int tile_x0 = blockIdx.x * kTileW;
    const int tile_y0 = blockIdx.y * kTileH;

    // Source-space origin of this block's window. floorf because the first
    // bilinear tap for the tile's first output pixel is floor() of its mapped
    // coordinate; tile_window_extent() already added the slack this rounding
    // costs.
    const int win_x0 = static_cast<int>(floorf(dst_to_src_x(p, static_cast<float>(tile_x0))));
    const int win_y0 = static_cast<int>(floorf(dst_to_src_y(p, static_cast<float>(tile_y0))));

    const uint8_t* src_n =
        src + static_cast<size_t>(n) * p.src_h * p.src_w * 3;

    // --- cooperative load -------------------------------------------------
    // Strided 2-D loop keeps consecutive lanes on consecutive lx, so the reads
    // coalesce. Coordinates are clamped rather than bounds-checked, which
    // replicates the edge pixel - the same border behaviour
    // bilinear_weights() assumes, and the same one OpenCV uses.
    for (int ly = threadIdx.y; ly < win_h; ly += blockDim.y) {
        const int sy = rcv_clampi(win_y0 + ly, 0, p.src_h - 1);
        for (int lx = threadIdx.x; lx < win_w; lx += blockDim.x) {
            const int sx = rcv_clampi(win_x0 + lx, 0, p.src_w - 1);
            const uint8_t* q =
                src_n + (static_cast<size_t>(sy) * p.src_w + sx) * 3;
            // Padded to uchar4 so each entry is 4-byte aligned. The wasted
            // byte per pixel is cheaper than the unaligned 3-byte shared
            // accesses and bank conflicts a packed uchar3 tile would cause.
            s_tile[ly * win_w + lx] = make_uchar4(q[0], q[1], q[2], 0);
        }
    }
    __syncthreads();

    // Guard AFTER the barrier: threads outside the output still had to help
    // with the load above, so they cannot return before everyone syncs.
    const int dx = tile_x0 + threadIdx.x;
    const int dy = tile_y0 + threadIdx.y;
    if (dx >= p.dst_w || dy >= p.dst_h) return;

    float r, g, b;
    if (in_padding(p, dx, dy)) {
        r = g = b = pad_value;
    } else {
        const float fx = dst_to_src_x(p, static_cast<float>(dx));
        const float fy = dst_to_src_y(p, static_cast<float>(dy));
        const BilinearWeights w = bilinear_weights(fx, fy, p.src_w, p.src_h);

        // Local (shared-memory) coordinates of the four taps. Clamped again as
        // cheap insurance: if a future change to tile_window_extent() ever
        // under-sized the window, this degrades the pixel instead of reading
        // out of bounds.
        const int lx0 = rcv_clampi(w.x0 - win_x0, 0, win_w - 1);
        const int lx1 = rcv_clampi(w.x1 - win_x0, 0, win_w - 1);
        const int ly0 = rcv_clampi(w.y0 - win_y0, 0, win_h - 1);
        const int ly1 = rcv_clampi(w.y1 - win_y0, 0, win_h - 1);

        // Four loads total, reused across all three channels.
        const uchar4 p00 = s_tile[ly0 * win_w + lx0];
        const uchar4 p10 = s_tile[ly0 * win_w + lx1];
        const uchar4 p01 = s_tile[ly1 * win_w + lx0];
        const uchar4 p11 = s_tile[ly1 * win_w + lx1];

        // Source is packed BGR (.x=B .y=G .z=R); output wants RGB.
        b = blend4(p00.x, p10.x, p01.x, p11.x, w.wx, w.wy) * norm_scale;
        g = blend4(p00.y, p10.y, p01.y, p11.y, w.wx, w.wy) * norm_scale;
        r = blend4(p00.z, p10.z, p01.z, p11.z, w.wx, w.wy) * norm_scale;
    }

    // HWC -> CHW: the three channels of one pixel land one plane apart.
    const size_t plane = static_cast<size_t>(p.dst_h) * p.dst_w;
    const size_t base  = static_cast<size_t>(n) * 3 * plane +
                         static_cast<size_t>(dy) * p.dst_w + dx;
    dst[base + 0 * plane] = r;
    dst[base + 1 * plane] = g;
    dst[base + 2 * plane] = b;
}

// ===========================================================================
// Variant B: straight from global memory (the fallback)
// ===========================================================================
//
// When the camera frame is much larger than the model input, the source window
// a 32x8 output tile needs grows as (32/scale) x (8/scale) and eventually will
// not fit in shared memory. Rather than shrink the tile - which would wreck
// the coalesced 128-byte stores that make the fast path fast - this variant
// drops the cache and reads taps directly from global memory, leaning on L2
// for the reuse instead. Identical arithmetic, identical output; only the
// source of the four taps differs.
__global__ void preprocess_global_kernel(const uint8_t* __restrict__ src,
                                         float* __restrict__ dst,
                                         const LetterboxParams p,
                                         const float norm_scale,
                                         const float pad_value) {
    const int n  = blockIdx.z;
    const int dx = blockIdx.x * kTileW + threadIdx.x;
    const int dy = blockIdx.y * kTileH + threadIdx.y;
    if (dx >= p.dst_w || dy >= p.dst_h) return;

    const uint8_t* src_n =
        src + static_cast<size_t>(n) * p.src_h * p.src_w * 3;

    float r, g, b;
    if (in_padding(p, dx, dy)) {
        r = g = b = pad_value;
    } else {
        const float fx = dst_to_src_x(p, static_cast<float>(dx));
        const float fy = dst_to_src_y(p, static_cast<float>(dy));
        const BilinearWeights w = bilinear_weights(fx, fy, p.src_w, p.src_h);

        const uint8_t* q00 = src_n + (static_cast<size_t>(w.y0) * p.src_w + w.x0) * 3;
        const uint8_t* q10 = src_n + (static_cast<size_t>(w.y0) * p.src_w + w.x1) * 3;
        const uint8_t* q01 = src_n + (static_cast<size_t>(w.y1) * p.src_w + w.x0) * 3;
        const uint8_t* q11 = src_n + (static_cast<size_t>(w.y1) * p.src_w + w.x1) * 3;

        b = blend4(q00[0], q10[0], q01[0], q11[0], w.wx, w.wy) * norm_scale;
        g = blend4(q00[1], q10[1], q01[1], q11[1], w.wx, w.wy) * norm_scale;
        r = blend4(q00[2], q10[2], q01[2], q11[2], w.wx, w.wy) * norm_scale;
    }

    const size_t plane = static_cast<size_t>(p.dst_h) * p.dst_w;
    const size_t base  = static_cast<size_t>(n) * 3 * plane +
                         static_cast<size_t>(dy) * p.dst_w + dx;
    dst[base + 0 * plane] = r;
    dst[base + 1 * plane] = g;
    dst[base + 2 * plane] = b;
}

// ===========================================================================
// Host side
// ===========================================================================

PreprocPlan plan_preprocess(const LetterboxParams& p, int batch, int device) {
    PreprocPlan plan;
    plan.tile_w = kTileW;
    plan.tile_h = kTileH;
    plan.block  = dim3(kTileW, kTileH, 1);
    plan.grid   = dim3((p.dst_w + kTileW - 1) / kTileW,
                       (p.dst_h + kTileH - 1) / kTileH,
                       static_cast<unsigned>(batch < 1 ? 1 : batch));

    plan.win_w = tile_window_extent(kTileW, lb_scale_x(p));
    plan.win_h = tile_window_extent(kTileH, lb_scale_y(p));
    plan.shared_bytes =
        static_cast<size_t>(plan.win_w) * plan.win_h * sizeof(uchar4);

    int limit = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(
        &limit, cudaDevAttrMaxSharedMemoryPerBlock, device));
    plan.shared_limit = static_cast<size_t>(limit);

    // Leave the last 1 KB alone: on some configurations a sliver of the
    // per-block allowance is reserved for driver use, and asking for exactly
    // the attribute value can fail at launch.
    plan.use_shared = plan.shared_bytes + 1024 <= plan.shared_limit;
    return plan;
}

void launch_preprocess(const uint8_t* d_src_bgr,
                       float* d_dst_nchw,
                       const LetterboxParams& p,
                       int batch,
                       const PreprocPlan& plan,
                       float norm_scale,
                       float pad_value,
                       cudaStream_t stream) {
    if (batch < 1) return;

    if (plan.use_shared) {
        preprocess_shared_kernel<<<plan.grid, plan.block, plan.shared_bytes,
                                   stream>>>(
            d_src_bgr, d_dst_nchw, p, plan.win_w, plan.win_h, norm_scale,
            pad_value);
    } else {
        preprocess_global_kernel<<<plan.grid, plan.block, 0, stream>>>(
            d_src_bgr, d_dst_nchw, p, norm_scale, pad_value);
    }

    // Catches launch-configuration errors (bad dims, too much shared memory)
    // at the launch site instead of letting them surface as corrupt tensor
    // data one stage later. Does not synchronise - see cuda_utils.cuh.
    CUDA_CHECK(cudaGetLastError());
}

} // namespace reflex
