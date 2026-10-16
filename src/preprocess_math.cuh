#pragma once
// ---------------------------------------------------------------------------
// Shared geometry math for preprocessing AND postprocessing.
//
// WHY THIS FILE EXISTS AT ALL:
// The preprocessing kernel squeezes a camera frame into the model's square
// input by scaling it and padding the leftover strip ("letterboxing"). The
// postprocessor has to undo exactly that transform to turn the model's box
// coordinates back into camera-frame pixels. If those two pieces of code
// disagree about the mapping by even half a pixel, every detection lands
// slightly off-centre, the turn command is biased, and the robot steers to one
// side of the target forever. That class of bug is miserable to find on real
// hardware, so the mapping is defined ONCE, here, and both sides call it.
//
// It compiles as both host and device code: the kernel uses it on the GPU, and
// test/test_preprocess_math.cpp uses it on the CPU to prove the forward and
// inverse maps really are inverses. That is also why the helpers avoid
// std::min/std::max and <algorithm> - those are host-only.
// ---------------------------------------------------------------------------

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define RCV_HD __host__ __device__
#else
#define RCV_HD
#endif

namespace reflex {

// The result of fitting a src_w x src_h image into a dst_w x dst_h square.
//
// Convention (matches Ultralytics' letterbox(), which is what YOLOv8 was
// trained with): scale by the SMALLER of the two ratios so nothing is cropped,
// then centre the scaled image and pad both sides equally.
struct LetterboxParams {
    float scale;        // uniform scale factor applied to the source
    int   new_w, new_h; // size of the scaled image, before padding
    float pad_x, pad_y; // padding on the left / top edge, in dst pixels
    int   src_w, src_h; // original frame size
    int   dst_w, dst_h; // model input size
};

RCV_HD inline float rcv_minf(float a, float b) { return a < b ? a : b; }
RCV_HD inline float rcv_maxf(float a, float b) { return a > b ? a : b; }
RCV_HD inline float rcv_clampf(float v, float lo, float hi) {
    return rcv_minf(rcv_maxf(v, lo), hi);
}
RCV_HD inline int rcv_clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

RCV_HD inline LetterboxParams make_letterbox(int src_w, int src_h,
                                             int dst_w, int dst_h) {
    LetterboxParams p;
    p.src_w = src_w; p.src_h = src_h;
    p.dst_w = dst_w; p.dst_h = dst_h;

    p.scale = rcv_minf(static_cast<float>(dst_w) / static_cast<float>(src_w),
                       static_cast<float>(dst_h) / static_cast<float>(src_h));

    // Round-half-up rather than lrintf(): identical for the positive values we
    // ever pass here, and it keeps this header free of libm linkage quirks on
    // the host side.
    p.new_w = static_cast<int>(src_w * p.scale + 0.5f);
    p.new_h = static_cast<int>(src_h * p.scale + 0.5f);
    if (p.new_w > dst_w) p.new_w = dst_w;
    if (p.new_h > dst_h) p.new_h = dst_h;
    if (p.new_w < 1) p.new_w = 1;
    if (p.new_h < 1) p.new_h = 1;

    p.pad_x = (dst_w - p.new_w) * 0.5f;
    p.pad_y = (dst_h - p.new_h) * 0.5f;
    return p;
}

// The *effective* scale actually realised after new_w/new_h were rounded to
// whole pixels. Using this rather than p.scale is what keeps the forward and
// inverse maps exact inverses of each other.
RCV_HD inline float lb_scale_x(const LetterboxParams& p) {
    return static_cast<float>(p.new_w) / static_cast<float>(p.src_w);
}
RCV_HD inline float lb_scale_y(const LetterboxParams& p) {
    return static_cast<float>(p.new_h) / static_cast<float>(p.src_h);
}

// --- dst (model space) -> src (camera frame space) -------------------------
//
// This is the map the resize kernel samples along, and the one the
// postprocessor uses to put boxes back on the original frame.
//
// The +/- 0.5f terms are the "half-pixel centre" convention: pixel i covers
// the interval [i, i+1), so its centre is at i + 0.5. OpenCV's INTER_LINEAR
// uses the same convention, which is what makes tools/verify_preprocess.cpp
// able to match cv::resize to within a rounding step instead of being off by
// half a pixel everywhere.
RCV_HD inline float dst_to_src_x(const LetterboxParams& p, float dx) {
    return (dx + 0.5f - p.pad_x) / lb_scale_x(p) - 0.5f;
}
RCV_HD inline float dst_to_src_y(const LetterboxParams& p, float dy) {
    return (dy + 0.5f - p.pad_y) / lb_scale_y(p) - 0.5f;
}

// --- src -> dst, the exact inverse of the two functions above ---------------
RCV_HD inline float src_to_dst_x(const LetterboxParams& p, float sx) {
    return (sx + 0.5f) * lb_scale_x(p) - 0.5f + p.pad_x;
}
RCV_HD inline float src_to_dst_y(const LetterboxParams& p, float sy) {
    return (sy + 0.5f) * lb_scale_y(p) - 0.5f + p.pad_y;
}

// True when a model-space pixel falls in the grey padding strip rather than on
// real image content. Those pixels are filled with a constant instead of being
// sampled, because sampling there would smear the edge pixels of the frame
// across the padding and give the detector a fake gradient to chew on.
RCV_HD inline bool in_padding(const LetterboxParams& p, int dx, int dy) {
    float fx = static_cast<float>(dx) + 0.5f;
    float fy = static_cast<float>(dy) + 0.5f;
    return fx < p.pad_x || fx >= p.pad_x + static_cast<float>(p.new_w) ||
           fy < p.pad_y || fy >= p.pad_y + static_cast<float>(p.new_h);
}

// --- bilinear interpolation -------------------------------------------------

// The four source pixels and two blend weights needed for one output pixel.
// Coordinates are clamped to the image (replicate-border), matching what
// OpenCV does at the frame edge.
struct BilinearWeights {
    int   x0, x1, y0, y1;
    float wx, wy;
};

RCV_HD inline BilinearWeights bilinear_weights(float fx, float fy,
                                               int w, int h) {
    BilinearWeights b;
    int ix = static_cast<int>(floorf(fx));
    int iy = static_cast<int>(floorf(fy));
    b.wx = fx - static_cast<float>(ix);
    b.wy = fy - static_cast<float>(iy);
    b.x0 = rcv_clampi(ix,     0, w - 1);
    b.x1 = rcv_clampi(ix + 1, 0, w - 1);
    b.y0 = rcv_clampi(iy,     0, h - 1);
    b.y1 = rcv_clampi(iy + 1, 0, h - 1);
    return b;
}

// The actual interpolation arithmetic, split out so the GPU kernel and the
// host-side test are provably running the same formula. The kernel calls this
// directly (it loads all four taps in one go for all three colour channels,
// rather than paying for four separate loads per channel).
RCV_HD inline float blend4(float v00, float v10, float v01, float v11,
                           float wx, float wy) {
    float top = v00 + (v10 - v00) * wx;
    float bot = v01 + (v11 - v01) * wx;
    return top + (bot - top) * wy;
}

// Fetch is any callable with signature float(int x, int y). A plain struct with
// operator() is used rather than a lambda so this works in device code without
// needing nvcc's --extended-lambda.
template <typename Fetch>
RCV_HD inline float bilinear_sample(Fetch fetch, const BilinearWeights& b) {
    return blend4(fetch(b.x0, b.y0), fetch(b.x1, b.y0),
                  fetch(b.x0, b.y1), fetch(b.x1, b.y1), b.wx, b.wy);
}

// --- shared-memory tile sizing ---------------------------------------------
//
// Called on the HOST to size the dynamic shared-memory allocation, and the
// result is passed into the kernel so every block loads a window of identical
// dimensions. Deriving it in one place removes any chance of the host
// allocating N bytes while the kernel indexes N+something.
//
// A block covering tile_w output columns spans (tile_w - 1)/scale source
// columns between its first and last pixel centre, plus 1 more for the right
// neighbour each bilinear tap needs, plus 1 for the floor() slack when the
// window origin is computed. Rounding up covers all three.
RCV_HD inline int tile_window_extent(int tile_extent, float scale) {
    float span = static_cast<float>(tile_extent) / rcv_maxf(scale, 1e-6f);
    return static_cast<int>(ceilf(span)) + 2;
}

} // namespace reflex
