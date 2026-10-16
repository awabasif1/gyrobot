// Tests for src/preprocess_math.cuh - the geometry shared by the CUDA
// preprocessing kernel and the postprocessor.
//
// These run on the CPU with no GPU present: the header is written so that
// every __host__ __device__ function compiles as plain host code. That is the
// point of structuring it that way - this arithmetic is where the subtle bugs
// live (half-pixel offsets, a tile window one pixel too small), and all of it
// is checkable without hardware.
//
// The interesting tests are the last two:
//   * the linear-ramp test, which verifies the interpolation independently of
//     the mapping, using the fact that bilinear interpolation of a linear
//     function is exact;
//   * the tile-coverage test, which replicates the kernel's shared-memory
//     window indexing and proves the window is always big enough. Getting
//     that wrong on the GPU produces subtly wrong pixels in the right-hand
//     column of every block, which is nearly impossible to spot by eye.

#include "../src/preprocess_math.cuh"
#include "test_util.h"

#include <cstdio>
#include <vector>

using namespace reflex;

// ===========================================================================
static void test_letterbox_params() {
    SECTION("make_letterbox");

    // The real case: VGA from the ESP32-CAM into a 640x640 model input.
    {
        const LetterboxParams p = make_letterbox(640, 480, 640, 640);
        CHECK_NEAR(p.scale, 1.0, 1e-6);
        CHECK(p.new_w == 640);
        CHECK(p.new_h == 480);
        CHECK_NEAR(p.pad_x, 0.0, 1e-6);
        CHECK_NEAR(p.pad_y, 80.0, 1e-6);   // (640-480)/2
    }

    // Downscaling, width-limited.
    {
        const LetterboxParams p = make_letterbox(1024, 768, 640, 640);
        CHECK_NEAR(p.scale, 640.0 / 1024.0, 1e-6);
        CHECK(p.new_w == 640);
        CHECK(p.new_h == 480);
        CHECK_NEAR(p.pad_y, 80.0, 1e-6);
    }

    // Portrait source: height-limited, padding goes on the sides.
    {
        const LetterboxParams p = make_letterbox(480, 640, 640, 640);
        CHECK(p.new_w == 480);
        CHECK(p.new_h == 640);
        CHECK_NEAR(p.pad_x, 80.0, 1e-6);
        CHECK_NEAR(p.pad_y, 0.0, 1e-6);
    }

    // Square source needs no padding at all.
    {
        const LetterboxParams p = make_letterbox(320, 320, 640, 640);
        CHECK(p.new_w == 640 && p.new_h == 640);
        CHECK_NEAR(p.pad_x, 0.0, 1e-6);
        CHECK_NEAR(p.pad_y, 0.0, 1e-6);
    }

    // The scaled size must never exceed the destination, even when rounding
    // would push it over.
    {
        const LetterboxParams p = make_letterbox(639, 479, 640, 640);
        CHECK(p.new_w <= 640);
        CHECK(p.new_h <= 640);
    }
}

// ===========================================================================
static void test_mapping_roundtrip() {
    SECTION("dst<->src mapping is exactly invertible");

    const int cases[][4] = {
        {640, 480, 640, 640},
        {1024, 768, 640, 640},
        {480, 640, 640, 640},
        {320, 240, 416, 416},
        {800, 600, 1280, 1280},
    };

    for (const auto& c : cases) {
        const LetterboxParams p = make_letterbox(c[0], c[1], c[2], c[3]);
        for (float v = -5.f; v < 700.f; v += 7.3f) {
            // src -> dst -> src must return the original value. If these two
            // functions ever disagree, detections land offset from the pixels
            // they were computed from.
            CHECK_NEAR(dst_to_src_x(p, src_to_dst_x(p, v)), v, 1e-3);
            CHECK_NEAR(dst_to_src_y(p, src_to_dst_y(p, v)), v, 1e-3);
        }
    }

    // And the specific values for the VGA case, worked out by hand:
    // pad_y = 80, scale 1.0, so model y maps to source y - 80.
    const LetterboxParams p = make_letterbox(640, 480, 640, 640);
    CHECK_NEAR(dst_to_src_x(p, 0.f), 0.0, 1e-5);
    CHECK_NEAR(dst_to_src_x(p, 320.f), 320.0, 1e-5);
    CHECK_NEAR(dst_to_src_y(p, 80.f), 0.0, 1e-5);
    CHECK_NEAR(dst_to_src_y(p, 320.f), 240.0, 1e-5);   // frame centre
    CHECK_NEAR(dst_to_src_y(p, 559.f), 479.0, 1e-5);   // last real row
}

// ===========================================================================
static void test_padding_region() {
    SECTION("in_padding");

    const LetterboxParams p = make_letterbox(640, 480, 640, 640);

    // Top and bottom 80 rows are padding; everything between is image.
    CHECK(in_padding(p, 0, 0));
    CHECK(in_padding(p, 320, 79));
    CHECK(!in_padding(p, 320, 80));
    CHECK(!in_padding(p, 320, 559));
    CHECK(in_padding(p, 320, 560));
    CHECK(in_padding(p, 320, 639));

    // No horizontal padding in this case.
    CHECK(!in_padding(p, 0, 320));
    CHECK(!in_padding(p, 639, 320));

    // Portrait source: padding on the sides instead.
    const LetterboxParams q = make_letterbox(480, 640, 640, 640);
    CHECK(in_padding(q, 0, 320));
    CHECK(in_padding(q, 79, 320));
    CHECK(!in_padding(q, 80, 320));
    CHECK(!in_padding(q, 559, 320));
    CHECK(in_padding(q, 560, 320));

    // Count the non-padding pixels: must equal new_w * new_h exactly. A
    // half-pixel error in the boundary test shows up here as an off-by-640.
    int count = 0;
    for (int y = 0; y < p.dst_h; ++y)
        for (int x = 0; x < p.dst_w; ++x)
            if (!in_padding(p, x, y)) ++count;
    CHECK(count == p.new_w * p.new_h);
}

// ===========================================================================
static void test_bilinear_weights() {
    SECTION("bilinear_weights");

    // Exactly on a pixel centre: weights 0, taps are that pixel and the next.
    {
        const BilinearWeights b = bilinear_weights(5.0f, 7.0f, 100, 100);
        CHECK(b.x0 == 5 && b.x1 == 6);
        CHECK(b.y0 == 7 && b.y1 == 8);
        CHECK_NEAR(b.wx, 0.0, 1e-6);
        CHECK_NEAR(b.wy, 0.0, 1e-6);
    }

    // Halfway between two pixels.
    {
        const BilinearWeights b = bilinear_weights(5.5f, 7.25f, 100, 100);
        CHECK(b.x0 == 5 && b.x1 == 6);
        CHECK_NEAR(b.wx, 0.5, 1e-6);
        CHECK_NEAR(b.wy, 0.25, 1e-6);
    }

    // Past the left/top edge: clamped to the first pixel (replicate border).
    {
        const BilinearWeights b = bilinear_weights(-0.4f, -0.4f, 100, 100);
        CHECK(b.x0 == 0 && b.y0 == 0);
        CHECK(b.x1 == 0 && b.y1 == 0);
    }

    // Past the right/bottom edge: clamped to the last pixel.
    {
        const BilinearWeights b = bilinear_weights(99.7f, 99.7f, 100, 100);
        CHECK(b.x0 == 99 && b.x1 == 99);
        CHECK(b.y0 == 99 && b.y1 == 99);
    }

    // A 1x1 image must not produce an out-of-range tap.
    {
        const BilinearWeights b = bilinear_weights(0.5f, 0.5f, 1, 1);
        CHECK(b.x0 == 0 && b.x1 == 0 && b.y0 == 0 && b.y1 == 0);
    }
}

// ===========================================================================
static void test_blend4() {
    SECTION("blend4");

    // Corners.
    CHECK_NEAR(blend4(10, 20, 30, 40, 0.f, 0.f), 10.0, 1e-6);
    CHECK_NEAR(blend4(10, 20, 30, 40, 1.f, 0.f), 20.0, 1e-6);
    CHECK_NEAR(blend4(10, 20, 30, 40, 0.f, 1.f), 30.0, 1e-6);
    CHECK_NEAR(blend4(10, 20, 30, 40, 1.f, 1.f), 40.0, 1e-6);

    // Centre is the mean of all four.
    CHECK_NEAR(blend4(10, 20, 30, 40, 0.5f, 0.5f), 25.0, 1e-6);

    // Constant input gives the constant back, whatever the weights.
    for (float wx = 0.f; wx <= 1.f; wx += 0.33f)
        for (float wy = 0.f; wy <= 1.f; wy += 0.33f)
            CHECK_NEAR(blend4(7, 7, 7, 7, wx, wy), 7.0, 1e-6);

    // bilinear_sample must agree with blend4 - they are the same formula and
    // the kernel relies on that.
    struct Fetch {
        float operator()(int x, int y) const {
            return static_cast<float>(10 * y + x);
        }
    };
    const BilinearWeights b = bilinear_weights(2.5f, 3.5f, 100, 100);
    CHECK_NEAR(bilinear_sample(Fetch{}, b),
               blend4(Fetch{}(b.x0, b.y0), Fetch{}(b.x1, b.y0),
                      Fetch{}(b.x0, b.y1), Fetch{}(b.x1, b.y1), b.wx, b.wy),
               1e-6);
}

// ===========================================================================
// Bilinear interpolation reproduces a linear function EXACTLY. So if the
// source image is a linear ramp, every interpolated output pixel must equal
// the ramp evaluated at the mapped source coordinate - which checks the
// interpolation and the mapping together, against an independent closed-form
// answer rather than against another copy of the same code.
static void test_linear_ramp_exactness() {
    SECTION("linear ramp is interpolated exactly");

    const int src_w = 640, src_h = 480;
    const LetterboxParams p = make_letterbox(src_w, src_h, 640, 640);

    // f(x, y) = 0.3x + 0.7y + 5
    const auto ramp = [](float x, float y) { return 0.3f * x + 0.7f * y + 5.f; };

    struct Fetch {
        float a, b, c;
        float operator()(int x, int y) const {
            return a * static_cast<float>(x) + b * static_cast<float>(y) + c;
        }
    };
    const Fetch fetch{0.3f, 0.7f, 5.f};

    int checked = 0;
    for (int dy = 0; dy < p.dst_h; dy += 11) {
        for (int dx = 0; dx < p.dst_w; dx += 13) {
            if (in_padding(p, dx, dy)) continue;

            const float fx = dst_to_src_x(p, static_cast<float>(dx));
            const float fy = dst_to_src_y(p, static_cast<float>(dy));

            // Skip the outermost half-pixel, where coordinate clamping
            // intentionally breaks linearity (it replicates the edge pixel).
            if (fx < 0.f || fy < 0.f ||
                fx > static_cast<float>(src_w - 1) ||
                fy > static_cast<float>(src_h - 1)) continue;

            const BilinearWeights b = bilinear_weights(fx, fy, src_w, src_h);
            const float got  = bilinear_sample(fetch, b);
            const float want = ramp(fx, fy);
            CHECK_NEAR(got, want, 2e-3);
            ++checked;
        }
    }
    std::printf("    checked %d interpolated pixels against the closed form\n",
                checked);
    CHECK(checked > 1000);
}

// ===========================================================================
// Replicates EXACTLY what preprocess_shared_kernel does when it computes its
// shared-memory window, and proves the window always contains every tap any
// thread in that block will read.
//
// If tile_window_extent() is ever one pixel too small, the kernel's defensive
// clamp silently substitutes an edge pixel and the right-hand column of every
// block is subtly wrong. On a GPU that is invisible; here it is a test
// failure.
static void test_tile_window_covers_all_taps() {
    SECTION("shared-memory tile window covers every bilinear tap");

    const int kTileW = 32, kTileH = 8;   // must match preprocess.cu

    const int cases[][4] = {
        {640, 480, 640, 640},     // VGA, scale 1.0
        {800, 600, 640, 640},     // SVGA, scale 0.8
        {1024, 768, 640, 640},    // XGA, scale 0.625
        {320, 240, 640, 640},     // upscaling, scale 2.0
        {1600, 1200, 640, 640},   // heavy downscale, scale 0.4
        {640, 480, 416, 416},     // a smaller model input
    };

    for (const auto& c : cases) {
        const LetterboxParams p = make_letterbox(c[0], c[1], c[2], c[3]);
        const int win_w = tile_window_extent(kTileW, lb_scale_x(p));
        const int win_h = tile_window_extent(kTileH, lb_scale_y(p));

        int worst_x = 0, worst_y = 0;

        for (int ty = 0; ty < p.dst_h; ty += kTileH) {
            for (int tx = 0; tx < p.dst_w; tx += kTileW) {
                // The kernel's window origin.
                const int win_x0 = static_cast<int>(
                    floorf(dst_to_src_x(p, static_cast<float>(tx))));
                const int win_y0 = static_cast<int>(
                    floorf(dst_to_src_y(p, static_cast<float>(ty))));

                for (int dy = ty; dy < ty + kTileH && dy < p.dst_h; ++dy) {
                    for (int dx = tx; dx < tx + kTileW && dx < p.dst_w; ++dx) {
                        if (in_padding(p, dx, dy)) continue;

                        const float fx = dst_to_src_x(p, static_cast<float>(dx));
                        const float fy = dst_to_src_y(p, static_cast<float>(dy));
                        const BilinearWeights b =
                            bilinear_weights(fx, fy, p.src_w, p.src_h);

                        // Every tap, in window-local coordinates, must be in
                        // range. Taps are already clamped to the image, so the
                        // only way to fail is a window that is too small.
                        const int lx0 = b.x0 - win_x0, lx1 = b.x1 - win_x0;
                        const int ly0 = b.y0 - win_y0, ly1 = b.y1 - win_y0;

                        CHECK(lx0 >= 0 && lx0 < win_w);
                        CHECK(lx1 >= 0 && lx1 < win_w);
                        CHECK(ly0 >= 0 && ly0 < win_h);
                        CHECK(ly1 >= 0 && ly1 < win_h);

                        if (lx1 + 1 > worst_x) worst_x = lx1 + 1;
                        if (ly1 + 1 > worst_y) worst_y = ly1 + 1;
                    }
                }
            }
        }

        const size_t bytes = static_cast<size_t>(win_w) * win_h * 4;
        std::printf("    %4dx%-4d -> %dx%d : window %dx%d (used %dx%d), "
                    "%zu B shared\n",
                    c[0], c[1], c[2], c[3], win_w, win_h, worst_x, worst_y,
                    bytes);

        // The window should be snug, not wildly oversized - shared memory is
        // what caps occupancy.
        CHECK(win_w - worst_x <= 3);
        CHECK(win_h - worst_y <= 3);
    }
}

// ===========================================================================
static void test_tile_window_extent_monotonic() {
    SECTION("tile_window_extent");

    // At scale 1 a 32-wide tile needs 32 source columns plus the bilinear and
    // floor slack.
    CHECK(tile_window_extent(32, 1.0f) == 34);
    // Halving the scale doubles the source span needed.
    CHECK(tile_window_extent(32, 0.5f) == 66);
    // Upscaling needs fewer source pixels than output pixels.
    CHECK(tile_window_extent(32, 2.0f) == 18);
    // A degenerate scale must not divide by zero or return something negative.
    CHECK(tile_window_extent(32, 0.0f) > 0);

    // Monotonic in scale.
    int prev = tile_window_extent(32, 0.1f);
    for (float s = 0.2f; s < 4.0f; s += 0.1f) {
        const int cur = tile_window_extent(32, s);
        CHECK(cur <= prev);
        prev = cur;
    }
}

// ===========================================================================
int main() {
    std::printf("test_preprocess_math\n");
    test_letterbox_params();
    test_mapping_roundtrip();
    test_padding_region();
    test_bilinear_weights();
    test_blend4();
    test_linear_ramp_exactness();
    test_tile_window_covers_all_taps();
    test_tile_window_extent_monotonic();
    return test_summary("test_preprocess_math");
}
