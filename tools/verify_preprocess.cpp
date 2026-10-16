// ===========================================================================
// verify_preprocess - does the CUDA kernel actually do what we think?
// ===========================================================================
//
// Runs src/preprocess.cu on a test image and compares the result, pixel by
// pixel, against the same transform built out of OpenCV calls on the CPU.
//
// This is the step the build guide means by "verify correctness by comparing
// against OpenCV's CPU equivalent on a test image before trusting your
// kernel's output". It is worth doing before the kernel is wired into the
// pipeline, because a preprocessing bug does not announce itself: the detector
// still runs, still outputs boxes, and the boxes are just quietly worse. You
// end up blaming the model, the threshold, the lighting - anything but the
// resize.
//
//   build:  cmake --build build --target verify_preprocess
//   run:    ./build/verify_preprocess [image.jpg] [--size 640] [--bench]
//
// With no image argument it generates a synthetic one full of hard edges,
// which is what actually catches half-pixel misalignment - a smooth photo
// hides a one-pixel shift almost completely.
//
// TWO REFERENCES ARE COMPARED, because they fail differently:
//
//   float path  - OpenCV resizing a float image. This is the mathematically
//                 exact answer, so the kernel should match it very closely.
//                 A large error here means the kernel's arithmetic or its
//                 coordinate mapping is wrong.
//
//   uint8 path  - OpenCV resizing the 8-bit image the way a straightforward
//                 CPU implementation would. OpenCV uses fixed-point weights
//                 and rounds to integers here, so a difference of a few parts
//                 in 255 is EXPECTED and is not a bug.
// ===========================================================================

#include <opencv2/opencv.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "cuda_utils.cuh"
#include "preprocess.cuh"

using namespace reflex;

namespace {

// A deliberately hostile test image: sharp edges in every direction, a
// single-pixel grid, and high-frequency checkerboard. Any coordinate error
// shows up as a large difference along the edges.
cv::Mat make_test_image(int w, int h) {
    cv::Mat img(h, w, CV_8UC3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            // Base gradient, different per channel so a channel swap is
            // immediately visible.
            int b = (x * 255) / std::max(1, w - 1);
            int g = (y * 255) / std::max(1, h - 1);
            int r = ((x + y) * 255) / std::max(1, w + h - 2);

            // 1-pixel grid lines.
            if (x % 16 == 0 || y % 16 == 0) { b = 255; g = 255; r = 255; }
            // Fine checkerboard in one quadrant.
            if (x > w / 2 && y > h / 2 && ((x / 2 + y / 2) & 1)) {
                b = 0; g = 0; r = 0;
            }
            img.at<cv::Vec3b>(y, x) =
                cv::Vec3b(static_cast<uchar>(b), static_cast<uchar>(g),
                          static_cast<uchar>(r));
        }
    }
    // A few hard-edged shapes.
    cv::circle(img, {w / 4, h / 4}, std::min(w, h) / 8, {0, 0, 255}, -1);
    cv::rectangle(img, {w / 2 + 10, 10}, {w - 10, h / 4}, {255, 0, 0}, -1);
    cv::line(img, {0, h - 1}, {w - 1, 0}, {0, 255, 0}, 3);
    return img;
}

// The CPU reference: exactly the transform the kernel claims to implement.
// `as_float` selects whether the resize happens in float (exact) or in uint8
// (what OpenCV would really do to a camera frame).
//
// Output is CHW RGB float, normalised - the same layout the kernel writes.
std::vector<float> cpu_reference(const cv::Mat& src, const LetterboxParams& p,
                                 float norm_scale, float pad_value,
                                 bool as_float) {
    cv::Mat work;
    if (as_float) {
        src.convertTo(work, CV_32FC3);
    } else {
        work = src;
    }

    cv::Mat resized;
    cv::resize(work, resized, cv::Size(p.new_w, p.new_h), 0, 0,
               cv::INTER_LINEAR);

    // The kernel's pad_x/pad_y are floats (half of an odd difference is .5),
    // so the integer borders are derived the same way on both sides.
    const int top    = static_cast<int>(p.pad_y);
    const int left   = static_cast<int>(p.pad_x);
    const int bottom = p.dst_h - p.new_h - top;
    const int right  = p.dst_w - p.new_w - left;

    const double pv = static_cast<double>(pad_value) /
                      static_cast<double>(norm_scale);  // back to 0-255 units

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, top, bottom, left, right,
                       cv::BORDER_CONSTANT, cv::Scalar(pv, pv, pv));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    cv::Mat f;
    rgb.convertTo(f, CV_32FC3, static_cast<double>(norm_scale));

    std::vector<cv::Mat> ch;
    cv::split(f, ch);

    const size_t plane = static_cast<size_t>(p.dst_h) * p.dst_w;
    std::vector<float> out(plane * 3);
    for (int c = 0; c < 3; ++c) {
        CV_Assert(ch[c].isContinuous());
        std::memcpy(out.data() + plane * c, ch[c].ptr<float>(),
                    plane * sizeof(float));
    }
    return out;
}

struct Diff {
    double max_abs = 0, mean_abs = 0;
    size_t over_tol = 0;
    int    worst_c = 0, worst_x = 0, worst_y = 0;
};

Diff compare(const std::vector<float>& a, const std::vector<float>& b,
             const LetterboxParams& p, double tol) {
    Diff d;
    const size_t plane = static_cast<size_t>(p.dst_h) * p.dst_w;
    double sum = 0;
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < p.dst_h; ++y) {
            for (int x = 0; x < p.dst_w; ++x) {
                const size_t i = plane * c + static_cast<size_t>(y) * p.dst_w + x;
                const double e = std::fabs(static_cast<double>(a[i]) -
                                           static_cast<double>(b[i]));
                sum += e;
                if (e > d.max_abs) {
                    d.max_abs = e;
                    d.worst_c = c; d.worst_x = x; d.worst_y = y;
                }
                if (e > tol) ++d.over_tol;
            }
        }
    }
    d.mean_abs = sum / static_cast<double>(plane * 3);
    return d;
}

void report(const char* name, const Diff& d, double tol, size_t total) {
    std::printf("  %-12s  max %.6f   mean %.6f   over tol(%.3f): %zu / %zu "
                "(%.4f%%)\n",
                name, d.max_abs, d.mean_abs, tol, d.over_tol, total,
                100.0 * static_cast<double>(d.over_tol) /
                    static_cast<double>(total));
    if (d.over_tol) {
        std::printf("  %-12s  worst at channel %d, pixel (%d,%d)\n", "",
                    d.worst_c, d.worst_x, d.worst_y);
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string image_path;
    int  dst_size = 640;
    bool bench    = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--size" && i + 1 < argc)      dst_size = std::atoi(argv[++i]);
        else if (a == "--bench")                bench = true;
        else if (a == "-h" || a == "--help") {
            std::printf("usage: %s [image] [--size N] [--bench]\n", argv[0]);
            return 0;
        }
        else image_path = a;
    }

    try {
        cv::Mat src;
        if (!image_path.empty()) {
            src = cv::imread(image_path, cv::IMREAD_COLOR);
            if (src.empty()) {
                std::fprintf(stderr, "cannot read image: %s\n",
                             image_path.c_str());
                return 1;
            }
            std::printf("image: %s (%dx%d)\n", image_path.c_str(), src.cols,
                        src.rows);
        } else {
            src = make_test_image(640, 480);
            std::printf("image: synthetic 640x480 (hard edges, 1px grid)\n");
        }

        const float norm_scale = 1.0f / 255.0f;
        const float pad_value  = 114.0f / 255.0f;

        const LetterboxParams p =
            make_letterbox(src.cols, src.rows, dst_size, dst_size);
        std::printf("letterbox: %dx%d -> %dx%d in %dx%d (scale %.4f, pad %.1f,%.1f)\n",
                    p.src_w, p.src_h, p.new_w, p.new_h, p.dst_w, p.dst_h,
                    p.scale, p.pad_x, p.pad_y);

        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::printf("gpu: %s (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);

        // --- device buffers ------------------------------------------------
        const size_t src_bytes = static_cast<size_t>(src.cols) * src.rows * 3;
        const size_t dst_elems = static_cast<size_t>(p.dst_w) * p.dst_h * 3;

        uint8_t* d_src = nullptr;
        float*   d_dst = nullptr;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_src), src_bytes));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_dst),
                              dst_elems * sizeof(float)));

        cv::Mat src_c = src.isContinuous() ? src : src.clone();
        CUDA_CHECK(cudaMemcpy(d_src, src_c.data, src_bytes,
                              cudaMemcpyHostToDevice));

        PreprocPlan plan = plan_preprocess(p, 1, dev);
        std::printf("plan: %s, grid %ux%ux%u, block %ux%u, window %dx%d, "
                    "shmem %zu/%zu B\n\n",
                    plan.use_shared ? "shared-memory tile" : "global fallback",
                    plan.grid.x, plan.grid.y, plan.grid.z, plan.block.x,
                    plan.block.y, plan.win_w, plan.win_h, plan.shared_bytes,
                    plan.shared_limit);

        std::vector<float> gpu(dst_elems);

        // --- run the kernel the pipeline would use -------------------------
        launch_preprocess(d_src, d_dst, p, 1, plan, norm_scale, pad_value,
                          nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(gpu.data(), d_dst, dst_elems * sizeof(float),
                              cudaMemcpyDeviceToHost));

        // --- compare against both CPU references ---------------------------
        const std::vector<float> ref_f =
            cpu_reference(src_c, p, norm_scale, pad_value, true);
        const std::vector<float> ref_u =
            cpu_reference(src_c, p, norm_scale, pad_value, false);

        const double tol_f = 0.002;   // exact-maths reference: should be tight
        const double tol_u = 0.020;   // uint8 reference: fixed-point + rounding

        std::printf("GPU vs CPU reference:\n");
        const Diff df = compare(gpu, ref_f, p, tol_f);
        const Diff du = compare(gpu, ref_u, p, tol_u);
        report("float path", df, tol_f, dst_elems);
        report("uint8 path", du, tol_u, dst_elems);

        // --- the two kernel variants must agree with each other ------------
        PreprocPlan forced = plan;
        forced.use_shared = !plan.use_shared;
        std::vector<float> other(dst_elems);
        launch_preprocess(d_src, d_dst, p, 1, forced, norm_scale, pad_value,
                          nullptr);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(other.data(), d_dst, dst_elems * sizeof(float),
                              cudaMemcpyDeviceToHost));
        const Diff dk = compare(gpu, other, p, 1e-6);
        std::printf("\nshared-memory kernel vs global-memory kernel:\n");
        report("variants", dk, 1e-6, dst_elems);

        // --- optional benchmark --------------------------------------------
        if (bench) {
            std::printf("\nbenchmark (500 iterations each):\n");
            cudaEvent_t a, b;
            CUDA_CHECK(cudaEventCreate(&a));
            CUDA_CHECK(cudaEventCreate(&b));

            for (int variant = 0; variant < 2; ++variant) {
                PreprocPlan pl = plan;
                pl.use_shared = (variant == 0);
                // warm up
                for (int i = 0; i < 20; ++i)
                    launch_preprocess(d_src, d_dst, p, 1, pl, norm_scale,
                                      pad_value, nullptr);
                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaEventRecord(a));
                for (int i = 0; i < 500; ++i)
                    launch_preprocess(d_src, d_dst, p, 1, pl, norm_scale,
                                      pad_value, nullptr);
                CUDA_CHECK(cudaEventRecord(b));
                CUDA_CHECK(cudaEventSynchronize(b));

                float ms = 0;
                CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
                std::printf("  %-22s %.4f ms/frame\n",
                            pl.use_shared ? "shared-memory tile"
                                          : "global memory",
                            ms / 500.0f);
            }
            CUDA_CHECK(cudaEventDestroy(a));
            CUDA_CHECK(cudaEventDestroy(b));
            std::printf("  (these are the before/after numbers for the README's"
                        " shared-memory claim)\n");
        }

        CUDA_CHECK(cudaFree(d_src));
        CUDA_CHECK(cudaFree(d_dst));

        // --- verdict -------------------------------------------------------
        const bool ok = df.max_abs <= tol_f * 8 && du.max_abs <= tol_u * 4 &&
                        dk.max_abs <= 1e-5;
        std::printf("\n%s\n", ok ? "PASS - the kernel matches the CPU reference"
                                 : "FAIL - see the worst-pixel locations above");
        if (!ok) {
            std::printf(
                "\nWhere to look:\n"
                "  * error concentrated on EDGES, roughly uniform elsewhere\n"
                "      -> a half-pixel offset. Check dst_to_src_x/y in\n"
                "         preprocess_math.cuh against cv::resize's convention.\n"
                "  * the R and B channels look swapped\n"
                "      -> the BGR->RGB mapping in the kernel.\n"
                "  * error only in the padding strip\n"
                "      -> pad_value units: the kernel takes it already\n"
                "         normalised (114/255), not as 114.\n"
                "  * error only in the last column/row of each 32x8 block\n"
                "      -> the shared-memory window is too small; run\n"
                "         test_preprocess_math, which checks exactly this.\n"
                "  * the two kernel variants disagree\n"
                "      -> the shared-memory path's local indexing.\n");
        }
        return ok ? 0 : 1;

    } catch (const std::exception& e) {
        std::fprintf(stderr, "\nfatal: %s\n", e.what());
        return 2;
    }
}
