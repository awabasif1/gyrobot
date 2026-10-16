// ---------------------------------------------------------------------------
// reflex-cv - vision pipeline entry point (runs on the laptop)
//
//   ESP32-CAM --WiFi/MJPEG--> [ capture | upload | preprocess | infer |
//                               decode | shape ] --WiFi/HTTP--> ESP32-CAM
//                                                              --UART--> Mega
//
// Read this file top to bottom to understand the runtime; read the other files
// for the detail of each stage. The ordering below is the dataflow order.
// ---------------------------------------------------------------------------

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>

#include "cuda_utils.cuh"
#include "host_profiler.h"
#include "stage_profiler.cuh"

#include "command_shaper.h"
#include "esp32_client.h"
#include "postprocess.h"
#include "preprocess.cuh"
#include "trt_infer.h"

using Clock = std::chrono::steady_clock;

// ===========================================================================
// Shutdown
// ===========================================================================
// Ctrl-C must not just kill the process: the Mega holds the last command it
// received for as long as it is powered, so exiting while a turn command is in
// flight leaves the robot spinning. The handler only flips a flag; the actual
// "send zeroes" happens on the normal exit path in main().
namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }
} // namespace

// ===========================================================================
// Options
// ===========================================================================
struct Options {
    std::string stream   = "http://192.168.4.1/stream";
    std::string engine   = "models/model.engine";
    std::string esp32;            // defaults to the stream's host:80
    std::string csv_path;         // empty => no CSV dump
    int   target_class = 0;       // COCO: 0 person, 32 sports ball
    float conf         = 0.35f;
    float iou          = 0.45f;
    int   batch        = 1;
    double send_hz     = 20.0;
    double fps_limit   = 0.0;     // 0 => unlimited
    int   max_frames   = 0;       // 0 => until Ctrl-C
    bool  headless     = false;
    bool  no_send      = false;
    bool  no_profile   = false;
    bool  verbose      = false;
};

static void usage(const char* argv0) {
    std::cout <<
R"(reflex-cv vision pipeline

usage: )" << argv0 << R"( [options]

  --stream URL      MJPEG stream from the ESP32-CAM
                    (default http://192.168.4.1/stream)
  --engine PATH     TensorRT engine built on THIS machine
                    (default models/model.engine)
  --esp32 HOST[:PORT]  command endpoint; defaults to the stream's host, port 80
  --class N         COCO class to follow: 0=person, 32=sports ball (default 0)
  --conf F          detection confidence threshold (default 0.35)
  --iou F           NMS IoU threshold (default 0.45)
  --batch N         frames per inference call (default 1; see note below)
  --send-hz F       max command rate to the robot (default 20)
  --fps-limit F     cap pipeline rate to save GPU (default unlimited)
  --max-frames N    stop after N frames (useful for benchmarking)
  --csv PATH        write per-frame stage timings as CSV on exit
  --headless        no display window (for running over SSH)
  --no-send         shape and print commands but never transmit them.
                    USE THIS FIRST: it exercises the whole pipeline with the
                    robot switched off.
  --no-profile      skip per-stage CUDA event timing (each stage's stop()
                    synchronises, so profiling costs a little throughput)
  --verbose         include TensorRT info-level logging
  -h, --help        this message

note on --batch: batching raises throughput but costs latency, because the
first frame of a batch waits for the last one to arrive. For a single camera
driving a balance loop, latency is what matters, so the default is 1. Larger
values exist to demonstrate and measure the tradeoff.
)";
}

static bool parse_args(int argc, char** argv, Options* o) {
    const auto need = [&](int i) -> bool {
        if (i + 1 >= argc) {
            std::cerr << "error: " << argv[i] << " needs a value\n";
            return false;
        }
        return true;
    };
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "-h" || a == "--help")        { usage(argv[0]); return false; }
        else if (a == "--stream")     { if (!need(i)) return false; o->stream = argv[++i]; }
        else if (a == "--engine")     { if (!need(i)) return false; o->engine = argv[++i]; }
        else if (a == "--esp32")      { if (!need(i)) return false; o->esp32 = argv[++i]; }
        else if (a == "--csv")        { if (!need(i)) return false; o->csv_path = argv[++i]; }
        else if (a == "--class")      { if (!need(i)) return false; o->target_class = std::atoi(argv[++i]); }
        else if (a == "--conf")       { if (!need(i)) return false; o->conf = std::stof(argv[++i]); }
        else if (a == "--iou")        { if (!need(i)) return false; o->iou = std::stof(argv[++i]); }
        else if (a == "--batch")      { if (!need(i)) return false; o->batch = std::max(1, std::atoi(argv[++i])); }
        else if (a == "--send-hz")    { if (!need(i)) return false; o->send_hz = std::stod(argv[++i]); }
        else if (a == "--fps-limit")  { if (!need(i)) return false; o->fps_limit = std::stod(argv[++i]); }
        else if (a == "--max-frames") { if (!need(i)) return false; o->max_frames = std::atoi(argv[++i]); }
        else if (a == "--headless")   { o->headless = true; }
        else if (a == "--no-send")    { o->no_send = true; }
        else if (a == "--no-profile") { o->no_profile = true; }
        else if (a == "--verbose")    { o->verbose = true; }
        else {
            std::cerr << "error: unknown argument '" << a << "'\n";
            usage(argv[0]);
            return false;
        }
    }
    return true;
}

// ===========================================================================
// Stage 1: capture
// ===========================================================================
// Why this needs its own thread:
//
// cv::VideoCapture on an HTTP MJPEG stream buffers frames internally. If the
// consumer is slower than the camera for even a moment, read() keeps returning
// progressively older frames and end-to-end latency climbs without bound - the
// robot ends up steering toward where the target was several seconds ago, and
// no amount of GPU optimisation fixes it because the delay is in a queue, not
// in compute.
//
// So this thread reads as fast as the stream delivers and keeps exactly ONE
// frame: the newest. Frames the pipeline was too busy to look at are counted
// and discarded. Dropping frames is the correct behaviour for a control loop;
// queueing them is not.
class FrameGrabber {
public:
    explicit FrameGrabber(std::string url) : url_(std::move(url)) {}
    ~FrameGrabber() { stop(); }

    bool open() {
        if (!cap_.open(url_, cv::CAP_FFMPEG)) return false;
        // Honoured by some backends only; harmless where it is not.
        cap_.set(cv::CAP_PROP_BUFFERSIZE, 1);
        return cap_.isOpened();
    }

    void start() {
        running_.store(true);
        thread_ = std::thread(&FrameGrabber::worker, this);
    }

    void stop() {
        if (!running_.exchange(false)) return;
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
        cap_.release();
    }

    // Waits up to timeout_s for a frame newer than the last one returned.
    bool latest(cv::Mat* out, double timeout_s) {
        std::unique_lock<std::mutex> lk(mu_);
        if (!cv_.wait_for(lk, std::chrono::duration<double>(timeout_s),
                          [this] { return has_new_ || !running_.load(); })) {
            return false;
        }
        if (!has_new_) return false;
        // Hand over the buffer rather than copying it: the grabber writes into
        // a fresh Mat each iteration, so there is no aliasing to worry about.
        *out     = buf_;
        has_new_ = false;
        return !out->empty();
    }

    unsigned long captured() const { return captured_.load(); }
    unsigned long dropped()  const { return dropped_.load(); }
    bool          alive()    const { return running_.load() && !failed_.load(); }

private:
    void worker() {
        while (running_.load()) {
            cv::Mat frame;
            if (!cap_.read(frame) || frame.empty()) {
                // A dropped WiFi connection or an ESP32 reboot lands here.
                // Back off briefly and retry rather than spinning on a dead
                // socket; if the stream never comes back, main() notices via
                // its own frame timeout.
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                if (!cap_.isOpened()) {
                    failed_.store(true);
                    if (!cap_.open(url_, cv::CAP_FFMPEG)) {
                        std::this_thread::sleep_for(std::chrono::milliseconds(500));
                        continue;
                    }
                    failed_.store(false);
                }
                continue;
            }
            captured_.fetch_add(1);
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (has_new_) dropped_.fetch_add(1);  // overwritten unseen
                buf_     = std::move(frame);
                has_new_ = true;
            }
            cv_.notify_one();
        }
    }

    std::string              url_;
    cv::VideoCapture         cap_;
    std::thread              thread_;
    std::mutex               mu_;
    std::condition_variable  cv_;
    std::atomic<bool>        running_{false};
    std::atomic<bool>        failed_{false};
    cv::Mat                  buf_;
    bool                     has_new_ = false;
    std::atomic<unsigned long> captured_{0};
    std::atomic<unsigned long> dropped_{0};
};

// ===========================================================================
// Output layout inference
// ===========================================================================
// The engine is the source of truth for the output shape; hard-coding
// 84x8400 would break the moment someone retrains on a different class count.
static reflex::OutputLayout infer_layout(const nvinfer1::Dims& d) {
    reflex::OutputLayout L;
    if (d.nbDims != 3) {
        throw std::runtime_error(
            "unsupported detector output rank " + std::to_string(d.nbDims) +
            " " + reflex::dims_to_string(d) +
            "\n  Expected a 3-D YOLOv8-style tensor, either [1,84,8400] or "
            "[1,8400,84].");
    }
    const long long a = d.d[1], b = d.d[2];
    // Attributes (84ish) are always far fewer than anchors (8400ish), which is
    // what lets the layout be detected instead of configured.
    if (a <= b) {
        L.num_attrs = static_cast<int>(a);
        L.num_anchors = static_cast<int>(b);
        L.attr_major = true;
    } else {
        L.num_attrs = static_cast<int>(b);
        L.num_anchors = static_cast<int>(a);
        L.attr_major = false;
    }
    if (L.num_attrs <= 4) {
        throw std::runtime_error("detector output has " +
                                 std::to_string(L.num_attrs) +
                                 " attributes; need >4 (4 box + >=1 class)");
    }
    return L;
}

// ===========================================================================
// Overlay
// ===========================================================================
static void draw_overlay(cv::Mat& img,
                         const std::vector<reflex::Detection>& dets,
                         int target_idx,
                         const reflex::Command& cmd,
                         double fps,
                         const reflex::LinkStats& link,
                         const reflex::Telemetry& telem) {
    for (size_t i = 0; i < dets.size(); ++i) {
        const auto& d = dets[i];
        const bool is_target = (static_cast<int>(i) == target_idx);
        const cv::Scalar col = is_target ? cv::Scalar(0, 255, 0)
                                         : cv::Scalar(120, 120, 120);
        cv::rectangle(img, cv::Point2f(d.x1, d.y1), cv::Point2f(d.x2, d.y2),
                      col, is_target ? 2 : 1);
        if (is_target) {
            std::ostringstream lbl;
            lbl << std::fixed << std::setprecision(2) << d.score;
            cv::putText(img, lbl.str(), cv::Point2f(d.x1, d.y1 - 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, col, 1);
        }
    }

    // Frame centre reference, so the turn command can be eyeballed against the
    // target's actual offset.
    const int cx = img.cols / 2;
    cv::line(img, {cx, 0}, {cx, img.rows}, cv::Scalar(60, 60, 60), 1);

    // Turn command as a bar growing from the centre line.
    const int bar_y = img.rows - 18;
    const int bar_px = static_cast<int>(cmd.turn * (img.cols * 0.45));
    cv::line(img, {cx, bar_y}, {cx + bar_px, bar_y}, cv::Scalar(0, 200, 255), 6);

    std::ostringstream hud;
    hud << std::fixed << std::setprecision(1) << fps << " fps  lean "
        << std::setprecision(2) << cmd.lean_deg << "deg  turn " << cmd.turn
        << (cmd.target_valid ? "  [target]" : "  [no target]");
    cv::putText(img, hud.str(), {8, 20}, cv::FONT_HERSHEY_SIMPLEX, 0.55,
                cv::Scalar(255, 255, 255), 1);

    std::ostringstream net;
    net << "link " << (link.connected ? "up" : "DOWN") << "  sent "
        << link.sent << "  drop " << link.dropped << "  fail " << link.failed
        << "  rtt " << std::setprecision(1) << link.last_rtt_ms << "ms";
    if (telem.valid) {
        net << "  | mega " << std::setprecision(1) << telem.angle_deg << "deg"
            << (telem.fallen ? " FALLEN" : "");
    }
    cv::putText(img, net.str(), {8, 40}, cv::FONT_HERSHEY_SIMPLEX, 0.45,
                cv::Scalar(200, 200, 200), 1);
}

// ===========================================================================
// main
// ===========================================================================
int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) return 1;

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    // Work out where commands go. Defaulting to the stream's host means the
    // common case needs one flag, not two.
    std::string cmd_host;
    int cmd_port = 80;
    {
        reflex::HttpUrl u;
        if (!opt.esp32.empty()) {
            reflex::HttpUrl tmp;
            if (reflex::parse_http_url("http://" + opt.esp32, &tmp)) {
                cmd_host = tmp.host;
                cmd_port = tmp.port;
            } else {
                std::cerr << "error: --esp32 '" << opt.esp32
                          << "' is not host[:port]\n";
                return 1;
            }
        } else if (reflex::parse_http_url(opt.stream, &u)) {
            cmd_host = u.host;
            cmd_port = 80;
        } else {
            std::cerr << "error: --stream '" << opt.stream
                      << "' is not a http://host[:port]/path URL, and no "
                         "--esp32 was given\n";
            return 1;
        }
    }

    try {
        // --- GPU + engine -------------------------------------------------
        int dev = 0;
        CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::cout << "GPU: " << prop.name << "  sm_" << prop.major
                  << prop.minor << "  " << (prop.totalGlobalMem >> 20)
                  << " MiB\n";

        std::cout << "loading engine: " << opt.engine << "\n";
        reflex::TrtEngine engine(opt.engine, opt.batch, opt.verbose);
        std::cout << engine.summary();

        const reflex::OutputLayout layout = infer_layout(engine.out().dims);
        std::cout << "detector output: " << layout.num_anchors << " anchors x "
                  << layout.num_attrs << " attrs ("
                  << layout.num_classes() << " classes, "
                  << (layout.attr_major ? "attr-major" : "anchor-major")
                  << ")\n";
        if (opt.target_class >= layout.num_classes()) {
            std::cerr << "error: --class " << opt.target_class
                      << " but the model only has " << layout.num_classes()
                      << " classes\n";
            return 1;
        }

        // --- capture ------------------------------------------------------
        std::cout << "opening stream: " << opt.stream << "\n";
        FrameGrabber grabber(opt.stream);
        if (!grabber.open()) {
            std::cerr << "error: cannot open stream '" << opt.stream << "'\n"
                      << "  Check it in a browser first. If the browser shows "
                         "video but this does not,\n"
                      << "  OpenCV is probably built without FFMPEG support.\n";
            return 1;
        }
        grabber.start();

        cv::Mat frame;
        std::cout << "waiting for first frame...\n";
        bool got = false;
        for (int tries = 0; tries < 50 && !g_stop.load(); ++tries) {
            if (grabber.latest(&frame, 0.2)) { got = true; break; }
        }
        if (!got) {
            std::cerr << "error: no frame arrived within ~10s\n";
            return 1;
        }
        const int src_w = frame.cols, src_h = frame.rows;
        std::cout << "frame size: " << src_w << "x" << src_h << "\n";

        // --- preprocessing plan -------------------------------------------
        const reflex::LetterboxParams lb = reflex::make_letterbox(
            src_w, src_h, engine.input_w(), engine.input_h());
        const reflex::PreprocPlan plan =
            reflex::plan_preprocess(lb, opt.batch, dev);

        std::cout << "letterbox: " << src_w << "x" << src_h << " -> "
                  << lb.new_w << "x" << lb.new_h << " padded into "
                  << lb.dst_w << "x" << lb.dst_h << " (scale "
                  << std::fixed << std::setprecision(3) << lb.scale
                  << ", pad " << lb.pad_x << "," << lb.pad_y << ")\n";
        std::cout << "kernel: " << (plan.use_shared ? "shared-memory tile"
                                                    : "global-memory fallback")
                  << "  grid " << plan.grid.x << "x" << plan.grid.y << "x"
                  << plan.grid.z << "  block " << plan.block.x << "x"
                  << plan.block.y << "  window " << plan.win_w << "x"
                  << plan.win_h << "  shmem " << plan.shared_bytes << "/"
                  << plan.shared_limit << " B\n";
        if (!plan.use_shared) {
            std::cout << "  (the source window for one output tile exceeds this "
                         "GPU's per-block shared\n   memory, so the "
                         "global-memory kernel was selected - see "
                         "preprocess.cu)\n";
        }

        // --- buffers ------------------------------------------------------
        // Pinned (page-locked) host staging buffers: cudaMemcpyAsync can only
        // actually overlap with compute from pinned memory, and even
        // synchronous copies are measurably faster than from pageable memory
        // because the driver skips its internal staging copy.
        const size_t src_bytes_one = static_cast<size_t>(src_w) * src_h * 3;
        const size_t src_bytes     = src_bytes_one * opt.batch;

        uint8_t* h_frames = nullptr;
        float*   h_output = nullptr;
        uint8_t* d_frames = nullptr;
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_frames), src_bytes,
                                 cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h_output),
                                 engine.output_bytes(), cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_frames), src_bytes));

        // Everything runs on the default stream. The stages are strictly
        // dependent - preprocess needs the upload, inference needs preprocess -
        // so there is nothing to overlap, and StageProfiler records its events
        // without naming a stream. Using one stream keeps those events
        // correctly bracketing the work they claim to measure.
        const cudaStream_t stream = nullptr;

        // --- command path -------------------------------------------------
        reflex::Esp32Link link(cmd_host, cmd_port, opt.send_hz);
        link.set_enabled(!opt.no_send);
        link.start();
        std::cout << "commands -> http://" << cmd_host << ":" << cmd_port
                  << "/cmd" << (opt.no_send ? "  [--no-send: NOT transmitting]"
                                            : "")
                  << "\n";

        reflex::PostprocConfig pp;
        pp.conf_thresh  = opt.conf;
        pp.iou_thresh   = opt.iou;
        pp.target_class = opt.target_class;

        reflex::CommandShaper shaper;

        StageProfiler gpu_prof;
        HostProfiler  host_prof;

        std::cout << "\nrunning - Ctrl-C to stop"
                  << (opt.headless ? "" : ", or q in the window") << "\n\n";

        // --- loop ---------------------------------------------------------
        unsigned long processed = 0;
        bool  prev_valid = false;
        float prev_cx    = static_cast<float>(src_w) * 0.5f;
        auto  last_frame_t = Clock::now();
        auto  last_log_t   = Clock::now();
        double fps_ema = 0.0;
        bool size_warned = false;

        while (!g_stop.load()) {
            if (opt.max_frames > 0 &&
                processed >= static_cast<unsigned long>(opt.max_frames)) break;

            const auto loop_t0 = Clock::now();

            // -- gather `batch` frames --------------------------------------
            host_prof.start("1_capture_wait");
            int filled = 0;
            for (int b = 0; b < opt.batch && !g_stop.load(); ++b) {
                cv::Mat f;
                if (!grabber.latest(&f, 1.0)) break;
                if (f.cols != src_w || f.rows != src_h || f.type() != CV_8UC3) {
                    if (!size_warned) {
                        std::cerr << "warning: frame is " << f.cols << "x"
                                  << f.rows << " type " << f.type()
                                  << ", expected " << src_w << "x" << src_h
                                  << " CV_8UC3 - skipping mismatched frames.\n"
                                  << "  Restart the program to pick up a new "
                                     "camera resolution.\n";
                        size_warned = true;
                    }
                    --b;
                    continue;
                }
                // A Mat from VideoCapture is normally continuous; copy row by
                // row if it is not, rather than assuming.
                if (f.isContinuous()) {
                    std::memcpy(h_frames + static_cast<size_t>(b) * src_bytes_one,
                                f.data, src_bytes_one);
                } else {
                    const size_t row = static_cast<size_t>(src_w) * 3;
                    for (int y = 0; y < src_h; ++y) {
                        std::memcpy(h_frames + static_cast<size_t>(b) * src_bytes_one +
                                        y * row,
                                    f.ptr(y), row);
                    }
                }
                if (b == opt.batch - 1) frame = f;   // keep the newest to draw on
                ++filled;
            }
            host_prof.stop("1_capture_wait");

            if (filled == 0) {
                if (!grabber.alive()) {
                    std::cerr << "stream is down - retrying\n";
                }
                // No frame: the balance loop is still running on the Mega, and
                // the shaper's loss timer is what decays the command to zero.
                const reflex::Command c = shaper.update(
                    false, 0.f, 0.f, src_w, src_h,
                    std::chrono::duration<float>(Clock::now() - last_frame_t).count());
                last_frame_t = Clock::now();
                link.post_command(c.lean_deg, c.turn);
                continue;
            }

            const float dt =
                std::chrono::duration<float>(Clock::now() - last_frame_t).count();
            last_frame_t = Clock::now();

            // -- upload ------------------------------------------------------
            if (!opt.no_profile) gpu_prof.start("2_upload");
            CUDA_CHECK(cudaMemcpyAsync(d_frames, h_frames,
                                       src_bytes_one * filled,
                                       cudaMemcpyHostToDevice, stream));
            if (!opt.no_profile) gpu_prof.stop("2_upload");

            // -- preprocess --------------------------------------------------
            if (!opt.no_profile) gpu_prof.start("3_preprocess");
            reflex::launch_preprocess(d_frames, engine.input_device(), lb,
                                      filled, plan, 1.0f / 255.0f,
                                      114.0f / 255.0f, stream);
            if (!opt.no_profile) gpu_prof.stop("3_preprocess");

            // -- inference ---------------------------------------------------
            if (!opt.no_profile) gpu_prof.start("4_inference");
            engine.enqueue(stream);
            if (!opt.no_profile) gpu_prof.stop("4_inference");

            // -- download ----------------------------------------------------
            if (!opt.no_profile) gpu_prof.start("5_download");
            CUDA_CHECK(cudaMemcpyAsync(h_output, engine.output_device(),
                                       engine.output_bytes(),
                                       cudaMemcpyDeviceToHost, stream));
            if (!opt.no_profile) gpu_prof.stop("5_download");
            CUDA_CHECK(cudaStreamSynchronize(stream));

            // -- decode + shape ----------------------------------------------
            // With batch > 1 only the newest frame's slice is used for control;
            // the rest were processed to measure throughput. Each batch item's
            // output occupies one stride of the output tensor.
            host_prof.start("6_postprocess");
            const size_t per_item = engine.output_elements() /
                                    static_cast<size_t>(engine.batch());
            const float* out_slice =
                h_output + per_item * static_cast<size_t>(filled - 1);

            std::vector<reflex::Detection> dets =
                reflex::decode_yolov8(out_slice, layout, lb, pp);
            reflex::nms(dets, pp.iou_thresh, pp.class_agnostic_nms, pp.max_dets);
            const int ti = reflex::select_target(dets, pp, prev_valid, prev_cx,
                                                 src_w);
            host_prof.stop("6_postprocess");

            reflex::Command cmd;
            if (ti >= 0) {
                const reflex::Detection& d = dets[static_cast<size_t>(ti)];
                prev_cx    = d.cx();
                prev_valid = true;
                cmd = shaper.update(true, d.cx(), d.height(), src_w, src_h, dt);
            } else {
                prev_valid = false;
                cmd = shaper.update(false, 0.f, 0.f, src_w, src_h, dt);
            }

            link.post_command(cmd.lean_deg, cmd.turn);

            const double loop_ms =
                std::chrono::duration<double, std::milli>(Clock::now() - loop_t0)
                    .count();
            host_prof.add_sample("7_end_to_end", loop_ms);

            const double inst_fps = (dt > 1e-6f) ? 1.0 / dt : 0.0;
            fps_ema = (fps_ema == 0.0) ? inst_fps : 0.9 * fps_ema + 0.1 * inst_fps;
            processed += static_cast<unsigned long>(filled);

            // -- display -----------------------------------------------------
            if (!opt.headless) {
                cv::Mat disp = frame.clone();
                draw_overlay(disp, dets, ti, cmd, fps_ema, link.stats(),
                             link.telemetry());
                cv::imshow("reflex-cv", disp);
                const int key = cv::waitKey(1) & 0xff;
                if (key == 'q' || key == 27) g_stop.store(true);
            }

            // -- periodic console line (so --headless still shows life) ------
            if (std::chrono::duration<double>(Clock::now() - last_log_t).count()
                >= 2.0) {
                const auto ls = link.stats();
                const auto tm = link.telemetry();
                std::cout << std::fixed << std::setprecision(1)
                          << "fps " << fps_ema
                          << "  dets " << dets.size()
                          << "  lean " << std::setprecision(2) << cmd.lean_deg
                          << "  turn " << cmd.turn
                          << "  | grabbed " << grabber.captured()
                          << " dropped " << grabber.dropped()
                          << "  | link " << (ls.connected ? "up" : "DOWN")
                          << " sent " << ls.sent << " fail " << ls.failed
                          << " rtt " << std::setprecision(1) << ls.last_rtt_ms
                          << "ms";
                if (tm.valid) {
                    std::cout << "  | mega " << tm.angle_deg << "deg"
                              << (tm.fallen ? " FALLEN" : "");
                }
                std::cout << "\n";
                last_log_t = Clock::now();
            }

            // -- optional rate cap -------------------------------------------
            if (opt.fps_limit > 0.0) {
                const double budget_ms = 1000.0 / opt.fps_limit;
                const double spent =
                    std::chrono::duration<double, std::milli>(Clock::now() - loop_t0)
                        .count();
                if (spent < budget_ms) {
                    std::this_thread::sleep_for(
                        std::chrono::duration<double, std::milli>(budget_ms - spent));
                }
            }
        }

        // --- shutdown -------------------------------------------------------
        std::cout << "\nstopping\n";
        grabber.stop();
        link.stop();

        // Zero the command before exiting. Without this the Mega keeps acting
        // on whatever it last heard.
        if (!opt.no_send) {
            std::cout << "sending zero command... "
                      << (link.send_stop_blocking() ? "ok" : "FAILED - power "
                         "the robot down manually")
                      << "\n";
        }

        if (!opt.no_profile) gpu_prof.report();
        host_prof.report();

        std::cout << "\nframes processed " << processed
                  << ", grabbed " << grabber.captured()
                  << ", dropped before processing " << grabber.dropped()
                  << "\n";
        const auto ls = link.stats();
        std::cout << "commands sent " << ls.sent << ", superseded "
                  << ls.dropped << ", failed " << ls.failed << ", reconnects "
                  << ls.reconnects << "\n";

        std::cout << "\nmarkdown rows for the README latency table:\n";
        host_prof.report_markdown(std::cout);

        if (!opt.csv_path.empty()) {
            std::ofstream csv(opt.csv_path);
            if (csv) {
                host_prof.write_csv(csv);
                std::cout << "\nwrote " << opt.csv_path << "\n";
            } else {
                std::cerr << "could not write " << opt.csv_path << "\n";
            }
        }

        cudaFreeHost(h_frames);
        cudaFreeHost(h_output);
        cudaFree(d_frames);
        if (!opt.headless) cv::destroyAllWindows();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nfatal: " << e.what() << "\n";
        return 2;
    }
}
