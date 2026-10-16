// Tests for src/postprocess.{h,cpp} - decoding the detector's output tensor,
// NMS, and target selection.
//
// No GPU, no engine file, no TensorRT: postprocess.h deliberately includes
// none of those, so this suite builds anywhere. The synthetic output tensors
// below are built by hand, which means the expected box coordinates can be
// worked out on paper and compared exactly.

#include "../src/postprocess.h"
#include "test_util.h"

#include <cstdio>
#include <vector>

using namespace reflex;

namespace {

struct Cand {
    float cx, cy, w, h;   // model-space box
    int   cls;
    float score;
};

// Builds a YOLOv8-shaped output tensor in either layout.
std::vector<float> build_tensor(const OutputLayout& L,
                                const std::vector<Cand>& cands) {
    std::vector<float> t(static_cast<size_t>(L.num_attrs) * L.num_anchors, 0.f);
    const auto set = [&](int a, int attr, float v) {
        t[L.attr_major ? static_cast<size_t>(attr) * L.num_anchors + a
                       : static_cast<size_t>(a) * L.num_attrs + attr] = v;
    };
    for (size_t i = 0; i < cands.size(); ++i) {
        const int a = static_cast<int>(i);
        set(a, 0, cands[i].cx);
        set(a, 1, cands[i].cy);
        set(a, 2, cands[i].w);
        set(a, 3, cands[i].h);
        set(a, 4 + cands[i].cls, cands[i].score);
    }
    return t;
}

} // namespace

// ===========================================================================
static void test_decode_single_box() {
    SECTION("decode_yolov8 maps one box back to camera pixels");

    // 640x480 camera into a 640x640 model input: scale 1.0, 80px of padding at
    // top and bottom. So model x passes through unchanged and model y maps to
    // (y - 80).
    const LetterboxParams lb = make_letterbox(640, 480, 640, 640);

    OutputLayout L;
    L.num_attrs = 84;
    L.num_anchors = 100;
    L.attr_major = true;

    // A box centred in the model input, 100 wide and 200 tall:
    //   model corners  x 270..370,  y 220..420
    //   camera corners x 270..370,  y 140..340
    const std::vector<Cand> cands = {{320.f, 320.f, 100.f, 200.f, 0, 0.9f}};
    const std::vector<float> t = build_tensor(L, cands);

    PostprocConfig cfg;
    cfg.conf_thresh = 0.3f;
    cfg.target_class = 0;

    const std::vector<Detection> d = decode_yolov8(t.data(), L, lb, cfg);
    CHECK(d.size() == 1);
    if (d.size() == 1) {
        CHECK_NEAR(d[0].x1, 270.0, 1e-3);
        CHECK_NEAR(d[0].x2, 370.0, 1e-3);
        CHECK_NEAR(d[0].y1, 140.0, 1e-3);
        CHECK_NEAR(d[0].y2, 340.0, 1e-3);
        CHECK_NEAR(d[0].score, 0.9, 1e-6);
        CHECK(d[0].cls == 0);

        // And the derived quantities the command shaper actually consumes.
        CHECK_NEAR(d[0].cx(), 320.0, 1e-3);   // dead centre of a 640-wide frame
        CHECK_NEAR(d[0].cy(), 240.0, 1e-3);   // dead centre of a 480-tall frame
        CHECK_NEAR(d[0].width(), 100.0, 1e-3);
        CHECK_NEAR(d[0].height(), 200.0, 1e-3);
        CHECK_NEAR(d[0].area(), 20000.0, 1.0);
    }
}

// ===========================================================================
static void test_both_layouts_agree() {
    SECTION("attr-major and anchor-major layouts decode identically");

    const LetterboxParams lb = make_letterbox(800, 600, 640, 640);
    const std::vector<Cand> cands = {
        {300.f, 300.f, 80.f, 120.f, 0, 0.8f},
        {500.f, 200.f, 40.f, 60.f,  0, 0.6f},
    };

    OutputLayout A; A.num_attrs = 84; A.num_anchors = 50; A.attr_major = true;
    OutputLayout B; B.num_attrs = 84; B.num_anchors = 50; B.attr_major = false;

    const std::vector<float> ta = build_tensor(A, cands);
    const std::vector<float> tb = build_tensor(B, cands);

    PostprocConfig cfg;
    cfg.conf_thresh = 0.5f;

    const std::vector<Detection> da = decode_yolov8(ta.data(), A, lb, cfg);
    const std::vector<Detection> db = decode_yolov8(tb.data(), B, lb, cfg);

    CHECK(da.size() == 2);
    CHECK(da.size() == db.size());
    for (size_t i = 0; i < da.size() && i < db.size(); ++i) {
        CHECK_NEAR(da[i].x1, db[i].x1, 1e-4);
        CHECK_NEAR(da[i].y1, db[i].y1, 1e-4);
        CHECK_NEAR(da[i].x2, db[i].x2, 1e-4);
        CHECK_NEAR(da[i].y2, db[i].y2, 1e-4);
        CHECK_NEAR(da[i].score, db[i].score, 1e-6);
    }
}

// ===========================================================================
static void test_confidence_and_class_filtering() {
    SECTION("confidence threshold and class filtering");

    const LetterboxParams lb = make_letterbox(640, 480, 640, 640);
    OutputLayout L; L.num_attrs = 84; L.num_anchors = 50; L.attr_major = true;

    const std::vector<Cand> cands = {
        {320.f, 320.f, 60.f, 60.f, 0, 0.90f},
        {100.f, 320.f, 60.f, 60.f, 0, 0.40f},
        {200.f, 320.f, 60.f, 60.f, 0, 0.10f},
    };
    const std::vector<float> t = build_tensor(L, cands);

    PostprocConfig cfg;
    cfg.target_class = 0;

    cfg.conf_thresh = 0.05f;
    CHECK(decode_yolov8(t.data(), L, lb, cfg).size() == 3);
    cfg.conf_thresh = 0.30f;
    CHECK(decode_yolov8(t.data(), L, lb, cfg).size() == 2);
    cfg.conf_thresh = 0.95f;
    CHECK(decode_yolov8(t.data(), L, lb, cfg).empty());

    // Class filtering: an anchor carrying a high score for the WRONG class and
    // a low score for the right one must be judged on the right one.
    {
        std::vector<float> t2(static_cast<size_t>(L.num_attrs) * L.num_anchors,
                              0.f);
        const auto set = [&](int a, int attr, float v) {
            t2[static_cast<size_t>(attr) * L.num_anchors + a] = v;
        };
        set(0, 0, 320.f); set(0, 1, 320.f); set(0, 2, 60.f); set(0, 3, 60.f);
        set(0, 4 + 0, 0.20f);   // class 0 (person): weak
        set(0, 4 + 5, 0.95f);   // class 5 (bus):    strong

        PostprocConfig c0; c0.conf_thresh = 0.5f; c0.target_class = 0;
        CHECK(decode_yolov8(t2.data(), L, lb, c0).empty());  // person too weak

        PostprocConfig c5; c5.conf_thresh = 0.5f; c5.target_class = 5;
        const std::vector<Detection> d5 = decode_yolov8(t2.data(), L, lb, c5);
        CHECK(d5.size() == 1);
        if (!d5.empty()) {
            CHECK(d5[0].cls == 5);
            CHECK_NEAR(d5[0].score, 0.95, 1e-6);
        }

        // target_class < 0 means "whatever is most confident".
        PostprocConfig any; any.conf_thresh = 0.5f; any.target_class = -1;
        const std::vector<Detection> da = decode_yolov8(t2.data(), L, lb, any);
        CHECK(da.size() == 1);
        if (!da.empty()) CHECK(da[0].cls == 5);
    }
}

// ===========================================================================
static void test_box_clamping() {
    SECTION("boxes are clamped to the frame");

    const LetterboxParams lb = make_letterbox(640, 480, 640, 640);
    OutputLayout L; L.num_attrs = 84; L.num_anchors = 50; L.attr_major = true;

    // A target half out of frame on the left, which is normal when the robot
    // is turning to follow something.
    const std::vector<Cand> cands = {{20.f, 320.f, 200.f, 200.f, 0, 0.9f}};
    const std::vector<float> t = build_tensor(L, cands);

    PostprocConfig cfg; cfg.conf_thresh = 0.3f;
    const std::vector<Detection> d = decode_yolov8(t.data(), L, lb, cfg);
    CHECK(d.size() == 1);
    if (!d.empty()) {
        CHECK(d[0].x1 >= 0.f);
        CHECK(d[0].x2 <= 639.f);
        CHECK(d[0].y1 >= 0.f);
        CHECK(d[0].y2 <= 479.f);
        CHECK(d[0].width() > 0.f);
    }

    // A box entirely inside the padding strip collapses to nothing after
    // clamping and must be dropped rather than returned with zero area.
    const std::vector<Cand> bad = {{320.f, 5.f, 40.f, 8.f, 0, 0.9f}};
    const std::vector<float> tb = build_tensor(L, bad);
    const std::vector<Detection> db = decode_yolov8(tb.data(), L, lb, cfg);
    for (const auto& x : db) CHECK(x.area() > 0.f);
}

// ===========================================================================
static void test_iou() {
    SECTION("iou");

    Detection a; a.x1 = 0;  a.y1 = 0;  a.x2 = 10; a.y2 = 10;
    Detection b; b.x1 = 0;  b.y1 = 0;  b.x2 = 10; b.y2 = 10;
    CHECK_NEAR(iou(a, b), 1.0, 1e-6);

    // Half overlap: intersection 50, union 150.
    Detection c; c.x1 = 5;  c.y1 = 0;  c.x2 = 15; c.y2 = 10;
    CHECK_NEAR(iou(a, c), 50.0 / 150.0, 1e-6);

    // Disjoint.
    Detection d; d.x1 = 20; d.y1 = 20; d.x2 = 30; d.y2 = 30;
    CHECK_NEAR(iou(a, d), 0.0, 1e-9);

    // Touching edges only - zero area of intersection, not a divide by zero.
    Detection e; e.x1 = 10; e.y1 = 0;  e.x2 = 20; e.y2 = 10;
    CHECK_NEAR(iou(a, e), 0.0, 1e-9);

    // Fully contained: intersection 25, union 100.
    Detection f; f.x1 = 0;  f.y1 = 0;  f.x2 = 5;  f.y2 = 5;
    CHECK_NEAR(iou(a, f), 25.0 / 100.0, 1e-6);

    CHECK_NEAR(iou(a, b), iou(b, a), 1e-9);   // symmetric
}

// ===========================================================================
static void test_nms() {
    SECTION("nms");

    const auto box = [](float x1, float y1, float x2, float y2, float s,
                        int cls) {
        Detection d; d.x1 = x1; d.y1 = y1; d.x2 = x2; d.y2 = y2;
        d.score = s; d.cls = cls;
        return d;
    };

    // Two heavily overlapping boxes of the same class: the weaker one goes.
    {
        std::vector<Detection> v = {box(0, 0, 10, 10, 0.9f, 0),
                                    box(1, 1, 11, 11, 0.8f, 0)};
        nms(v, 0.45f);
        CHECK(v.size() == 1);
        if (!v.empty()) CHECK_NEAR(v[0].score, 0.9, 1e-6);
    }

    // Same geometry, different classes: both survive under per-class NMS,
    // because a ball in front of a person is two real detections.
    {
        std::vector<Detection> v = {box(0, 0, 10, 10, 0.9f, 0),
                                    box(1, 1, 11, 11, 0.8f, 1)};
        nms(v, 0.45f, /*class_agnostic=*/false);
        CHECK(v.size() == 2);
    }

    // ...and only one survives when NMS is told to ignore class.
    {
        std::vector<Detection> v = {box(0, 0, 10, 10, 0.9f, 0),
                                    box(1, 1, 11, 11, 0.8f, 1)};
        nms(v, 0.45f, /*class_agnostic=*/true);
        CHECK(v.size() == 1);
    }

    // Barely-overlapping boxes both survive.
    {
        std::vector<Detection> v = {box(0, 0, 10, 10, 0.9f, 0),
                                    box(9, 0, 19, 10, 0.8f, 0)};
        nms(v, 0.45f);
        CHECK(v.size() == 2);
    }

    // Output is sorted by score descending.
    {
        std::vector<Detection> v = {box(0, 0, 10, 10, 0.3f, 0),
                                    box(50, 50, 60, 60, 0.9f, 0),
                                    box(100, 100, 110, 110, 0.6f, 0)};
        nms(v, 0.45f);
        CHECK(v.size() == 3);
        for (size_t i = 1; i < v.size(); ++i) {
            CHECK(v[i - 1].score >= v[i].score);
        }
    }

    // max_dets caps the result, keeping the strongest.
    {
        std::vector<Detection> v;
        for (int i = 0; i < 50; ++i) {
            v.push_back(box(static_cast<float>(i * 20), 0.f,
                            static_cast<float>(i * 20 + 10), 10.f,
                            0.1f + 0.01f * static_cast<float>(i), 0));
        }
        nms(v, 0.45f, false, 5);
        CHECK(v.size() == 5);
        if (v.size() == 5) CHECK(v[0].score > v[4].score);
    }

    // Degenerate inputs must not crash.
    {
        std::vector<Detection> empty;
        nms(empty, 0.45f);
        CHECK(empty.empty());

        std::vector<Detection> one = {box(0, 0, 10, 10, 0.9f, 0)};
        nms(one, 0.45f);
        CHECK(one.size() == 1);
    }
}

// ===========================================================================
static void test_select_target() {
    SECTION("select_target");

    const auto box = [](float cx, float s, int cls) {
        Detection d;
        d.x1 = cx - 25.f; d.x2 = cx + 25.f;
        d.y1 = 100.f;     d.y2 = 200.f;
        d.score = s;      d.cls = cls;
        return d;
    };

    PostprocConfig cfg;
    cfg.target_class = 0;

    // Nothing to pick.
    {
        std::vector<Detection> none;
        CHECK(select_target(none, cfg, false, 0.f, 640) == -1);
    }

    // No prior target: take the most confident.
    {
        const std::vector<Detection> v = {box(100.f, 0.85f, 0),
                                          box(500.f, 0.90f, 0)};
        CHECK(select_target(v, cfg, false, 0.f, 640) == 1);
    }

    // With a prior target at x=500 and stickiness on, the slightly weaker
    // detection right next to it wins - the robot keeps following the same
    // person instead of snapping to a stranger who scored 0.05 higher.
    {
        const std::vector<Detection> v = {box(100.f, 0.90f, 0),
                                          box(500.f, 0.85f, 0)};
        cfg.stickiness = 2.0f;
        CHECK(select_target(v, cfg, true, 500.f, 640) == 1);

        // Stickiness 0 reverts to pure confidence.
        cfg.stickiness = 0.0f;
        CHECK(select_target(v, cfg, true, 500.f, 640) == 0);
        cfg.stickiness = 0.5f;
    }

    // Stickiness must not override a decisively better detection.
    {
        const std::vector<Detection> v = {box(100.f, 0.95f, 0),
                                          box(500.f, 0.20f, 0)};
        cfg.stickiness = 0.5f;
        CHECK(select_target(v, cfg, true, 500.f, 640) == 0);
    }

    // Detections of other classes are ignored entirely.
    {
        const std::vector<Detection> v = {box(100.f, 0.99f, 7),
                                          box(500.f, 0.40f, 0)};
        CHECK(select_target(v, cfg, false, 0.f, 640) == 1);
    }

    // target_class < 0 accepts anything.
    {
        PostprocConfig any = cfg;
        any.target_class = -1;
        const std::vector<Detection> v = {box(100.f, 0.99f, 7),
                                          box(500.f, 0.40f, 0)};
        CHECK(select_target(v, any, false, 0.f, 640) == 0);
    }
}

// ===========================================================================
// An empty / all-zero tensor is what arrives if the engine is fed a black
// frame or the output buffer was never written. It must produce no detections
// rather than a storm of zero-area boxes at the origin.
static void test_degenerate_inputs() {
    SECTION("degenerate inputs");

    const LetterboxParams lb = make_letterbox(640, 480, 640, 640);
    OutputLayout L; L.num_attrs = 84; L.num_anchors = 100; L.attr_major = true;

    const std::vector<float> zeros(
        static_cast<size_t>(L.num_attrs) * L.num_anchors, 0.f);
    PostprocConfig cfg; cfg.conf_thresh = 0.25f;
    CHECK(decode_yolov8(zeros.data(), L, lb, cfg).empty());

    // A null pointer must be refused, not dereferenced.
    CHECK(decode_yolov8(nullptr, L, lb, cfg).empty());

    // Nonsense layouts must be refused.
    OutputLayout bad1 = L; bad1.num_anchors = 0;
    CHECK(decode_yolov8(zeros.data(), bad1, lb, cfg).empty());
    OutputLayout bad2 = L; bad2.num_attrs = 4;     // no class scores at all
    CHECK(decode_yolov8(zeros.data(), bad2, lb, cfg).empty());
}

// ===========================================================================
int main() {
    std::printf("test_postprocess\n");
    test_decode_single_box();
    test_both_layouts_agree();
    test_confidence_and_class_filtering();
    test_box_clamping();
    test_iou();
    test_nms();
    test_select_target();
    test_degenerate_inputs();
    return test_summary("test_postprocess");
}
