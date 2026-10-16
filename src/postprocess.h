#pragma once
// ---------------------------------------------------------------------------
// Turning the detector's raw output tensor into "where is my target, in camera
// pixels".
//
// Deliberately free of any CUDA or TensorRT include. Everything here is plain
// C++ on plain floats, which is what lets test/test_postprocess.cpp build and
// run this logic on any machine - no GPU, no engine file, no NVIDIA headers.
// Decoding and NMS are exactly the kind of fiddly index arithmetic that is
// painful to debug on a robot and trivial to debug in a unit test.
// ---------------------------------------------------------------------------

#include <cstddef>
#include <vector>
#include "preprocess_math.cuh"   // LetterboxParams + the shared inverse mapping

namespace reflex {

// A detection in ORIGINAL camera-frame pixel coordinates (not model space -
// the mapping back has already been applied).
struct Detection {
    float x1 = 0.f, y1 = 0.f, x2 = 0.f, y2 = 0.f;
    float score = 0.f;
    int   cls = -1;

    float width()  const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float cx()     const { return 0.5f * (x1 + x2); }
    float cy()     const { return 0.5f * (y1 + y2); }
    float area()   const { float w = width(), h = height();
                           return (w > 0.f && h > 0.f) ? w * h : 0.f; }
};

struct PostprocConfig {
    float conf_thresh = 0.35f;  // below this a detection is ignored entirely
    float iou_thresh  = 0.45f;  // NMS overlap above which the weaker box dies
    int   target_class = 0;     // COCO class to follow; 0 = person, 32 = sports ball
    bool  class_agnostic_nms = false;
    int   max_dets = 100;       // cap kept boxes, so a pathological frame
                                // cannot stall the loop in NMS
    // How strongly to prefer the detection nearest to last frame's target when
    // several candidates qualify. 0 = always take the highest score (flips
    // between two equally-good targets); higher = stickier.
    float stickiness = 0.5f;
};

// How the detector's output tensor is laid out.
//
// A YOLOv8 ONNX export is normally [1, 84, 8400] - attribute-major, i.e. all
// 8400 values of attribute 0, then all 8400 of attribute 1. Some exporters and
// some ONNX graph surgery produce the transposed [1, 8400, 84] instead. Both
// appear in the wild, so the layout is a parameter rather than an assumption;
// main.cpp infers it from the engine's reported dims.
struct OutputLayout {
    int  num_attrs   = 84;    // 4 box values + one score per class
    int  num_anchors = 8400;  // candidate boxes
    bool attr_major  = true;  // true => [1, num_attrs, num_anchors]

    int num_classes() const { return num_attrs - 4; }
};

// Decode raw output into camera-space boxes, keeping only candidates above
// cfg.conf_thresh. No NMS yet - run nms() next.
//
// Box values are expected as (cx, cy, w, h) in MODEL pixel space, which is
// what YOLOv8 emits, and class scores are expected already activated (the
// Ultralytics export bakes the sigmoid in).
std::vector<Detection> decode_yolov8(const float* out,
                                     const OutputLayout& layout,
                                     const LetterboxParams& lb,
                                     const PostprocConfig& cfg);

// Intersection-over-union of two boxes; 0 when they do not overlap.
float iou(const Detection& a, const Detection& b);

// Greedy non-maximum suppression, in place. Sorts by score descending and
// drops any lower-scoring box overlapping a kept one by more than iou_thresh.
// Per-class by default: a ball in front of a person should not suppress the
// person just because the boxes overlap.
void nms(std::vector<Detection>& dets, float iou_thresh,
         bool class_agnostic = false, int max_dets = 100);

// Pick which detection to actually chase.
//
// Returns an index into `dets`, or -1 if nothing suitable is present. When
// `prev_valid`, candidates are scored by confidence minus a penalty on how far
// their centre moved from `prev_cx`, so the robot keeps following the same
// person instead of snapping between two of them frame to frame.
int select_target(const std::vector<Detection>& dets,
                  const PostprocConfig& cfg,
                  bool prev_valid,
                  float prev_cx,
                  int src_w);

} // namespace reflex
