#include "postprocess.h"

#include <algorithm>
#include <cmath>

namespace reflex {

std::vector<Detection> decode_yolov8(const float* out,
                                     const OutputLayout& layout,
                                     const LetterboxParams& lb,
                                     const PostprocConfig& cfg) {
    std::vector<Detection> dets;
    if (out == nullptr || layout.num_anchors <= 0 || layout.num_attrs <= 4) {
        return dets;
    }

    const int n_cls   = layout.num_classes();
    const int n_anch  = layout.num_anchors;
    const int n_attr  = layout.num_attrs;

    // Only one class matters for steering, but the class loop still has to run
    // over all of them when target_class is out of range (treated as "any").
    const bool any_class = cfg.target_class < 0 || cfg.target_class >= n_cls;

    dets.reserve(64);

    for (int a = 0; a < n_anch; ++a) {
        // Indexing differs only by stride; pulled out so the hot loop below
        // reads the same either way.
        const auto at = [&](int attr) -> float {
            return layout.attr_major ? out[static_cast<size_t>(attr) * n_anch + a]
                                     : out[static_cast<size_t>(a) * n_attr + attr];
        };

        int   best_cls   = -1;
        float best_score = 0.f;

        if (any_class) {
            for (int c = 0; c < n_cls; ++c) {
                const float s = at(4 + c);
                if (s > best_score) { best_score = s; best_cls = c; }
            }
        } else {
            best_score = at(4 + cfg.target_class);
            best_cls   = cfg.target_class;
        }

        if (best_cls < 0 || best_score < cfg.conf_thresh) continue;

        // Model-space centre/size -> model-space corners.
        const float cx = at(0);
        const float cy = at(1);
        const float w  = at(2);
        const float h  = at(3);

        const float mx1 = cx - 0.5f * w;
        const float my1 = cy - 0.5f * h;
        const float mx2 = cx + 0.5f * w;
        const float my2 = cy + 0.5f * h;

        // Undo the letterbox. Same functions the preprocessing kernel sampled
        // along, so the box lands where the pixels actually came from - see the
        // header comment in preprocess_math.cuh for why this matters.
        Detection d;
        d.x1 = dst_to_src_x(lb, mx1);
        d.y1 = dst_to_src_y(lb, my1);
        d.x2 = dst_to_src_x(lb, mx2);
        d.y2 = dst_to_src_y(lb, my2);

        // A box can legitimately extend past the frame edge when the target is
        // half out of view; clamp so downstream geometry stays sane.
        const float maxx = static_cast<float>(lb.src_w - 1);
        const float maxy = static_cast<float>(lb.src_h - 1);
        d.x1 = rcv_clampf(d.x1, 0.f, maxx);
        d.y1 = rcv_clampf(d.y1, 0.f, maxy);
        d.x2 = rcv_clampf(d.x2, 0.f, maxx);
        d.y2 = rcv_clampf(d.y2, 0.f, maxy);

        if (d.x2 <= d.x1 || d.y2 <= d.y1) continue;  // degenerate after clamping

        d.score = best_score;
        d.cls   = best_cls;
        dets.push_back(d);
    }

    return dets;
}

float iou(const Detection& a, const Detection& b) {
    const float ix1 = std::max(a.x1, b.x1);
    const float iy1 = std::max(a.y1, b.y1);
    const float ix2 = std::min(a.x2, b.x2);
    const float iy2 = std::min(a.y2, b.y2);

    const float iw = ix2 - ix1;
    const float ih = iy2 - iy1;
    if (iw <= 0.f || ih <= 0.f) return 0.f;

    const float inter = iw * ih;
    const float uni   = a.area() + b.area() - inter;
    return uni > 0.f ? inter / uni : 0.f;
}

void nms(std::vector<Detection>& dets, float iou_thresh,
         bool class_agnostic, int max_dets) {
    if (dets.size() < 2) {
        if (max_dets >= 0 && dets.size() > static_cast<size_t>(max_dets)) {
            dets.resize(static_cast<size_t>(max_dets));
        }
        return;
    }

    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });

    std::vector<char> dead(dets.size(), 0);
    std::vector<Detection> kept;
    kept.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); ++i) {
        if (dead[i]) continue;
        kept.push_back(dets[i]);
        if (max_dets >= 0 && kept.size() >= static_cast<size_t>(max_dets)) break;

        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (dead[j]) continue;
            if (!class_agnostic && dets[j].cls != dets[i].cls) continue;
            if (iou(dets[i], dets[j]) > iou_thresh) dead[j] = 1;
        }
    }

    dets.swap(kept);
}

int select_target(const std::vector<Detection>& dets,
                  const PostprocConfig& cfg,
                  bool prev_valid,
                  float prev_cx,
                  int src_w) {
    int   best_i = -1;
    float best_v = -1e30f;

    const float norm = (src_w > 1) ? static_cast<float>(src_w) : 1.f;
    const bool any_class = cfg.target_class < 0;

    for (size_t i = 0; i < dets.size(); ++i) {
        const Detection& d = dets[i];
        if (!any_class && d.cls != cfg.target_class) continue;

        float v = d.score;
        if (prev_valid && cfg.stickiness > 0.f) {
            // Penalty is in units of "fraction of frame width moved", so the
            // tuning is resolution-independent.
            v -= cfg.stickiness * std::fabs(d.cx() - prev_cx) / norm;
        }
        if (v > best_v) { best_v = v; best_i = static_cast<int>(i); }
    }

    return best_i;
}

} // namespace reflex
