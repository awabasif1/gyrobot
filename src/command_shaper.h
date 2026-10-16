#pragma once
// ---------------------------------------------------------------------------
// Detection -> motion command.
//
// This is the smallest file in the vision pipeline and the one most likely to
// decide whether the robot looks controlled or looks drunk. The detector's
// output jitters by a few pixels every frame even on a stationary target, and
// frames arrive over WiFi at an uneven rate. Feeding that straight into a
// balancing robot's setpoint produces visible twitching at best and a fall at
// worst, so every value leaving here is deadbanded, low-passed, and
// rate-limited first.
//
// Like postprocess.*, this is deliberately plain C++ with no CUDA/TensorRT
// dependency, so test/test_command_shaper.cpp can drive it through hundreds of
// simulated frames - including target loss - without any hardware.
//
// UNITS (must match mcu/SERIAL_PROTOCOL.md exactly):
//   lean_deg : degrees, added to the Mega's balance setpoint.
//              positive = nose-down/forward = drive forward.
//   turn     : dimensionless, -1 .. +1, a differential applied to the two
//              motors. positive = turn right (clockwise viewed from above).
// ---------------------------------------------------------------------------

namespace reflex {

struct ShaperConfig {
    // --- turn channel (horizontal tracking) ---
    float turn_gain     = 0.9f;  // per unit of normalised horizontal offset
    float max_turn      = 1.0f;
    float turn_deadband = 0.05f; // offsets this small are treated as centred

    // --- lean channel (distance keeping) ---
    // Bounding-box height as a fraction of frame height is the distance proxy:
    // a target that looks short is far away, so lean forward. It is crude but
    // monotonic, needs no camera calibration, and is robust to the target
    // being partly out of frame horizontally.
    float lean_gain           = 12.0f; // degrees per unit height-fraction error
    float desired_height_frac = 0.45f; // "this big" = "the right distance away"
    float max_lean_deg        = 4.0f;  // hard cap; see note below
    float lean_deadband       = 0.03f;

    // --- smoothing ---
    // A time constant, not a fixed EMA weight: the per-frame coefficient is
    // recomputed from dt, so the response is the same whether frames arrive at
    // 30fps or stutter to 8fps. A fixed weight would make the filter
    // effectively faster or slower with the WiFi link's mood.
    float smooth_tau_s = 0.12f;

    // --- slew limits ---
    float max_turn_rate     = 4.0f;  // turn units per second
    float max_lean_rate_deg = 12.0f; // degrees per second

    // --- target loss behaviour ---
    // Brief dropouts are normal (one bad JPEG, one missed detection, one
    // retransmit). Holding the last command briefly keeps the robot turning
    // toward a target that is still there; zeroing out after that stops it
    // spinning forever toward something that has gone.
    float hold_s  = 0.35f;
    float decay_s = 0.40f;
};

struct Command {
    float lean_deg     = 0.f;
    float turn         = 0.f;
    bool  target_valid = false; // was a target seen on THIS update
};

class CommandShaper {
public:
    CommandShaper() = default;
    explicit CommandShaper(const ShaperConfig& cfg) : cfg_(cfg) {}

    // Call once per processed frame.
    //   have_target : did the detector produce a target this frame
    //   target_cx   : target centre x, in camera-frame pixels
    //   target_h    : target bounding-box height, in camera-frame pixels
    //   frame_w/h   : camera frame size
    //   dt          : seconds since the previous update (clamped internally)
    Command update(bool have_target, float target_cx, float target_h,
                   int frame_w, int frame_h, float dt);

    // Drops all filter state and zeroes the outputs. Used on shutdown and
    // whenever the pipeline is restarted, so stale motion never resumes.
    void reset();

    const Command& last() const { return out_; }
    float seconds_since_target() const { return lost_s_; }
    const ShaperConfig& config() const { return cfg_; }

private:
    ShaperConfig cfg_{};
    Command      out_{};
    float        lost_s_ = 1e9f;  // "no target has ever been seen"
    bool         primed_ = false; // has the low-pass been seeded yet
};

// --- small helpers, exposed because the tests check them directly -----------

// Removes a dead zone around zero WITHOUT introducing a jump: values inside
// the band go to zero, values outside are rescaled so the output still spans
// the full range. A hard cut would make the command snap from 0 to
// gain*deadband the instant the target drifted past the edge.
float apply_deadband(float v, float deadband);

// Moves `current` toward `target` by at most rate*dt.
float slew_limit(float current, float target, float rate, float dt);

// First-order low-pass coefficient for a given time constant and timestep.
// Returns 1.0 (pass straight through) when tau is zero or dt is large
// relative to it.
float lowpass_alpha(float tau_s, float dt);

} // namespace reflex
