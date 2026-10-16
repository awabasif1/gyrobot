#include "command_shaper.h"

#include <cmath>

namespace reflex {

namespace {
inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
} // namespace

float apply_deadband(float v, float deadband) {
    if (deadband <= 0.f) return v;
    if (deadband >= 1.f) return 0.f;
    const float a = std::fabs(v);
    if (a <= deadband) return 0.f;
    const float sign = (v < 0.f) ? -1.f : 1.f;
    return sign * (a - deadband) / (1.f - deadband);
}

float slew_limit(float current, float target, float rate, float dt) {
    if (rate <= 0.f || dt <= 0.f) return target;
    const float max_step = rate * dt;
    const float delta    = target - current;
    if (delta >  max_step) return current + max_step;
    if (delta < -max_step) return current - max_step;
    return target;
}

float lowpass_alpha(float tau_s, float dt) {
    if (tau_s <= 1e-6f || dt <= 0.f) return 1.f;
    // 1 - exp(-dt/tau): the exact discrete equivalent of a first-order RC
    // filter, so the cutoff does not drift with frame rate.
    const float a = 1.f - std::exp(-dt / tau_s);
    return clampf(a, 0.f, 1.f);
}

void CommandShaper::reset() {
    out_    = Command{};
    lost_s_ = 1e9f;
    primed_ = false;
}

Command CommandShaper::update(bool have_target, float target_cx,
                              float target_h, int frame_w, int frame_h,
                              float dt) {
    // A wild dt (first frame, a debugger pause, the stream stalling for two
    // seconds) must not be allowed to blow through the slew limits in one
    // step, which is exactly the situation that throws a balancing robot.
    dt = clampf(dt, 0.f, 0.25f);

    float want_turn = 0.f;
    float want_lean = 0.f;

    if (have_target && frame_w > 1 && frame_h > 1) {
        lost_s_ = 0.f;

        // --- turn: horizontal offset from frame centre, normalised to -1..1
        const float half_w = 0.5f * static_cast<float>(frame_w);
        const float offset = (target_cx - half_w) / half_w;
        want_turn = clampf(cfg_.turn_gain * apply_deadband(offset, cfg_.turn_deadband),
                           -cfg_.max_turn, cfg_.max_turn);

        // --- lean: how far off the target's apparent size is from the size we
        // want it to be. Positive error (target looks too small) => lean
        // forward to close the gap.
        const float height_frac = target_h / static_cast<float>(frame_h);
        const float size_err    = cfg_.desired_height_frac - height_frac;
        want_lean = clampf(cfg_.lean_gain * apply_deadband(size_err, cfg_.lean_deadband),
                           -cfg_.max_lean_deg, cfg_.max_lean_deg);
    } else {
        lost_s_ += dt;

        if (lost_s_ <= cfg_.hold_s) {
            // Coast on the previous command through a short dropout.
            want_turn = out_.turn;
            want_lean = out_.lean_deg;
        } else {
            // Ramp to a standstill rather than cutting to zero instantly - a
            // step to zero is itself a disturbance the balance loop has to
            // absorb.
            const float t = clampf((lost_s_ - cfg_.hold_s) /
                                       (cfg_.decay_s > 0.f ? cfg_.decay_s : 1e-6f),
                                   0.f, 1.f);
            const float k = 1.f - t;
            want_turn = out_.turn * k;
            want_lean = out_.lean_deg * k;
        }
    }

    // --- low-pass, then slew-limit -----------------------------------------
    // Order matters: filtering first removes the per-frame detector jitter,
    // and the slew limit then bounds how fast even a genuine, sustained change
    // is allowed to reach the motors.
    if (!primed_ && have_target) {
        // Seed the filter on the first real detection instead of letting it
        // crawl up from zero, which would make the robot ignore the first
        // ~tau of a target that is already well off-centre.
        out_.turn     = want_turn;
        out_.lean_deg = want_lean;
        primed_       = true;
    } else {
        const float a = lowpass_alpha(cfg_.smooth_tau_s, dt);

        const float turn_f = out_.turn + a * (want_turn - out_.turn);
        const float lean_f = out_.lean_deg + a * (want_lean - out_.lean_deg);

        out_.turn     = slew_limit(out_.turn, turn_f, cfg_.max_turn_rate, dt);
        out_.lean_deg = slew_limit(out_.lean_deg, lean_f, cfg_.max_lean_rate_deg, dt);
    }

    out_.turn     = clampf(out_.turn, -cfg_.max_turn, cfg_.max_turn);
    out_.lean_deg = clampf(out_.lean_deg, -cfg_.max_lean_deg, cfg_.max_lean_deg);
    out_.target_valid = have_target;

    return out_;
}

} // namespace reflex
