// Tests for src/command_shaper.{h,cpp} - the stage that turns a detection into
// the lean/turn command the robot acts on.
//
// This is the smallest piece of the vision pipeline and the one with the most
// direct influence on whether the robot looks controlled or looks drunk, so it
// gets the most behavioural testing: not just "does the arithmetic work" but
// "does a jittery detection produce a smooth command", "does the robot stop
// when the target disappears", and "can a stalled pipeline produce a step
// change big enough to throw it over".

#include "../src/command_shaper.h"
#include "test_util.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace reflex;

static const int W = 640;
static const int H = 480;

// ===========================================================================
static void test_helpers() {
    SECTION("apply_deadband / slew_limit / lowpass_alpha");

    // Inside the band -> exactly zero.
    CHECK_NEAR(apply_deadband(0.0f, 0.1f), 0.0, 1e-9);
    CHECK_NEAR(apply_deadband(0.05f, 0.1f), 0.0, 1e-9);
    CHECK_NEAR(apply_deadband(-0.1f, 0.1f), 0.0, 1e-9);

    // Just outside -> small but nonzero, and CONTINUOUS. A naive deadband that
    // simply passes the value through once it clears the band would jump from
    // 0 to 0.1 here; this one eases in from 0.
    const float just_out = apply_deadband(0.1001f, 0.1f);
    CHECK(just_out > 0.0f);
    CHECK(just_out < 0.01f);

    // Full scale still maps to full scale, so the deadband costs no range.
    CHECK_NEAR(apply_deadband(1.0f, 0.1f), 1.0, 1e-6);
    CHECK_NEAR(apply_deadband(-1.0f, 0.1f), -1.0, 1e-6);

    // Symmetric.
    CHECK_NEAR(apply_deadband(0.5f, 0.1f), -apply_deadband(-0.5f, 0.1f), 1e-6);

    // Degenerate bands.
    CHECK_NEAR(apply_deadband(0.5f, 0.0f), 0.5, 1e-6);
    CHECK_NEAR(apply_deadband(0.5f, 1.0f), 0.0, 1e-6);

    // Slew limiting.
    CHECK_NEAR(slew_limit(0.f, 10.f, 1.f, 1.f), 1.0, 1e-6);
    CHECK_NEAR(slew_limit(0.f, -10.f, 1.f, 1.f), -1.0, 1e-6);
    CHECK_NEAR(slew_limit(0.f, 0.5f, 1.f, 1.f), 0.5, 1e-6);   // reaches target
    CHECK_NEAR(slew_limit(5.f, 5.f, 1.f, 1.f), 5.0, 1e-6);    // already there
    CHECK_NEAR(slew_limit(0.f, 10.f, 0.f, 1.f), 10.0, 1e-6);  // rate 0 = off

    // Low-pass coefficient.
    CHECK_NEAR(lowpass_alpha(0.f, 0.01f), 1.0, 1e-6);    // tau 0 = passthrough
    CHECK(lowpass_alpha(0.1f, 0.01f) > 0.0f);
    CHECK(lowpass_alpha(0.1f, 0.01f) < 1.0f);
    // A longer step gives a larger coefficient, which is what keeps the
    // response consistent as the frame rate varies.
    CHECK(lowpass_alpha(0.1f, 0.05f) > lowpass_alpha(0.1f, 0.01f));
    // dt >> tau approaches 1 (fully follow the new value).
    CHECK(lowpass_alpha(0.01f, 1.0f) > 0.99f);
}

// ===========================================================================
static void test_turn_direction_and_centre() {
    SECTION("turn direction");

    ShaperConfig cfg;
    CommandShaper s(cfg);

    // A target dead centre must produce no turn.
    Command c = s.update(true, W * 0.5f, 150.f, W, H, 0.033f);
    CHECK_NEAR(c.turn, 0.0, 1e-6);
    CHECK(c.target_valid);

    // Target to the RIGHT of centre -> positive turn (turn right).
    s.reset();
    for (int i = 0; i < 100; ++i) c = s.update(true, 560.f, 150.f, W, H, 0.033f);
    CHECK(c.turn > 0.2f);

    // Target to the LEFT -> negative turn.
    s.reset();
    for (int i = 0; i < 100; ++i) c = s.update(true, 80.f, 150.f, W, H, 0.033f);
    CHECK(c.turn < -0.2f);

    // Magnitude is bounded by max_turn no matter how far off-centre.
    s.reset();
    for (int i = 0; i < 200; ++i) c = s.update(true, 100000.f, 150.f, W, H, 0.033f);
    CHECK(c.turn <= cfg.max_turn + 1e-5f);

    // A small offset inside the deadband is treated as centred, which is what
    // stops the robot twitching at a stationary target.
    s.reset();
    const float small = W * 0.5f + 0.02f * (W * 0.5f);   // 2% off centre
    for (int i = 0; i < 100; ++i) c = s.update(true, small, 150.f, W, H, 0.033f);
    CHECK_NEAR(c.turn, 0.0, 1e-6);
}

// ===========================================================================
static void test_lean_direction() {
    SECTION("lean direction (distance keeping)");

    ShaperConfig cfg;
    CommandShaper s(cfg);
    Command c;

    // Target looks SMALL => it is far away => lean forward (positive).
    const float small_h = 0.20f * H;
    for (int i = 0; i < 200; ++i)
        c = s.update(true, W * 0.5f, small_h, W, H, 0.033f);
    CHECK(c.lean_deg > 0.5f);

    // Target looks BIG => too close => lean back (negative).
    s.reset();
    const float big_h = 0.80f * H;
    for (int i = 0; i < 200; ++i)
        c = s.update(true, W * 0.5f, big_h, W, H, 0.033f);
    CHECK(c.lean_deg < -0.5f);

    // Target at exactly the desired size => no lean.
    s.reset();
    const float ok_h = cfg.desired_height_frac * H;
    for (int i = 0; i < 200; ++i)
        c = s.update(true, W * 0.5f, ok_h, W, H, 0.033f);
    CHECK_NEAR(c.lean_deg, 0.0, 1e-6);

    // Bounded in both directions.
    s.reset();
    for (int i = 0; i < 400; ++i) c = s.update(true, W * 0.5f, 1.f, W, H, 0.033f);
    CHECK(c.lean_deg <= cfg.max_lean_deg + 1e-5f);
    s.reset();
    for (int i = 0; i < 400; ++i)
        c = s.update(true, W * 0.5f, 10.f * H, W, H, 0.033f);
    CHECK(c.lean_deg >= -cfg.max_lean_deg - 1e-5f);
}

// ===========================================================================
// The headline behaviour: a detection that jitters a few pixels frame to frame
// (which every detector does, even on a perfectly still target) must not
// produce a command that jitters.
static void test_jitter_rejection() {
    SECTION("jitter rejection");

    ShaperConfig cfg;
    CommandShaper s(cfg);

    const float base   = 500.f;   // well off-centre, so we are out of the deadband
    const float jitter = 20.f;    // +/- 20 px of detector noise
    const float dt     = 1.f / 30.f;

    // Settle first.
    for (int i = 0; i < 200; ++i) s.update(true, base, 150.f, W, H, dt);

    float lo = 1e9f, hi = -1e9f;
    for (int i = 0; i < 400; ++i) {
        const float cx = base + ((i % 2) ? jitter : -jitter);
        const Command c = s.update(true, cx, 150.f, W, H, dt);
        if (c.turn < lo) lo = c.turn;
        if (c.turn > hi) hi = c.turn;
    }
    const float out_ripple = hi - lo;

    // What the command would have done with no smoothing at all.
    const auto raw_turn = [&](float cx) {
        const float half = 0.5f * static_cast<float>(W);
        return cfg.turn_gain * apply_deadband((cx - half) / half,
                                              cfg.turn_deadband);
    };
    const float raw_ripple = raw_turn(base + jitter) - raw_turn(base - jitter);

    std::printf("    raw ripple %.4f -> smoothed ripple %.4f (%.0f%% of raw)\n",
                static_cast<double>(raw_ripple), static_cast<double>(out_ripple),
                static_cast<double>(100.f * out_ripple / raw_ripple));

    CHECK(out_ripple < 0.40f * raw_ripple);
    CHECK(out_ripple > 0.0f);   // not simply frozen
}

// ===========================================================================
static void test_target_loss() {
    SECTION("target loss: hold, then decay to zero");

    ShaperConfig cfg;
    CommandShaper s(cfg);
    const float dt = 1.f / 30.f;

    // Establish a solid turn command.
    Command c;
    for (int i = 0; i < 300; ++i) c = s.update(true, 600.f, 150.f, W, H, dt);
    const float established = c.turn;
    CHECK(established > 0.3f);

    // Brief dropout, inside hold_s: the command is held, because the target is
    // probably still there and we do not want to stop steering over one bad
    // JPEG.
    const int hold_steps = static_cast<int>(0.5f * cfg.hold_s / dt);
    for (int i = 0; i < hold_steps; ++i) c = s.update(false, 0, 0, W, H, dt);
    CHECK_NEAR(c.turn, established, 1e-4);
    CHECK(!c.target_valid);

    // Keep it missing: the command must decay to a standstill, not keep the
    // robot turning forever toward something that has gone.
    for (int i = 0; i < static_cast<int>(3.0f / dt); ++i)
        c = s.update(false, 0, 0, W, H, dt);
    CHECK_NEAR(c.turn, 0.0, 0.02);
    CHECK_NEAR(c.lean_deg, 0.0, 0.05);

    // The decay must be monotonic - no overshoot past zero into a reverse
    // turn, which would look like the robot changing its mind.
    s.reset();
    for (int i = 0; i < 300; ++i) s.update(true, 600.f, 150.f, W, H, dt);
    float prev = s.last().turn;
    for (int i = 0; i < static_cast<int>(3.0f / dt); ++i) {
        const Command d = s.update(false, 0, 0, W, H, dt);
        CHECK(d.turn <= prev + 1e-4f);
        CHECK(d.turn >= -1e-3f);
        prev = d.turn;
    }

    // Reacquiring the target brings the command back.
    for (int i = 0; i < 300; ++i) c = s.update(true, 600.f, 150.f, W, H, dt);
    CHECK(c.turn > 0.3f);
    CHECK(c.target_valid);
}

// ===========================================================================
// A stalled pipeline (a long GC pause on the host, a debugger breakpoint, the
// stream freezing for two seconds) hands the shaper an enormous dt. That must
// not let a single update step the command across its whole range - a step
// change in setpoint is a disturbance the balance loop has to absorb, and a
// big enough one throws the robot over.
static void test_large_dt_is_bounded() {
    SECTION("a huge dt cannot produce a huge step");

    ShaperConfig cfg;
    CommandShaper s(cfg);

    // Prime at centre.
    for (int i = 0; i < 100; ++i) s.update(true, W * 0.5f, 150.f, W, H, 0.033f);
    const float before = s.last().turn;

    // Now the pipeline stalls for 5 seconds and the target has jumped to the
    // far edge.
    const Command c = s.update(true, 639.f, 150.f, W, H, 5.0f);

    // dt is clamped internally to 0.25s, so the slew limit still applies and
    // bounds the step.
    const float max_step = cfg.max_turn_rate * 0.25f;
    CHECK(std::fabs(c.turn - before) <= max_step + 1e-4f);
    CHECK(std::fabs(c.turn) <= cfg.max_turn + 1e-5f);

    // Negative or zero dt must not corrupt state either.
    const Command z = s.update(true, 639.f, 150.f, W, H, 0.0f);
    CHECK(std::fabs(z.turn) <= cfg.max_turn + 1e-5f);
    const Command n = s.update(true, 639.f, 150.f, W, H, -1.0f);
    CHECK(std::fabs(n.turn) <= cfg.max_turn + 1e-5f);
}

// ===========================================================================
static void test_slew_rate_respected() {
    SECTION("slew rate is respected across a normal sweep");

    ShaperConfig cfg;
    CommandShaper s(cfg);
    const float dt = 1.f / 30.f;

    for (int i = 0; i < 100; ++i) s.update(true, 40.f, 150.f, W, H, dt);

    float prev = s.last().turn;
    // Target teleports from the far left to the far right.
    for (int i = 0; i < 200; ++i) {
        const Command c = s.update(true, 600.f, 150.f, W, H, dt);
        CHECK(std::fabs(c.turn - prev) <= cfg.max_turn_rate * dt + 1e-4f);
        CHECK(std::fabs(c.lean_deg - s.last().lean_deg) <=
              cfg.max_lean_rate_deg * dt + 1e-4f);
        prev = c.turn;
    }
}

// ===========================================================================
static void test_reset_and_priming() {
    SECTION("reset and first-detection priming");

    CommandShaper s;

    // The first detection primes the filter rather than crawling up from zero,
    // so a target that is already well off-centre is acted on immediately.
    const Command first = s.update(true, 620.f, 150.f, W, H, 0.033f);
    CHECK(first.turn > 0.3f);

    // reset() must clear everything.
    s.reset();
    CHECK_NEAR(s.last().turn, 0.0, 1e-9);
    CHECK_NEAR(s.last().lean_deg, 0.0, 1e-9);
    CHECK(!s.last().target_valid);
    CHECK(s.seconds_since_target() > 1.0f);

    // A degenerate frame size must not divide by zero.
    const Command d = s.update(true, 10.f, 10.f, 0, 0, 0.033f);
    CHECK(std::fabs(d.turn) < 1e6f);
    CHECK(std::fabs(d.lean_deg) < 1e6f);
}

// ===========================================================================
int main() {
    std::printf("test_command_shaper\n");
    test_helpers();
    test_turn_direction_and_centre();
    test_lean_direction();
    test_jitter_rejection();
    test_target_loss();
    test_large_dt_is_bounded();
    test_slew_rate_respected();
    test_reset_and_priming();
    return test_summary("test_command_shaper");
}
