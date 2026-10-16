// Tests for mcu/balance_core.h - the logic the Arduino Mega runs.
//
// Build and run:  make -C test   (or see test/Makefile)
//
// The last section is the one worth understanding: it closes the loop around a
// simulated inverted pendulum. That is what actually proves the SIGN of the
// feedback is right, which is the single most common way a first balancing
// robot fails - with the wrong sign it drives itself over, enthusiastically,
// and it looks identical to badly-tuned gains.

#include "../mcu/balance_core.h"
#include "test_util.h"

#include <cmath>
#include <cstdio>

using namespace reflex;

// ===========================================================================
static void test_parse_float() {
    SECTION("parse_float");
    float v = 0.f;

    CHECK(CommandParser::parse_float("0", nullptr, &v));        CHECK_NEAR(v, 0.0, 1e-6);
    CHECK(CommandParser::parse_float("12", nullptr, &v));       CHECK_NEAR(v, 12.0, 1e-6);
    CHECK(CommandParser::parse_float("-3.25", nullptr, &v));    CHECK_NEAR(v, -3.25, 1e-5);
    CHECK(CommandParser::parse_float("+1.50", nullptr, &v));    CHECK_NEAR(v, 1.5, 1e-5);
    CHECK(CommandParser::parse_float("  7.125", nullptr, &v));  CHECK_NEAR(v, 7.125, 1e-5);
    CHECK(CommandParser::parse_float(".5", nullptr, &v));       CHECK_NEAR(v, 0.5, 1e-6);
    CHECK(CommandParser::parse_float("-0.07", nullptr, &v));    CHECK_NEAR(v, -0.07, 1e-5);

    // Rejections: these must not silently produce a number.
    CHECK(!CommandParser::parse_float("", nullptr, &v));
    CHECK(!CommandParser::parse_float("abc", nullptr, &v));
    CHECK(!CommandParser::parse_float("-", nullptr, &v));
    CHECK(!CommandParser::parse_float(".", nullptr, &v));
    CHECK(!CommandParser::parse_float("1e5", nullptr, &v));   // exponent rejected on purpose

    // The parser reports where it stopped, which is how parse_line walks the
    // string without a second pass.
    const char* end = nullptr;
    CHECK(CommandParser::parse_float("2.50,rest", &end, &v));
    CHECK_NEAR(v, 2.5, 1e-5);
    CHECK(end != nullptr && *end == ',');
}

// ===========================================================================
static void test_parse_line() {
    SECTION("parse_line");
    float lean = 9.f, turn = 9.f;

    CHECK(CommandParser::parse_line("L:1.25,T:-0.40", &lean, &turn));
    CHECK_NEAR(lean, 1.25, 1e-5);
    CHECK_NEAR(turn, -0.40, 1e-5);

    CHECK(CommandParser::parse_line("L:0.00,T:0.00", &lean, &turn));
    CHECK_NEAR(lean, 0.0, 1e-6);
    CHECK_NEAR(turn, 0.0, 1e-6);

    // Tolerated variations.
    CHECK(CommandParser::parse_line("L:2.0, T:1.0", &lean, &turn));
    CHECK(CommandParser::parse_line("l:2.0,t:1.0", &lean, &turn));
    CHECK(CommandParser::parse_line(" L:2.0,T:1.0 ", &lean, &turn));

    // Rejected: every one of these is something a noisy UART really produces.
    CHECK(!CommandParser::parse_line("", &lean, &turn));
    CHECK(!CommandParser::parse_line("L:1.0", &lean, &turn));           // truncated
    CHECK(!CommandParser::parse_line("L:1.0;T:2.0", &lean, &turn));     // wrong separator
    CHECK(!CommandParser::parse_line("X:1.0,T:2.0", &lean, &turn));     // wrong key
    CHECK(!CommandParser::parse_line("L1.0,T:2.0", &lean, &turn));      // missing colon
    CHECK(!CommandParser::parse_line("L:1.0,T:2.0garbage", &lean, &turn)); // trailing junk
    CHECK(!CommandParser::parse_line("L:,T:2.0", &lean, &turn));        // empty value
    CHECK(!CommandParser::parse_line("T:2.0,L:1.0", &lean, &turn));     // swapped order

    // A rejected line must leave the previous good values untouched.
    CHECK(CommandParser::parse_line("L:3.0,T:0.5", &lean, &turn));
    float keep_lean = lean, keep_turn = turn;
    CHECK(!CommandParser::parse_line("garbage", &lean, &turn));
    CHECK_NEAR(lean, keep_lean, 1e-6);
    CHECK_NEAR(turn, keep_turn, 1e-6);
}

// ===========================================================================
static void test_command_parser_stream() {
    SECTION("CommandParser byte stream");
    CommandParser p;

    const char* line = "L:1.50,T:-0.25\n";
    bool got = false;
    for (const char* c = line; *c; ++c) got = p.feed(*c);
    CHECK(got);
    CHECK_NEAR(p.lean_deg(), 1.50, 1e-5);
    CHECK_NEAR(p.turn(), -0.25, 1e-5);
    CHECK(p.good_lines() == 1);
    CHECK(p.bad_lines() == 0);

    // CRLF sender.
    got = false;
    for (const char* c = "L:2.00,T:0.10\r\n"; *c; ++c) got = p.feed(*c);
    CHECK(got);
    CHECK_NEAR(p.lean_deg(), 2.00, 1e-5);

    // Partial line interrupted by a reset mid-stream, then a good line. The
    // good line must still be accepted.
    p.feed('L'); p.feed(':'); p.feed('9');
    p.reset();
    got = false;
    for (const char* c = "L:0.50,T:0.00\n"; *c; ++c) got = p.feed(*c);
    CHECK(got);
    CHECK_NEAR(p.lean_deg(), 0.50, 1e-5);

    // Garbage line is counted, not accepted, and does not wedge the parser.
    const uint32_t bad_before = p.bad_lines();
    for (const char* c = "!!!noise!!!\n"; *c; ++c) p.feed(*c);
    CHECK(p.bad_lines() == bad_before + 1);
    CHECK_NEAR(p.lean_deg(), 0.50, 1e-5);     // unchanged

    // Over-long line is poisoned, and the parser recovers on the next line.
    for (int i = 0; i < 200; ++i) p.feed('A');
    CHECK(!p.feed('\n'));
    got = false;
    for (const char* c = "L:1.00,T:1.00\n"; *c; ++c) got = p.feed(*c);
    CHECK(got);
    CHECK_NEAR(p.lean_deg(), 1.00, 1e-5);
    CHECK_NEAR(p.turn(), 1.00, 1e-5);
}

// ===========================================================================
static void test_complementary_filter() {
    SECTION("ComplementaryFilter");
    ComplementaryFilter f;
    f.begin(0.75f, 0.f);

    // With a steady accelerometer reading and no rotation, the estimate must
    // converge to the accelerometer.
    for (int i = 0; i < 2000; ++i) f.update(10.f, 0.f, 0.005f);
    CHECK_NEAR(f.angle(), 10.0, 0.2);

    // Short-term: a gyro rate with a contradictory (stale) accelerometer
    // should move the estimate mostly with the gyro over a fraction of tau.
    ComplementaryFilter g;
    g.begin(0.75f, 0.f);
    for (int i = 0; i < 20; ++i) g.update(0.f, 100.f, 0.005f);   // 0.1s at 100dps
    CHECK(g.angle() > 7.0f);     // gyro says ~10 deg
    CHECK(g.angle() < 10.5f);    // accel pulls it back slightly

    // Same tau at a different loop rate must give nearly the same trajectory -
    // this is the property that makes tau safer than a raw coefficient.
    ComplementaryFilter a, b;
    a.begin(0.5f, 0.f);
    b.begin(0.5f, 0.f);
    for (int i = 0; i < 100; ++i) a.update(20.f, 0.f, 0.010f);   // 1.0s @100Hz
    for (int i = 0; i < 400; ++i) b.update(20.f, 0.f, 0.0025f);  // 1.0s @400Hz
    CHECK_NEAR(a.angle(), b.angle(), 0.5);

    CHECK_NEAR(f.rate(), 0.0, 1e-6);
    f.seed(-5.f);
    CHECK_NEAR(f.angle(), -5.0, 1e-6);
}

// ===========================================================================
static void test_pid() {
    SECTION("Pid");

    // P only.
    {
        Pid pid;
        pid.begin({2.f, 0.f, 0.f}, 255.f, 100.f);
        CHECK_NEAR(pid.update(0.f, -5.f, 0.f, 0.005f), 10.0, 1e-4);  // err=+5
        CHECK_NEAR(pid.update(0.f, +5.f, 0.f, 0.005f), -10.0, 1e-4);
    }

    // D acts on the measurement's rate and opposes it.
    {
        Pid pid;
        pid.begin({0.f, 0.f, 0.5f}, 255.f, 100.f);
        CHECK_NEAR(pid.update(0.f, 0.f, 20.f, 0.005f), -10.0, 1e-4);
        CHECK_NEAR(pid.update(0.f, 0.f, -20.f, 0.005f), 10.0, 1e-4);
    }

    // No derivative kick: stepping the SETPOINT must not produce a D spike.
    // This matters because an incoming vision command is exactly a setpoint
    // step.
    {
        Pid pid;
        pid.begin({1.f, 0.f, 10.f}, 255.f, 100.f);
        const float before = pid.update(0.f, 0.f, 0.f, 0.005f);
        const float after  = pid.update(5.f, 0.f, 0.f, 0.005f);
        CHECK_NEAR(before, 0.0, 1e-5);
        CHECK_NEAR(after, 5.0, 1e-5);   // pure P response, no 10*5/dt spike
    }

    // Integral accumulates toward its clamp.
    {
        Pid pid;
        pid.begin({0.f, 1.f, 0.f}, 255.f, 10.f);
        for (int i = 0; i < 10000; ++i) pid.update(1.f, 0.f, 0.f, 0.005f);
        CHECK_NEAR(pid.integral(), 10.0, 1e-3);     // clamped, not unbounded
    }

    // Anti-windup: with the output already saturated and the error still
    // pushing the same way, the integral must stop growing. This is the "held
    // the robot off the ground and it lurched when set down" bug.
    //
    // Gains chosen so that saturation is reached BY THE INTEGRAL (kp is small,
    // ki is large). With a large kp the P term alone saturates the output in
    // both directions, the integral correctly never moves at all, and the test
    // could not tell working anti-windup from a frozen integral.
    {
        Pid pid;
        pid.begin({1.f, 50.f, 0.f}, 255.f, 1000.f);

        for (int i = 0; i < 100; ++i) pid.update(50.f, 0.f, 0.f, 0.005f);
        const float wound = pid.integral();
        CHECK(wound > 1.f);                        // it really did wind up
        CHECK(wound < 1000.f);                     // but nowhere near i_limit

        // The output sits AT or JUST BELOW the clamp, not exactly on it: the
        // integral stops at the last value whose next step would have crossed
        // the limit, so the resting output can be up to one integral step
        // (ki * err * dt = 12.5 here) short of it.
        CHECK(pid.last_output() <= 255.f + 1e-3f);
        CHECK(pid.last_output() > 255.f - 50.f * 50.f * 0.005f - 1e-3f);

        for (int i = 0; i < 1000; ++i) pid.update(50.f, 0.f, 0.f, 0.005f);
        CHECK_NEAR(pid.integral(), wound, 1e-3);   // frozen while saturated

        // Once the error reverses the output leaves saturation, and the
        // integral must be free to unwind again - a windup guard that latches
        // is just a broken integrator.
        pid.update(-50.f, 0.f, 0.f, 0.005f);
        CHECK(pid.integral() < wound);
    }

    // Output clamp is respected in both directions.
    {
        Pid pid;
        pid.begin({1000.f, 0.f, 0.f}, 200.f, 10.f);
        CHECK_NEAR(pid.update(10.f, 0.f, 0.f, 0.005f), 200.0, 1e-3);
        CHECK_NEAR(pid.update(-10.f, 0.f, 0.f, 0.005f), -200.0, 1e-3);
    }

    // dt == 0 must not divide by zero or change state.
    {
        Pid pid;
        pid.begin({1.f, 1.f, 1.f}, 255.f, 10.f);
        const float a = pid.update(1.f, 0.f, 0.f, 0.005f);
        const float b = pid.update(1.f, 0.f, 0.f, 0.f);
        CHECK_NEAR(a, b, 1e-6);
    }
}

// ===========================================================================
static void test_safety_monitor() {
    SECTION("SafetyMonitor");
    SafetyMonitor s;
    s.begin(35.f, 4.f, 0.5f);

    // Must start disarmed - motors off at power-on, no exceptions.
    CHECK(!s.armed());
    CHECK(s.fallen());

    // Held upright but not yet long enough.
    for (int i = 0; i < 50; ++i) CHECK(!s.update(1.f, 0.005f));   // 0.25s
    CHECK(!s.armed());
    CHECK(s.arming_progress() > 0.4f);

    // Long enough -> arms.
    for (int i = 0; i < 60; ++i) s.update(1.f, 0.005f);
    CHECK(s.armed());
    CHECK(s.update(2.f, 0.005f));

    // Tilting past the fall angle disarms immediately and latches.
    CHECK(!s.update(40.f, 0.005f));
    CHECK(!s.armed());
    // Returning upright must NOT instantly re-arm; it needs the full dwell
    // again, so recovery is always a deliberate human action.
    CHECK(!s.update(0.f, 0.005f));
    for (int i = 0; i < 50; ++i) s.update(0.f, 0.005f);
    CHECK(!s.armed());
    for (int i = 0; i < 60; ++i) s.update(0.f, 0.005f);
    CHECK(s.armed());

    // Interrupting the dwell resets it.
    s.force_disarm();
    for (int i = 0; i < 80; ++i) s.update(1.f, 0.005f);
    s.update(20.f, 0.005f);          // knocked out of the band
    CHECK(s.arming_progress() < 0.1f);
    CHECK(!s.armed());
}

// ===========================================================================
static void test_mixing() {
    SECTION("mix_motors / deadzone / slew");

    // Straight ahead: both wheels equal.
    {
        const MotorOut m = mix_motors(100.f, 0.f, 80.f, 255.f);
        CHECK_NEAR(m.left, 100.0, 1e-4);
        CHECK_NEAR(m.right, 100.0, 1e-4);
    }

    // Turning adds a differential.
    {
        const MotorOut m = mix_motors(0.f, 1.f, 80.f, 255.f);
        CHECK_NEAR(m.left, 80.0, 1e-4);
        CHECK_NEAR(m.right, -80.0, 1e-4);
    }

    // Clipping must preserve the differential RATIO rather than clipping each
    // wheel separately (which would quietly change how hard it turns).
    {
        const MotorOut m = mix_motors(240.f, 1.f, 80.f, 255.f);
        CHECK(std::fabs(m.left) <= 255.0f + 1e-3f);
        CHECK(std::fabs(m.right) <= 255.0f + 1e-3f);
        const float want_ratio = (240.f - 80.f) / (240.f + 80.f);
        CHECK_NEAR(m.right / m.left, want_ratio, 1e-4);
    }

    // Deadzone maps the output above the PWM floor without changing sign.
    {
        CHECK_NEAR(apply_motor_deadzone(0.f, 40.f, 255.f), 0.0, 1e-6);
        const float small = apply_motor_deadzone(1.f, 40.f, 255.f);
        CHECK(small > 40.0f && small < 42.0f);
        CHECK_NEAR(apply_motor_deadzone(255.f, 40.f, 255.f), 255.0, 1e-3);
        CHECK(apply_motor_deadzone(-10.f, 40.f, 255.f) < 0.f);
    }

    // Slew limiting.
    CHECK_NEAR(slew_toward(0.f, 10.f, 1.f), 1.0, 1e-6);
    CHECK_NEAR(slew_toward(0.f, -10.f, 1.f), -1.0, 1e-6);
    CHECK_NEAR(slew_toward(0.f, 0.5f, 1.f), 0.5, 1e-6);
}

// ===========================================================================
static void test_velocity_estimator() {
    SECTION("VelocityEstimator");
    VelocityEstimator v;
    v.begin(1200.f, 0.05f);          // 1200 ticks/rev

    CHECK_NEAR(v.update(0, 0.005f), 0.0, 1e-6);    // first call primes

    // One rev per second => 1200 ticks/s => 6 ticks per 5ms tick.
    int32_t ticks = 0;
    for (int i = 0; i < 400; ++i) { ticks += 6; v.update(ticks, 0.005f); }
    CHECK_NEAR(v.velocity_rps(), 1.0, 0.05);

    // Stopping decays back toward zero.
    for (int i = 0; i < 400; ++i) v.update(ticks, 0.005f);
    CHECK_NEAR(v.velocity_rps(), 0.0, 0.02);
}

// ===========================================================================
// The important one: close the loop around a simulated inverted pendulum.
//
// Model (inverted pendulum driven by the horizontal acceleration of its base):
//
//     theta_ddot = ( g*sin(theta) - a_base*cos(theta) ) / L
//
// theta > 0 means leaning FORWARD. Accelerating the base forward (a_base > 0)
// produces a negative contribution, i.e. it rights the robot - which is the
// whole trick of balancing: drive the wheels toward the direction of the fall.
//
// This test runs the REAL ComplementaryFilter, the REAL Pid, and the REAL sign
// convention used in balance_controller.ino. If the sign convention were
// inverted, theta would run away to 90 degrees here instead of settling, and
// the assertions would fail on a laptop instead of on the floor.
static void test_closed_loop_balance() {
    SECTION("closed-loop balance simulation");

    const float g        = 9.81f;
    const float L        = 0.12f;   // pivot to centre of mass, metres
    const float k_motor  = 11.0f;   // m/s^2 of base acceleration at full PWM
    const float dt       = 0.005f;  // 200 Hz, matching the firmware
    const float max_out  = 255.f;

    // The starting gains documented in balance_controller.ino.
    const PidGains gains{18.0f, 60.0f, 1.2f};

    ComplementaryFilter filt;
    Pid pid;
    filt.begin(0.75f, 3.0f);
    pid.begin(gains, max_out, 40.f);

    float theta     = 3.0f * static_cast<float>(M_PI) / 180.f;  // 3 deg lean
    float theta_dot = 0.f;
    float worst     = 0.f;

    for (int step = 0; step < 1200; ++step) {    // 6 simulated seconds
        const float theta_deg = theta * 180.f / static_cast<float>(M_PI);
        const float rate_dps  = theta_dot * 180.f / static_cast<float>(M_PI);

        // The sensors the firmware actually reads. The accelerometer sees the
        // true tilt; the gyro sees the true rate. (Noise is exercised in the
        // filter tests above; this test is about the sign of the loop.)
        const float angle_est = filt.update(theta_deg, rate_dps, dt);

        // --- exactly what balance_controller.ino does ---------------------
        // Negated: the textbook PID form gives error = setpoint - measurement,
        // which is NEGATIVE for a forward lean, but a forward lean needs the
        // wheels driven FORWARD. See the comment at the call site in the .ino.
        const float u_pwm = -pid.update(0.f, angle_est, filt.rate(), dt);
        const float u     = clampf(u_pwm / max_out, -1.f, 1.f);

        // --- plant --------------------------------------------------------
        const float a_base    = k_motor * u;
        const float theta_ddot = (g * std::sin(theta) -
                                  a_base * std::cos(theta)) / L;
        theta_dot += theta_ddot * dt;
        theta     += theta_dot * dt;

        if (step > 20) {
            const float a = std::fabs(theta * 180.f / static_cast<float>(M_PI));
            if (a > worst) worst = a;
        }

        // Hard stop: if the sign is wrong this fires almost immediately.
        if (std::fabs(theta) > 1.4f) break;   // ~80 deg = fallen over
    }

    const float final_deg = std::fabs(theta * 180.f / static_cast<float>(M_PI));
    std::printf("    worst excursion %.2f deg, settled at %.3f deg\n",
                static_cast<double>(worst), static_cast<double>(final_deg));

    CHECK(final_deg < 1.0f);    // it balanced
    CHECK(worst < 12.0f);       // without a wild excursion on the way

    // And it must recover from a shove, not just from a gentle initial lean.
    theta_dot += 2.5f;          // kick: +2.5 rad/s
    float worst_after_kick = 0.f;
    for (int step = 0; step < 1200; ++step) {
        const float theta_deg = theta * 180.f / static_cast<float>(M_PI);
        const float rate_dps  = theta_dot * 180.f / static_cast<float>(M_PI);
        const float angle_est = filt.update(theta_deg, rate_dps, dt);
        const float u_pwm = -pid.update(0.f, angle_est, filt.rate(), dt);
        const float u     = clampf(u_pwm / max_out, -1.f, 1.f);
        const float a_base = k_motor * u;
        const float theta_ddot = (g * std::sin(theta) -
                                  a_base * std::cos(theta)) / L;
        theta_dot += theta_ddot * dt;
        theta     += theta_dot * dt;
        const float a = std::fabs(theta * 180.f / static_cast<float>(M_PI));
        if (a > worst_after_kick) worst_after_kick = a;
        if (std::fabs(theta) > 1.4f) break;
    }
    const float after_kick = std::fabs(theta * 180.f / static_cast<float>(M_PI));
    std::printf("    after a shove: worst %.2f deg, settled at %.3f deg\n",
                static_cast<double>(worst_after_kick),
                static_cast<double>(after_kick));
    CHECK(after_kick < 1.5f);

    // A non-zero lean setpoint - what a vision "lean forward" command does -
    // must drive the robot to that lean and hold it, not destabilise it.
    float hold = 0.f;
    for (int step = 0; step < 1200; ++step) {
        const float theta_deg = theta * 180.f / static_cast<float>(M_PI);
        const float rate_dps  = theta_dot * 180.f / static_cast<float>(M_PI);
        const float angle_est = filt.update(theta_deg, rate_dps, dt);
        const float u_pwm = -pid.update(2.0f, angle_est, filt.rate(), dt);
        const float u     = clampf(u_pwm / max_out, -1.f, 1.f);
        const float a_base = k_motor * u;
        const float theta_ddot = (g * std::sin(theta) -
                                  a_base * std::cos(theta)) / L;
        theta_dot += theta_ddot * dt;
        theta     += theta_dot * dt;
        hold = theta * 180.f / static_cast<float>(M_PI);
        if (std::fabs(theta) > 1.4f) break;
    }
    std::printf("    holding a +2.0 deg lean command: %.3f deg\n",
                static_cast<double>(hold));
    CHECK_NEAR(hold, 2.0, 0.5);
}

// ===========================================================================
int main() {
    std::printf("test_balance_core\n");
    test_parse_float();
    test_parse_line();
    test_command_parser_stream();
    test_complementary_filter();
    test_pid();
    test_safety_monitor();
    test_mixing();
    test_velocity_estimator();
    test_closed_loop_balance();
    return test_summary("test_balance_core");
}
