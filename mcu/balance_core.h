#pragma once
// ---------------------------------------------------------------------------
// The balance loop's logic, with every trace of Arduino removed.
//
// WHY THE FIRMWARE IS SPLIT THIS WAY:
// Everything that is hard to get right about this firmware is arithmetic and
// state machines - the sensor fusion, the PID's anti-windup, the serial line
// parser, the decision about when it is safe to energise the motors. None of
// that needs a microcontroller to test, but all of it is miserable to debug on
// one: the robot is on the floor, the only output is a serial cable, and a bug
// in the parser looks exactly like a bug in the PID looks exactly like a loose
// wire.
//
// So the logic lives here, as plain C++ with no #include <Arduino.h>, and
// test/test_balance_core.cpp drives it on a laptop at thousands of simulated
// timesteps per second - including the cases that are genuinely dangerous to
// reproduce deliberately, like "what happens if the robot is pushed past the
// fall threshold while the integral term is saturated".
//
// balance_controller.ino is then a thin shell: read the IMU, call into here,
// write the motors. It compiles for the Mega; this header compiles for both.
//
// Keep it free of Arduino.h, String, Serial, delay() and dynamic allocation -
// that constraint is what keeps it testable.
// ---------------------------------------------------------------------------

#include <math.h>
#include <stdint.h>
#include <string.h>

namespace reflex {

inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Moves `current` toward `target` by at most max_delta. Used on the incoming
// vision setpoint: the laptop already rate-limits it, but a corrupted packet
// or a restarted pipeline should not be able to step the balance setpoint.
// Defence in depth, 3 lines.
inline float slew_toward(float current, float target, float max_delta) {
    const float d = target - current;
    if (d >  max_delta) return current + max_delta;
    if (d < -max_delta) return current - max_delta;
    return target;
}

// ===========================================================================
// Sensor fusion
// ===========================================================================
//
// The accelerometer knows which way down is, but it cannot tell gravity from
// the robot's own acceleration, so it is correct on average and useless
// instantaneously. The gyro measures rotation rate precisely and
// instantaneously, but integrating it accumulates drift without bound.
//
// A complementary filter takes the gyro for the short term and the
// accelerometer for the long term:
//
//   angle = a * (angle + gyro * dt) + (1 - a) * accel_angle
//
// with `a` just under 1. Expressed as a time constant, a = tau / (tau + dt):
// disturbances faster than tau come from the gyro, slower than tau from the
// accelerometer. Using tau rather than a raw coefficient means the behaviour
// does not change if the loop rate changes - which it will, the first time
// anything is added to the loop.
class ComplementaryFilter {
public:
    void begin(float tau_s, float initial_angle_deg = 0.f) {
        tau_   = tau_s > 1e-4f ? tau_s : 1e-4f;
        angle_ = initial_angle_deg;
        rate_  = 0.f;
        primed_ = true;
    }

    // accel_angle_deg : tilt from the accelerometer alone
    // gyro_rate_dps   : rotation rate about the same axis, degrees/second
    float update(float accel_angle_deg, float gyro_rate_dps, float dt) {
        rate_ = gyro_rate_dps;
        if (!primed_) { begin(tau_, accel_angle_deg); return angle_; }
        if (dt <= 0.f) return angle_;

        const float a = tau_ / (tau_ + dt);
        angle_ = a * (angle_ + gyro_rate_dps * dt) + (1.f - a) * accel_angle_deg;
        return angle_;
    }

    // Snaps straight to the accelerometer reading. Used once at startup so the
    // filter does not spend its first second converging from zero.
    void seed(float accel_angle_deg) { angle_ = accel_angle_deg; primed_ = true; }

    float angle() const { return angle_; }
    float rate()  const { return rate_; }

private:
    float tau_    = 0.75f;
    float angle_  = 0.f;
    float rate_   = 0.f;
    bool  primed_ = false;
};

// ===========================================================================
// PID
// ===========================================================================
struct PidGains {
    float kp = 0.f, ki = 0.f, kd = 0.f;
};

class Pid {
public:
    void begin(PidGains g, float out_limit, float i_limit) {
        gains_     = g;
        out_limit_ = out_limit;
        i_limit_   = i_limit;
        reset();
    }

    void reset() { integral_ = 0.f; last_out_ = 0.f; }
    void set_gains(PidGains g) { gains_ = g; }
    PidGains gains() const { return gains_; }
    float integral() const { return integral_; }
    float last_output() const { return last_out_; }

    // measurement_rate is the measured derivative - here, the gyro's rate,
    // which is a direct sensor reading rather than a difference of two noisy
    // angle estimates. Differentiating the angle instead would amplify sensor
    // noise by 1/dt (200x at 200Hz) and is the usual reason a balance bot
    // buzzes.
    //
    // Two deliberate choices:
    //  * D acts on the measurement, not the error, so a step change in the
    //    setpoint (which is what an incoming vision command IS) does not
    //    produce a derivative spike - "derivative kick".
    //  * The integral only accumulates when doing so would not push an
    //    already-saturated output further into saturation. Without that, a
    //    robot held off the ground winds the integral up to enormous values and
    //    then lurches violently when set down.
    float update(float setpoint, float measurement, float measurement_rate,
                 float dt) {
        if (dt <= 0.f) return last_out_;

        const float err = setpoint - measurement;

        const float p_term = gains_.kp * err;
        const float d_term = -gains_.kd * measurement_rate;

        // Tentative integral, then decide whether to keep it.
        float candidate_i = integral_ + err * dt;
        candidate_i = clampf(candidate_i, -i_limit_, i_limit_);

        float out = p_term + gains_.ki * candidate_i + d_term;

        const bool saturated_high = out >  out_limit_;
        const bool saturated_low  = out < -out_limit_;
        const bool pushing_further =
            (saturated_high && err > 0.f) || (saturated_low && err < 0.f);

        if (!pushing_further) {
            integral_ = candidate_i;
        } else {
            // Recompute with the integral left where it was.
            out = p_term + gains_.ki * integral_ + d_term;
        }

        last_out_ = clampf(out, -out_limit_, out_limit_);
        return last_out_;
    }

private:
    PidGains gains_{};
    float    out_limit_ = 255.f;
    float    i_limit_   = 50.f;
    float    integral_  = 0.f;
    float    last_out_  = 0.f;
};

// ===========================================================================
// Serial command parser
// ===========================================================================
//
// Wire format (see SERIAL_PROTOCOL.md):   L:<lean_deg>,T:<turn>\n
//
// Requirements that shaped this:
//  * Byte at a time, never blocking. The balance loop cannot wait for a line
//    to finish arriving; it has a 5ms budget and the line may be split across
//    several loop iterations.
//  * No String, no malloc. Heap fragmentation on a 8KB AVR is a slow death.
//  * Hostile input is normal. Power-on garbage, a half-line from before a
//    reset, 5V noise on the UART - none of it may be accepted as a command or
//    wedge the parser. A line that does not parse is counted and dropped.
class CommandParser {
public:
    void reset() {
        len_ = 0;
        overflow_ = false;
    }

    // Returns true exactly when a complete, valid line has just been parsed;
    // lean_deg()/turn() are then updated.
    bool feed(char c) {
        if (c == '\r') return false;         // tolerate CRLF senders
        if (c != '\n') {
            if (len_ < kBufMax - 1) {
                buf_[len_++] = c;
            } else {
                overflow_ = true;           // line too long: poison it
            }
            return false;
        }

        // End of line.
        const bool was_overflow = overflow_;
        const uint8_t n = len_;
        buf_[n] = '\0';
        len_ = 0;
        overflow_ = false;

        if (was_overflow || n == 0) { ++bad_; return false; }

        float lean = 0.f, turn = 0.f;
        if (!parse_line(buf_, &lean, &turn)) { ++bad_; return false; }

        lean_ = lean;
        turn_ = turn;
        ++good_;
        return true;
    }

    float lean_deg() const { return lean_; }
    float turn()     const { return turn_; }
    uint32_t good_lines() const { return good_; }
    uint32_t bad_lines()  const { return bad_; }

    // Exposed for the unit tests, and usable standalone.
    //
    // Hand-rolled rather than sscanf/atof: sscanf's %f is not linked into AVR
    // builds by default (it needs -Wl,-u,vfscanf -lscanf_flt), and atof
    // follows the C locale's decimal separator. Both of those are
    // failure modes that appear only on the real hardware or only on one
    // teammate's machine. 30 lines of explicit parsing avoids the whole class
    // of problem and is directly testable.
    static bool parse_float(const char* s, const char** end, float* out) {
        if (!s || !out) return false;
        const char* p = s;
        while (*p == ' ' || *p == '\t') ++p;

        float sign = 1.f;
        if (*p == '+' || *p == '-') { if (*p == '-') sign = -1.f; ++p; }

        bool any_digit = false;
        float ip = 0.f;
        while (*p >= '0' && *p <= '9') {
            ip = ip * 10.f + static_cast<float>(*p - '0');
            ++p;
            any_digit = true;
        }

        float fp = 0.f;
        if (*p == '.') {
            ++p;
            float scale = 0.1f;
            while (*p >= '0' && *p <= '9') {
                fp += static_cast<float>(*p - '0') * scale;
                scale *= 0.1f;
                ++p;
                any_digit = true;
            }
        }

        if (!any_digit) return false;
        // Exponent notation is deliberately NOT accepted: the sender never
        // produces it, so seeing it means the line is corrupt.
        if (*p == 'e' || *p == 'E') return false;

        *out = sign * (ip + fp);
        if (end) *end = p;
        return true;
    }

    static bool parse_line(const char* s, float* lean, float* turn) {
        const char* p = s;
        while (*p == ' ') ++p;
        if (*p != 'L' && *p != 'l') return false;
        ++p;
        if (*p != ':') return false;
        ++p;
        if (!parse_float(p, &p, lean)) return false;

        while (*p == ' ') ++p;
        if (*p != ',') return false;
        ++p;
        while (*p == ' ') ++p;
        if (*p != 'T' && *p != 't') return false;
        ++p;
        if (*p != ':') return false;
        ++p;
        if (!parse_float(p, &p, turn)) return false;

        while (*p == ' ' || *p == '\t') ++p;
        return *p == '\0';   // trailing junk means a corrupt line
    }

private:
    static const uint8_t kBufMax = 32;
    char     buf_[kBufMax] = {0};
    uint8_t  len_ = 0;
    bool     overflow_ = false;
    float    lean_ = 0.f;
    float    turn_ = 0.f;
    uint32_t good_ = 0;
    uint32_t bad_  = 0;
};

// ===========================================================================
// Safety
// ===========================================================================
//
// Two rules, both learned the hard way by everyone who builds one of these:
//
//  1. Motors start DISABLED and stay disabled until the robot has been held
//     near-upright for a continuous dwell time. Otherwise the thing leaps off
//     the bench the instant it is plugged in, with hands still near the wheels.
//
//  2. Past the fall angle the motors cut out completely and latch off. A
//     fallen robot spinning its wheels at full PWM burns the driver, chews the
//     gearbox, and looks exactly like a robot that is about to right itself,
//     which tempts you to reach for it.
//
// Re-arming requires the same deliberate hold-upright gesture as a cold start,
// so recovery is always a decision a human made.
class SafetyMonitor {
public:
    void begin(float fall_deg, float arm_deg, float arm_dwell_s) {
        fall_deg_    = fall_deg;
        arm_deg_     = arm_deg;
        arm_dwell_s_ = arm_dwell_s;
        armed_       = false;
        dwell_       = 0.f;
    }

    // Returns true when the motors may be driven this tick.
    bool update(float angle_deg, float dt) {
        const float a = fabsf(angle_deg);

        if (armed_) {
            if (a > fall_deg_) {   // fallen
                armed_ = false;
                dwell_ = 0.f;
            }
            return armed_;
        }

        // Disarmed: accumulate dwell only while held inside the arming band.
        if (a <= arm_deg_) {
            dwell_ += dt;
            if (dwell_ >= arm_dwell_s_) {
                armed_ = true;
                dwell_ = 0.f;
            }
        } else {
            dwell_ = 0.f;
        }
        return armed_;
    }

    bool  armed() const { return armed_; }
    bool  fallen() const { return !armed_; }
    float arming_progress() const {
        return arm_dwell_s_ > 0.f ? clampf(dwell_ / arm_dwell_s_, 0.f, 1.f) : 1.f;
    }
    void force_disarm() { armed_ = false; dwell_ = 0.f; }

private:
    float fall_deg_    = 35.f;
    float arm_deg_     = 4.f;
    float arm_dwell_s_ = 0.5f;
    bool  armed_       = false;
    float dwell_       = 0.f;
};

// ===========================================================================
// Motor mixing
// ===========================================================================
struct MotorOut {
    float left = 0.f, right = 0.f;
};

// Balance output drives both wheels together; turn adds a differential.
//
// When the sum would clip, both wheels are scaled back by the same factor
// instead of being clipped independently. Independent clipping silently
// changes the differential - i.e. the robot turns by a different amount than
// commanded, but only when it is also leaning hard - which is an infuriating
// bug to chase. Scaling preserves the ratio and just reduces authority.
inline MotorOut mix_motors(float balance_out, float turn, float turn_authority,
                           float max_out) {
    const float diff = turn * turn_authority;
    MotorOut m;
    m.left  = balance_out + diff;
    m.right = balance_out - diff;

    const float peak = fabsf(m.left) > fabsf(m.right) ? fabsf(m.left)
                                                      : fabsf(m.right);
    if (peak > max_out && peak > 0.f) {
        const float k = max_out / peak;
        m.left  *= k;
        m.right *= k;
    }
    return m;
}

// Gear motors do not move at all below some PWM - they just hum and heat up.
// Mapping the live output range above that floor means small corrections
// actually produce motion, which matters because small corrections are what
// balancing is made of.
inline float apply_motor_deadzone(float out, float deadzone, float max_out) {
    if (fabsf(out) < 1e-3f) return 0.f;
    const float sign = out < 0.f ? -1.f : 1.f;
    const float mag  = fabsf(out);
    const float span = max_out - deadzone;
    if (span <= 0.f) return sign * max_out;
    return sign * (deadzone + mag * span / max_out);
}

// ===========================================================================
// Optional: wheel velocity from encoders
// ===========================================================================
// Angle-only balancing holds the robot upright but lets it drift across the
// room, because standing still and rolling at constant speed are both
// perfectly balanced states. Feeding wheel velocity back as a small damping
// term is what makes it hold position. Off by default - get the angle loop
// tuned first (see balance_controller.ino).
class VelocityEstimator {
public:
    void begin(float ticks_per_rev, float tau_s) {
        ticks_per_rev_ = ticks_per_rev > 0.f ? ticks_per_rev : 1.f;
        tau_ = tau_s > 1e-4f ? tau_s : 1e-4f;
        vel_ = 0.f;
        last_ticks_ = 0;
        primed_ = false;
    }

    // Call with the cumulative encoder count. Returns revolutions/second,
    // low-pass filtered (raw tick differences at 200Hz are extremely coarse -
    // often 0 or 1 ticks - so unfiltered they are nearly pure quantisation
    // noise).
    float update(int32_t cumulative_ticks, float dt) {
        if (!primed_) {
            last_ticks_ = cumulative_ticks;
            primed_ = true;
            return 0.f;
        }
        if (dt <= 0.f) return vel_;

        const int32_t d = cumulative_ticks - last_ticks_;
        last_ticks_ = cumulative_ticks;

        const float raw = (static_cast<float>(d) / ticks_per_rev_) / dt;
        const float a   = dt / (tau_ + dt);
        vel_ += a * (raw - vel_);
        return vel_;
    }

    float velocity_rps() const { return vel_; }

private:
    float   ticks_per_rev_ = 1.f;
    float   tau_  = 0.1f;
    float   vel_  = 0.f;
    int32_t last_ticks_ = 0;
    bool    primed_ = false;
};

} // namespace reflex
