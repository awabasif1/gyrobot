// ===========================================================================
// reflex-cv - Arduino Mega balance controller
// ===========================================================================
//
// This is the robot's reflexes. It has exactly one job that must never fail:
// keep the robot upright. Everything else - following a target, obeying the
// laptop - is strictly secondary and is allowed to degrade.
//
// Deliberately NOT in this file: anything slow, anything that waits. No
// delay(), no blocking serial reads, no I2C library that retries forever, no
// Serial.print() of floats inside the control path. The loop below has a 5ms
// budget (200 Hz) and a single blocking call anywhere in it is a fall.
//
// The arithmetic and state machines live in balance_core.h and are unit-tested
// on a laptop (see test/test_balance_core.cpp, which includes a closed-loop
// pendulum simulation). This file is only the hardware layer: read the IMU,
// call into that logic, drive the motors.
//
// ---------------------------------------------------------------------------
// WIRING (Arduino Mega 2560)
// ---------------------------------------------------------------------------
//
//   MPU6050 (I2C)                 TB6612FNG motor driver
//   ---------------               ----------------------
//   VCC -> 5V                     VM   -> LiPo +  (motor power, raw battery)
//   GND -> GND                    VCC  -> 5V      (logic, from the UBEC)
//   SDA -> pin 20                 GND  -> GND
//   SCL -> pin 21                 STBY -> pin 4
//   AD0 -> GND  (address 0x68)    PWMA -> pin 5    AIN1 -> 7   AIN2 -> 8
//                                 PWMB -> pin 6    BIN1 -> 9   BIN2 -> 10
//                                 AO1/AO2 -> left motor
//                                 BO1/BO2 -> right motor
//
//   ESP32-CAM link (UART)
//   ---------------------
//   Mega TX1 (pin 18) --[ LEVEL SHIFTER / divider ]--> ESP32 RX (GPIO 3)
//   Mega RX1 (pin 19) <-------------------------------- ESP32 TX (GPIO 1)
//   GND -------------------------------------------------- GND
//
//   *** THE LEVEL SHIFTER IS NOT OPTIONAL ON THE MEGA->ESP32 DIRECTION. ***
//   The Mega's TX idles at 5V; the ESP32's GPIO is rated 3.3V. Feeding 5V
//   into it will damage the ESP32, sometimes immediately and sometimes a week
//   later, which is worse. A 2-resistor divider (1k from Mega TX to ESP32 RX,
//   2k from ESP32 RX to GND) is enough. The ESP32->Mega direction needs
//   nothing: 3.3V is a valid logic HIGH for a 5V AVR input.
//
//   Note pins 18/19 are Serial1's TX/RX. Serial (USB) stays free for
//   debugging, which is the whole reason for using a Mega rather than an Uno.
//
// ---------------------------------------------------------------------------
// BRING-UP ORDER - do not skip steps
// ---------------------------------------------------------------------------
//  1. Run mcu/imu_calibrate.ino first. It tells you the gyro bias values and
//     lets you confirm the ANGLE SIGN by tilting the robot by hand. Copy its
//     output into the constants below. Nothing else will work until the sign
//     is right.
//  2. Prop the robot up so the wheels are off the ground. Power on. Confirm
//     over USB that `angle` tracks your hand and that the motors spin in the
//     direction that WOULD correct the lean. If a wheel turns the wrong way,
//     flip its INVERT flag below - do not rewire and do not flip the PID sign.
//  3. Tune in the order P, then D, then I. See TUNING at the bottom.
//  4. Only then connect the ESP32-CAM.
// ===========================================================================

#include <Arduino.h>
#include <Wire.h>

#include "balance_core.h"

using namespace reflex;

// ===========================================================================
// Configuration
// ===========================================================================

// ---- loop rate ----
// 200 Hz. The pendulum's own fall timescale is ~sqrt(L/g) ~ 0.11s for a
// 12cm centre of mass, so 5ms per correction is comfortably inside it. Going
// faster mostly buys noise; going much slower makes the D term useless.
static const uint32_t LOOP_HZ        = 200;
static const uint32_t LOOP_PERIOD_US = 1000000UL / LOOP_HZ;

// ---- IMU ----
static const uint8_t  MPU_ADDR         = 0x68;
static const float    GYRO_LSB_PER_DPS = 65.5f;   // matches FS_SEL=1 (+/-500 dps)

// Sign of the derived pitch angle and gyro rate. These depend ENTIRELY on how
// the IMU happens to be bolted on, which is why imu_calibrate.ino exists:
// it prints the angle live so you can tilt the robot forward and see whether
// the number goes positive. Required convention:
//
//        angle > 0  <=>  robot leaning FORWARD (in the direction it drives
//                        when both motors get a positive command)
//        rate  > 0  <=>  angle increasing
//
// Get this wrong and the robot drives itself over, confidently.
static const float IMU_ANGLE_SIGN = +1.0f;
static const float IMU_RATE_SIGN  = +1.0f;

// Gyro zero-rate bias, in raw LSB. From imu_calibrate.ino. A gyro's bias is
// specific to the individual chip and drifts with temperature, so these are
// also re-measured at every boot (see calibrate_gyro()); the constants here
// are only a sanity fallback.
static const float GYRO_BIAS_FALLBACK = 0.0f;

// Mechanical zero: the angle reading when the robot is actually balanced.
// Almost never 0 - the IMU is never mounted perfectly square and the centre of
// mass is never exactly over the axle. Find it by letting the robot balance
// and watching which way it creeps: if it consistently drifts forward, nudge
// this positive.
static const float ANGLE_TRIM_DEG = 0.0f;

// ---- filter ----
static const float FILTER_TAU_S = 0.75f;

// ---- PID ----
// Starting gains. These are the values the closed-loop simulation in
// test/test_balance_core.cpp stabilises with for a 12cm centre of mass, so
// they are a sane starting point rather than a guess - but every chassis is
// different and you WILL retune. See TUNING at the bottom.
static const float PID_KP = 18.0f;
static const float PID_KI = 60.0f;
static const float PID_KD = 1.2f;

static const float PID_OUT_LIMIT = 255.0f;   // one full PWM range
static const float PID_I_LIMIT   = 40.0f;

// ---- safety ----
static const float FALL_ANGLE_DEG = 35.0f;  // past this, motors cut and latch
static const float ARM_ANGLE_DEG  = 4.0f;   // must be held inside this band...
static const float ARM_DWELL_S    = 0.75f;  // ...for this long, to arm

// ---- vision command handling ----
static const uint32_t CMD_TIMEOUT_MS   = 400;  // no command for this long -> zero it
static const float MAX_LEAN_CMD_DEG    = 5.0f;
static const float MAX_TURN_CMD        = 1.0f;
static const float LEAN_SLEW_DEG_PER_S = 8.0f; // second line of defence; the
static const float TURN_SLEW_PER_S     = 3.0f; // laptop already rate-limits

// How much PWM differential a full-scale turn command is worth. Low on
// purpose: a balancing robot that tries to turn hard loses the wheel authority
// it needs to stay upright, and falls while turning.
static const float TURN_AUTHORITY = 60.0f;

// ---- motors ----
static const uint8_t PIN_STBY = 4;
static const uint8_t PIN_PWMA = 5,  PIN_AIN1 = 7, PIN_AIN2 = 8;    // left
static const uint8_t PIN_PWMB = 6,  PIN_BIN1 = 9, PIN_BIN2 = 10;   // right

// Flip these if a wheel spins the wrong way. Changing them is always the right
// fix; inverting the PID output is never the right fix, because that would
// also invert the balance response.
static const bool LEFT_INVERT  = false;
static const bool RIGHT_INVERT = false;

// Gear motors stall below some PWM. Measure yours: ramp a wheel up off the
// ground and note where it starts turning reliably. Too high makes the robot
// jittery; too low makes small corrections do nothing.
static const float MOTOR_DEADZONE = 35.0f;

// 0 = stock Arduino PWM (~490 Hz on pins 5/6): audible whine, but known-good.
// 1 = ~62.5 kHz, inaudible and kinder to the gearboxes. Turn this on only
//     AFTER the robot balances, so that if something changes you know what
//     changed. It reconfigures Timer3 and Timer4 only; Timer0 (millis) and
//     the Serial ports are untouched.
#define QUIET_PWM 0

// 0 = angle-only balancing (what the build guide's step 6 asks for).
// 1 = also feed wheel velocity back as a damping term, which stops the robot
//     drifting slowly across the room. Angle-only balancing cannot tell
//     "standing still" from "rolling at constant speed" - both are upright.
//     Get angle-only working and tuned FIRST.
#define ENABLE_ENCODER_DAMPING 0

#if ENABLE_ENCODER_DAMPING
// Must be interrupt-capable on the Mega: 2, 3, 18, 19 (20/21 are the I2C bus).
// 18/19 are Serial1 here, so the A channels use 2 and 3.
static const uint8_t PIN_ENC_LA = 2,  PIN_ENC_LB = 16;
static const uint8_t PIN_ENC_RA = 3,  PIN_ENC_RB = 17;
static const float   ENC_TICKS_PER_REV = 1200.0f;  // CHECK YOUR MOTOR'S SPEC
static const float   VEL_KD            = 25.0f;    // PWM per rev/s - tune last
#endif

// ===========================================================================
// State
// ===========================================================================
static ComplementaryFilter g_filter;
static Pid                 g_pid;
static CommandParser       g_parser;
static SafetyMonitor       g_safety;

static float    g_gyro_bias     = GYRO_BIAS_FALLBACK;
static uint32_t g_last_tick_us  = 0;
static uint32_t g_last_cmd_ms   = 0;
static uint32_t g_last_telem_ms = 0;
static uint32_t g_overruns      = 0;

static float g_lean_applied = 0.0f;   // slew-limited versions of the commanded
static float g_turn_applied = 0.0f;   // values, which is what the PID sees

static bool  g_imu_ok = false;

#if ENABLE_ENCODER_DAMPING
static volatile int32_t g_ticks_l = 0;
static volatile int32_t g_ticks_r = 0;
static VelocityEstimator g_vel_l, g_vel_r;

static void isr_enc_l() {
    // Single-edge quadrature: the B channel's level at the A edge gives the
    // direction. Half the resolution of full quadrature decoding, which is
    // plenty for a damping term and costs a quarter of the interrupt load.
    g_ticks_l += (digitalRead(PIN_ENC_LB) ? 1 : -1);
}
static void isr_enc_r() {
    g_ticks_r += (digitalRead(PIN_ENC_RB) ? 1 : -1);
}
#endif

// ===========================================================================
// IMU - direct register access, no library
// ===========================================================================
// Written against the datasheet rather than pulling in a library, for two
// reasons: most MPU6050 libraries wrap a blocking read with retries (fine in a
// sketch, fatal in a 5ms control loop), and the register map is 6 writes and
// one burst read. There is nothing a library would add except a dependency and
// an unknown worst-case latency.

static bool mpu_write(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);
    Wire.write(val);
    return Wire.endTransmission() == 0;
}

static bool mpu_init() {
    Wire.begin();
    // 400 kHz: a 14-byte burst read takes ~0.4ms instead of ~1.4ms at 100kHz.
    // At 200 Hz that is the difference between 8% and 28% of the loop budget
    // spent waiting on I2C.
    Wire.setClock(400000);

    // Wake up, clock from the X gyro PLL (more stable than the internal
    // oscillator, which is what 0x00 would select).
    if (!mpu_write(0x6B, 0x01)) return false;
    delay(10);

    mpu_write(0x1A, 0x03);  // CONFIG: DLPF ~44 Hz. Cuts motor/gearbox vibration
                            // before it reaches the filter. Without this the
                            // accelerometer channel is mostly chassis buzz.
    mpu_write(0x19, 0x04);  // SMPLRT_DIV: 1kHz/(1+4) = 200 Hz, matching LOOP_HZ
    mpu_write(0x1B, 0x08);  // GYRO_CONFIG: FS_SEL=1 -> +/-500 dps. +/-250 would
                            // give finer resolution but clips during a fall.
    mpu_write(0x1C, 0x00);  // ACCEL_CONFIG: AFS_SEL=0 -> +/-2g, best resolution

    // WHO_AM_I should read 0x68.
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x75);
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1) != 1) return false;
    const uint8_t who = Wire.read();
    return who == 0x68 || who == 0x72;   // 0x72 = MPU6500/9250 clone
}

// Reads one big-endian 16-bit sample off the wire.
//
// This exists as a function rather than being written inline as
// `(Wire.read() << 8) | Wire.read()`, which is the obvious way to write it and
// is WRONG: the order in which C++ evaluates the two operands of `|` is
// unspecified, so the compiler is free to read the low byte first. That
// produces byte-swapped samples - garbage angles - and it can change between
// compiler versions or optimisation levels, which makes it a nightmare to
// track down. Two separate statements are sequenced, so this is correct by
// construction.
static int16_t wire_read16() {
    const uint8_t hi = Wire.read();
    const uint8_t lo = Wire.read();
    return (int16_t)(((uint16_t)hi << 8) | (uint16_t)lo);
}

// Reads accel XYZ and gyro XYZ in one burst. Returns false if the bus did not
// hand back a full frame, in which case the caller reuses the previous sample
// rather than feeding a partially-updated reading into the filter.
static bool mpu_read(int16_t* ax, int16_t* ay, int16_t* az,
                     int16_t* gx, int16_t* gy, int16_t* gz) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);                       // ACCEL_XOUT_H
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14) != 14) return false;

    *ax = wire_read16();
    *ay = wire_read16();
    *az = wire_read16();
    (void)wire_read16();                     // temperature, unused
    *gx = wire_read16();
    *gy = wire_read16();
    *gz = wire_read16();
    return true;
}

// Averages the gyro at rest to find its zero-rate offset. Every MPU6050 has
// one, it is different per chip, and it drifts with temperature - so this runs
// at every boot rather than trusting a constant. An uncorrected bias of even
// 2 dps makes the complementary filter lean on the accelerometer to cancel a
// phantom rotation, which shows up as the robot slowly creeping.
static void calibrate_gyro() {
    const int N = 600;              // ~3s at 200 Hz
    float sum = 0.0f;
    int   got = 0;
    for (int i = 0; i < N; ++i) {
        int16_t ax, ay, az, gx, gy, gz;
        if (mpu_read(&ax, &ay, &az, &gx, &gy, &gz)) {
            sum += (float)gy;       // the pitch-axis gyro; see IMU_RATE_SIGN
            ++got;
        }
        delay(5);
    }
    if (got > N / 2) g_gyro_bias = sum / (float)got;
}

// ===========================================================================
// Motors
// ===========================================================================
static void motors_init() {
    pinMode(PIN_STBY, OUTPUT);
    pinMode(PIN_PWMA, OUTPUT); pinMode(PIN_AIN1, OUTPUT); pinMode(PIN_AIN2, OUTPUT);
    pinMode(PIN_PWMB, OUTPUT); pinMode(PIN_BIN1, OUTPUT); pinMode(PIN_BIN2, OUTPUT);
    digitalWrite(PIN_STBY, LOW);             // start in standby = coasting
    analogWrite(PIN_PWMA, 0);
    analogWrite(PIN_PWMB, 0);

#if QUIET_PWM
    // Fast PWM, 8-bit, prescaler 1 -> 16MHz/256 = 62.5 kHz on pins 5 (Timer3A)
    // and 6 (Timer4A). Above hearing, and well inside the TB6612FNG's 100 kHz
    // input rating. analogWrite() keeps working normally on these pins.
    TCCR3A = (TCCR3A & 0xFC) | _BV(WGM30);
    TCCR3B = (TCCR3B & 0xE0) | _BV(WGM32) | _BV(CS30);
    TCCR4A = (TCCR4A & 0xFC) | _BV(WGM40);
    TCCR4B = (TCCR4B & 0xE0) | _BV(WGM42) | _BV(CS40);
#endif
}

static void motor_write(uint8_t pwm_pin, uint8_t in1, uint8_t in2, float v,
                        bool invert) {
    if (invert) v = -v;
    const bool forward = (v >= 0.0f);
    float mag = v >= 0.0f ? v : -v;
    if (mag > 255.0f) mag = 255.0f;

    digitalWrite(in1, forward ? HIGH : LOW);
    digitalWrite(in2, forward ? LOW : HIGH);
    analogWrite(pwm_pin, (int)mag);
}

static void motors_coast() {
    digitalWrite(PIN_STBY, LOW);     // STBY low = both bridges high-impedance
    analogWrite(PIN_PWMA, 0);
    analogWrite(PIN_PWMB, 0);
}

static void motors_drive(float left, float right) {
    digitalWrite(PIN_STBY, HIGH);
    motor_write(PIN_PWMA, PIN_AIN1, PIN_AIN2, left,  LEFT_INVERT);
    motor_write(PIN_PWMB, PIN_BIN1, PIN_BIN2, right, RIGHT_INVERT);
}

// ===========================================================================
// setup
// ===========================================================================
void setup() {
    Serial.begin(115200);    // USB, for humans
    Serial1.begin(115200);   // ESP32-CAM bridge

    motors_init();           // FIRST: get the driver into standby before
                             // anything else can take time or fail

    Serial.println(F("reflex-cv balance controller"));

    g_imu_ok = mpu_init();
    if (!g_imu_ok) {
        Serial.println(F("FATAL: MPU6050 not responding. Check SDA=20, SCL=21, "
                         "3.3-5V, AD0->GND."));
        // Deliberately does not proceed: a balance loop with no angle sensor
        // would drive the motors on garbage. Blink and wait for a power cycle.
        pinMode(LED_BUILTIN, OUTPUT);
        for (;;) {
            digitalWrite(LED_BUILTIN, HIGH); delay(120);
            digitalWrite(LED_BUILTIN, LOW);  delay(120);
        }
    }

    Serial.println(F("calibrating gyro - hold the robot COMPLETELY STILL"));
    calibrate_gyro();
    Serial.print(F("gyro bias (raw LSB): "));
    Serial.println(g_gyro_bias, 2);

    g_filter.begin(FILTER_TAU_S, 0.0f);
    g_pid.begin({PID_KP, PID_KI, PID_KD}, PID_OUT_LIMIT, PID_I_LIMIT);
    g_safety.begin(FALL_ANGLE_DEG, ARM_ANGLE_DEG, ARM_DWELL_S);

    // Seed the filter from the accelerometer so it starts at the true angle
    // instead of converging from zero over the first second.
    {
        int16_t ax, ay, az, gx, gy, gz;
        if (mpu_read(&ax, &ay, &az, &gx, &gy, &gz)) {
            const float a = IMU_ANGLE_SIGN *
                            atan2f((float)ax, (float)az) * 57.29578f;
            g_filter.seed(a - ANGLE_TRIM_DEG);
        }
    }

#if ENABLE_ENCODER_DAMPING
    pinMode(PIN_ENC_LA, INPUT_PULLUP); pinMode(PIN_ENC_LB, INPUT_PULLUP);
    pinMode(PIN_ENC_RA, INPUT_PULLUP); pinMode(PIN_ENC_RB, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(PIN_ENC_LA), isr_enc_l, RISING);
    attachInterrupt(digitalPinToInterrupt(PIN_ENC_RA), isr_enc_r, RISING);
    g_vel_l.begin(ENC_TICKS_PER_REV, 0.05f);
    g_vel_r.begin(ENC_TICKS_PER_REV, 0.05f);
#endif

    Serial.println(F("hold the robot upright to arm the motors"));
    g_last_tick_us  = micros();
    g_last_cmd_ms   = millis();
    g_last_telem_ms = millis();
}

// ===========================================================================
// loop
// ===========================================================================
void loop() {
    // ---- 1. drain the command UART (never blocks) -----------------------
    // Bounded per iteration so a flood of bytes cannot starve the control
    // tick. At 115200 baud, ~58 bytes can arrive per 5ms tick; 64 covers it
    // with margin while still being a hard ceiling.
    for (int i = 0; i < 64 && Serial1.available(); ++i) {
        if (g_parser.feed((char)Serial1.read())) {
            g_last_cmd_ms = millis();
        }
    }

    // ---- 2. fixed-rate control tick --------------------------------------
    const uint32_t now_us = micros();
    // Unsigned subtraction, so this stays correct across the ~71 minute
    // micros() rollover. Comparing (now > deadline) would not.
    if ((uint32_t)(now_us - g_last_tick_us) < LOOP_PERIOD_US) return;

    const uint32_t elapsed_us = now_us - g_last_tick_us;
    float dt = (float)elapsed_us * 1e-6f;

    // Advance by exactly one period to avoid accumulating drift. If we have
    // fallen more than a period behind (a long interrupt, a slow I2C frame),
    // resync to now instead of trying to catch up with a burst of ticks -
    // catching up would run the filter and PID with tiny dt values and spike
    // the derivative.
    g_last_tick_us += LOOP_PERIOD_US;
    if ((uint32_t)(now_us - g_last_tick_us) > LOOP_PERIOD_US) {
        g_last_tick_us = now_us;
        ++g_overruns;
    }
    if (dt > 0.05f) dt = 0.05f;   // clamp a pathological gap

    // ---- 3. read the IMU -------------------------------------------------
    static float s_angle_accel = 0.0f;
    static float s_rate_dps    = 0.0f;
    {
        int16_t ax, ay, az, gx, gy, gz;
        if (mpu_read(&ax, &ay, &az, &gx, &gy, &gz)) {
            // atan2 of the two axes in the plane the robot tips through. Using
            // atan2 rather than asin(ax/g) keeps it correct past 45 degrees,
            // which matters for fall detection even though the balance range is
            // only a few degrees.
            s_angle_accel = IMU_ANGLE_SIGN *
                            atan2f((float)ax, (float)az) * 57.29578f;
            s_rate_dps = IMU_RATE_SIGN *
                         (((float)gy - g_gyro_bias) / GYRO_LSB_PER_DPS);
        }
        // On a failed read both values keep their previous contents on
        // purpose: one stale sample is far safer than a zero or a half-updated
        // one, and the next tick is 5ms away.
    }

    const float angle = g_filter.update(s_angle_accel - ANGLE_TRIM_DEG,
                                        s_rate_dps, dt);

    // ---- 4. safety -------------------------------------------------------
    const bool armed = g_safety.update(angle, dt);

    // ---- 5. vision command, with a timeout -------------------------------
    // If the laptop or the WiFi link dies, the robot must stop steering and go
    // back to simply standing still - not keep executing the last turn it
    // heard about forever.
    const bool cmd_fresh = (uint32_t)(millis() - g_last_cmd_ms) < CMD_TIMEOUT_MS;
    const float lean_target = cmd_fresh
        ? clampf(g_parser.lean_deg(), -MAX_LEAN_CMD_DEG, MAX_LEAN_CMD_DEG)
        : 0.0f;
    const float turn_target = cmd_fresh
        ? clampf(g_parser.turn(), -MAX_TURN_CMD, MAX_TURN_CMD)
        : 0.0f;

    g_lean_applied = slew_toward(g_lean_applied, lean_target,
                                 LEAN_SLEW_DEG_PER_S * dt);
    g_turn_applied = slew_toward(g_turn_applied, turn_target,
                                 TURN_SLEW_PER_S * dt);

    // ---- 6. control ------------------------------------------------------
    if (!armed) {
        motors_coast();
        // Reset the integrator while disarmed. Otherwise it accumulates the
        // whole time the robot lies on its side and dumps that into the motors
        // the moment it re-arms.
        g_pid.reset();
        g_lean_applied = 0.0f;
        g_turn_applied = 0.0f;
    } else {
        // The setpoint is the commanded lean: asking the robot to lean forward
        // IS how you ask it to drive forward. There is no separate throttle.
        //
        // NOTE THE MINUS SIGN. The PID is the textbook form, so its error is
        // (setpoint - measurement), which is NEGATIVE when the robot has
        // pitched forward past its setpoint. But a forward pitch has to be
        // caught by driving the wheels FORWARD, i.e. with a positive command.
        // Negating closes the loop the right way round. The closed-loop
        // simulation in test/test_balance_core.cpp exists specifically to pin
        // this down - with the sign flipped, it diverges.
        float u = -g_pid.update(g_lean_applied, angle, g_filter.rate(), dt);

#if ENABLE_ENCODER_DAMPING
        // Oppose wheel motion a little, so "upright and rolling" is no longer
        // an equilibrium and the robot holds its ground.
        noInterrupts();
        const int32_t tl = g_ticks_l, tr = g_ticks_r;
        interrupts();
        const float vl = g_vel_l.update(tl, dt);
        const float vr = g_vel_r.update(tr, dt);
        u -= VEL_KD * 0.5f * (vl + vr);
        u = clampf(u, -PID_OUT_LIMIT, PID_OUT_LIMIT);
#endif

        MotorOut m = mix_motors(u, g_turn_applied, TURN_AUTHORITY,
                                PID_OUT_LIMIT);
        m.left  = apply_motor_deadzone(m.left,  MOTOR_DEADZONE, PID_OUT_LIMIT);
        m.right = apply_motor_deadzone(m.right, MOTOR_DEADZONE, PID_OUT_LIMIT);
        motors_drive(m.left, m.right);
    }

    // ---- 7. telemetry ----------------------------------------------------
    // 20 Hz. Format is fixed by SERIAL_PROTOCOL.md. dtostrf rather than
    // Serial.print(float) because it writes into a buffer we send in one go,
    // which keeps the formatting cost off the control path.
    if ((uint32_t)(millis() - g_last_telem_ms) >= 50) {
        g_last_telem_ms = millis();
        char line[32];
        char abuf[10];
        dtostrf(angle, 1, 2, abuf);
        snprintf(line, sizeof(line), "A:%s,F:%d\n", abuf, armed ? 0 : 1);
        Serial1.print(line);
    }

    // ---- 8. human-readable debug on USB only -----------------------------
    // 5 Hz, and never on the ESP32 link. Printing is slow; this is outside
    // the control path and deliberately sparse.
    static uint32_t s_last_dbg = 0;
    if ((uint32_t)(millis() - s_last_dbg) >= 200) {
        s_last_dbg = millis();
        Serial.print(F("ang=")); Serial.print(angle, 2);
        Serial.print(F(" rate=")); Serial.print(g_filter.rate(), 1);
        Serial.print(F(" set=")); Serial.print(g_lean_applied, 2);
        Serial.print(F(" turn=")); Serial.print(g_turn_applied, 2);
        Serial.print(F(" out=")); Serial.print(g_pid.last_output(), 0);
        Serial.print(armed ? F(" ARMED") : F(" disarmed"));
        if (!armed) {
            Serial.print(F(" arm%=")); Serial.print(g_safety.arming_progress()*100.0f, 0);
        }
        if (!cmd_fresh) Serial.print(F(" [no cmd]"));
        Serial.print(F(" rx=")); Serial.print(g_parser.good_lines());
        Serial.print(F("/")); Serial.print(g_parser.bad_lines());
        if (g_overruns) { Serial.print(F(" ovr=")); Serial.print(g_overruns); }
        Serial.println();
    }
}

// ===========================================================================
// TUNING
// ===========================================================================
//
// Do this with the robot held so the wheels just touch the ground, and a hand
// ready. Change ONE gain at a time.
//
// 0. Confirm the sign first. Disarmed, tilt the robot forward by hand and
//    watch `ang` on USB: it must go POSITIVE. Then arm it and tilt forward
//    slightly: the wheels must spin FORWARD. If the angle sign is wrong, flip
//    IMU_ANGLE_SIGN. If a single wheel is wrong, flip that wheel's INVERT.
//    Nothing below will work until both are right.
//
// 1. Set PID_KI = 0 and PID_KD = 0. Raise PID_KP until the robot oscillates
//    steadily around upright - not until it balances, until it OSCILLATES.
//    Then back off about 30%.
//
// 2. Raise PID_KD until the oscillation damps out and the robot feels stiff
//    against a gentle push. Too much KD makes it buzz and twitch at the
//    gearbox's resonance; back off if it sounds angry.
//
// 3. Add PID_KI, small, until the slow drift in one direction stops. KI
//    fights the steady-state lean that KP alone cannot remove. Too much makes
//    a slow wallowing oscillation over a second or two - much slower than the
//    KP oscillation, which is how you tell them apart.
//
// 4. Set ANGLE_TRIM_DEG to kill any remaining consistent creep.
//
// 5. Only now connect the ESP32-CAM. The balance gains should not need to
//    change when vision commands arrive - if they do, the commands are too
//    aggressive, so lower them in the laptop's ShaperConfig
//    (src/command_shaper.h), not here.
//
// If `ovr=` climbs steadily in the debug line, the loop is missing its
// deadline: something in it is taking longer than 5ms. The usual culprit is
// I2C at 100kHz instead of 400kHz, or debug printing that got moved into the
// control path.
// ===========================================================================
