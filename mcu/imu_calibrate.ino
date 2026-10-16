// ===========================================================================
// reflex-cv - IMU calibration and sign check
// ===========================================================================
//
// RUN THIS BEFORE balance_controller.ino. It answers the three questions that
// have to be settled before a balance loop can work at all:
//
//   1. Is the MPU6050 actually wired up and talking?
//   2. Which SIGN does the pitch angle have when the robot leans forward?
//   3. What is this particular gyro's zero-rate bias?
//
// Question 2 is the one that matters most. The angle's sign depends entirely
// on how the IMU happens to be bolted to the chassis, and there is no way to
// know it from the code. Get it wrong and the balance controller drives the
// robot over instead of catching it - and it does so smoothly and
// confidently, which makes it look like a tuning problem rather than a sign
// problem. People lose days to this.
//
// Upload this, open the Serial Monitor at 115200, and follow the prompts.
//
// (The MPU register code is duplicated from balance_controller.ino rather than
// shared. That is deliberate: these are two independent Arduino sketches, and
// a calibration tool that depends on the flight firmware is a tool you cannot
// use to debug the flight firmware.)
// ===========================================================================

#include <Arduino.h>
#include <Wire.h>

static const uint8_t MPU_ADDR         = 0x68;
static const float   GYRO_LSB_PER_DPS = 65.5f;   // FS_SEL=1, +/-500 dps
static const float   ACCEL_LSB_PER_G  = 16384.0f; // AFS_SEL=0, +/-2g

// Measured in setup(), used by loop(). Declared up here because the Arduino
// IDE only auto-generates forward declarations for FUNCTIONS, not variables -
// a global used in setup() but declared below it does not compile.
static float g_bias_y = 0.0f;

static bool mpu_write(uint8_t reg, uint8_t val) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(reg);
    Wire.write(val);
    return Wire.endTransmission() == 0;
}

// Separate statements, because the order of evaluation of `|`'s operands is
// unspecified in C++ - see the long note in balance_controller.ino.
static int16_t wire_read16() {
    const uint8_t hi = Wire.read();
    const uint8_t lo = Wire.read();
    return (int16_t)(((uint16_t)hi << 8) | (uint16_t)lo);
}

static bool mpu_read(int16_t* ax, int16_t* ay, int16_t* az,
                     int16_t* gx, int16_t* gy, int16_t* gz) {
    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x3B);
    if (Wire.endTransmission(false) != 0) return false;
    if (Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)14) != 14) return false;
    *ax = wire_read16(); *ay = wire_read16(); *az = wire_read16();
    (void)wire_read16();
    *gx = wire_read16(); *gy = wire_read16(); *gz = wire_read16();
    return true;
}

void setup() {
    Serial.begin(115200);
    while (!Serial) { /* native-USB boards only; harmless on the Mega */ }

    Wire.begin();
    Wire.setClock(400000);

    Serial.println(F("\n=== reflex-cv IMU calibration ==="));

    if (!mpu_write(0x6B, 0x01)) {
        Serial.println(F("FATAL: no response at address 0x68."));
        Serial.println(F("  check: SDA -> pin 20, SCL -> pin 21, VCC -> 5V,"));
        Serial.println(F("         GND -> GND, AD0 -> GND"));
        for (;;) delay(1000);
    }
    delay(10);
    mpu_write(0x1A, 0x03);   // DLPF ~44 Hz
    mpu_write(0x19, 0x04);   // 200 Hz sample rate
    mpu_write(0x1B, 0x08);   // gyro +/-500 dps
    mpu_write(0x1C, 0x00);   // accel +/-2g

    Wire.beginTransmission(MPU_ADDR);
    Wire.write(0x75);                       // WHO_AM_I
    Wire.endTransmission(false);
    Wire.requestFrom((uint8_t)MPU_ADDR, (uint8_t)1);
    Serial.print(F("WHO_AM_I = 0x"));
    Serial.println(Wire.read(), HEX);
    Serial.println(F("  (0x68 = genuine MPU6050, 0x70/0x72 = a clone - both work)"));

    // ---- gyro bias -------------------------------------------------------
    Serial.println(F("\nStep 1: gyro bias."));
    Serial.println(F("Put the robot down and DO NOT TOUCH IT for 5 seconds."));
    for (int i = 5; i > 0; --i) { Serial.print(i); Serial.print(' '); delay(1000); }
    Serial.println();

    double sx = 0, sy = 0, sz = 0;
    double ax_sum = 0, ay_sum = 0, az_sum = 0;
    int n = 0;
    for (int i = 0; i < 1000; ++i) {
        int16_t ax, ay, az, gx, gy, gz;
        if (mpu_read(&ax, &ay, &az, &gx, &gy, &gz)) {
            sx += gx; sy += gy; sz += gz;
            ax_sum += ax; ay_sum += ay; az_sum += az;
            ++n;
        }
        delay(5);
    }
    if (n < 500) {
        Serial.println(F("FATAL: too many failed I2C reads. Check wiring/pullups."));
        for (;;) delay(1000);
    }

    const double bx = sx / n, by = sy / n, bz = sz / n;
    Serial.println(F("\n--- paste into balance_controller.ino ---"));
    Serial.print(F("  GYRO_BIAS_FALLBACK = "));
    Serial.print(by, 2);
    Serial.println(F("f;   // the pitch-axis gyro (gy)"));
    Serial.println(F("----------------------------------------"));
    Serial.print(F("(all three axes, raw LSB:  gx="));
    Serial.print(bx, 2); Serial.print(F("  gy="));
    Serial.print(by, 2); Serial.print(F("  gz="));
    Serial.print(bz, 2); Serial.println(F(")"));
    Serial.println(F("Note: balance_controller.ino re-measures this at every"));
    Serial.println(F("boot, because gyro bias drifts with temperature. The"));
    Serial.println(F("constant above is only a fallback."));

    // ---- which axis is vertical ------------------------------------------
    Serial.print(F("\nAt rest, accel (g):  x="));
    Serial.print(ax_sum / n / ACCEL_LSB_PER_G, 3);
    Serial.print(F("  y="));
    Serial.print(ay_sum / n / ACCEL_LSB_PER_G, 3);
    Serial.print(F("  z="));
    Serial.println(az_sum / n / ACCEL_LSB_PER_G, 3);
    Serial.println(F("The axis reading about +/-1.0 is the one pointing up or"));
    Serial.println(F("down. balance_controller.ino assumes atan2f(ax, az), i.e."));
    Serial.println(F("Z vertical and X pointing along the direction of travel."));
    Serial.println(F("If a different axis reads 1.0, change that atan2 call to"));
    Serial.println(F("match before going further."));

    Serial.println(F("\nStep 2: THE SIGN CHECK - this is the important one."));
    Serial.println(F("Tilt the robot FORWARD (the way it drives) and watch"));
    Serial.println(F("'angle' below."));
    Serial.println(F("  angle goes POSITIVE  -> IMU_ANGLE_SIGN = +1.0f  (correct)"));
    Serial.println(F("  angle goes NEGATIVE  -> IMU_ANGLE_SIGN = -1.0f"));
    Serial.println(F("Then rock it forward quickly and check 'rate' has the"));
    Serial.println(F("SAME sign as the direction the angle is moving."));
    Serial.println(F("  it does    -> IMU_RATE_SIGN = +1.0f"));
    Serial.println(F("  it doesn't -> IMU_RATE_SIGN = -1.0f"));
    Serial.println(F("\nAlso note the angle when the robot is balanced on its"));
    Serial.println(F("wheels, held upright by hand: that value is"));
    Serial.println(F("ANGLE_TRIM_DEG.\n"));
    delay(2000);

    g_bias_y = (float)by;
}

void loop() {
    int16_t ax, ay, az, gx, gy, gz;
    if (!mpu_read(&ax, &ay, &az, &gx, &gy, &gz)) {
        Serial.println(F("i2c read failed"));
        delay(200);
        return;
    }

    const float angle = atan2f((float)ax, (float)az) * 57.29578f;
    const float rate  = ((float)gy - g_bias_y) / GYRO_LSB_PER_DPS;

    Serial.print(F("angle="));
    Serial.print(angle, 2);
    Serial.print(F(" deg   rate="));
    Serial.print(rate, 1);
    Serial.print(F(" dps   accel g: "));
    Serial.print((float)ax / ACCEL_LSB_PER_G, 2); Serial.print(' ');
    Serial.print((float)ay / ACCEL_LSB_PER_G, 2); Serial.print(' ');
    Serial.print((float)az / ACCEL_LSB_PER_G, 2);
    Serial.println();

    delay(100);   // 10 Hz: fast enough to follow by hand, slow enough to read
}
