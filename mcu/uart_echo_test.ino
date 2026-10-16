// ===========================================================================
// reflex-cv - UART link test (throwaway sketch, no balancing)
// ===========================================================================
//
// Purpose: prove the ESP32-CAM <-> Mega serial link works, on its own, before
// either side's real firmware is in the picture.
//
// This is the sketch the build guide means by "test the ESP32 side against a
// throwaway Mega script that just echoes what it receives". It does three
// things and nothing else:
//
//   * prints every byte arriving on Serial1 to the USB serial monitor,
//   * parses whole lines with the SAME parser the real firmware uses, so a
//     line that is accepted here will be accepted there,
//   * sends fake telemetry back at 20 Hz with a slowly sweeping angle, so the
//     laptop's /telemetry endpoint has something plausible to display.
//
// THE MOTORS ARE NEVER TOUCHED. Run this with the robot lying on its side if
// you like - it cannot move.
//
// How to use it:
//   1. Upload this to the Mega. Open the Serial Monitor at 115200.
//   2. With the ESP32-CAM flashed and on the network, from the laptop run:
//          curl "http://<esp32-ip>/cmd?lean=1.5&turn=-0.25"
//      You should see the line arrive here, and be marked OK.
//   3. Then check the return path:
//          curl "http://<esp32-ip>/telemetry"
//      You should get back the sweeping angle this sketch is sending.
//
// If step 2 shows nothing: wrong UART pins, missing common ground, or the
// level shifter is wired backwards. If step 2 shows garbage characters: baud
// rate mismatch between the two boards.
// ===========================================================================

#include <Arduino.h>

#include "balance_core.h"

using namespace reflex;

static CommandParser g_parser;
static uint32_t      g_last_telem_ms = 0;
static uint32_t      g_bytes         = 0;
static float         g_fake_angle    = 0.0f;

void setup() {
    Serial.begin(115200);
    Serial1.begin(115200);
    Serial.println(F("=== reflex-cv UART echo test (motors inactive) ==="));
    Serial.println(F("listening on Serial1 (pins 18=TX1, 19=RX1) at 115200"));
    Serial.println(F("expecting lines of the form:  L:<lean>,T:<turn>\\n"));
    g_last_telem_ms = millis();
}

void loop() {
    // ---- inbound ---------------------------------------------------------
    while (Serial1.available()) {
        const char c = (char)Serial1.read();
        ++g_bytes;

        // Show the raw byte, so a baud mismatch or a floating line is obvious
        // rather than just looking like "nothing arrived".
        if (c == '\n')      Serial.println(F("[\\n]"));
        else if (c == '\r') Serial.print(F("[\\r]"));
        else if (c >= 32 && c < 127) Serial.print(c);
        else { Serial.print(F("[0x")); Serial.print((uint8_t)c, HEX); Serial.print(F("]")); }

        if (g_parser.feed(c)) {
            Serial.print(F("  -> OK  lean="));
            Serial.print(g_parser.lean_deg(), 2);
            Serial.print(F("  turn="));
            Serial.print(g_parser.turn(), 2);
            Serial.print(F("   (good="));
            Serial.print(g_parser.good_lines());
            Serial.print(F(" bad="));
            Serial.print(g_parser.bad_lines());
            Serial.println(F(")"));
        }
    }

    // ---- outbound telemetry ---------------------------------------------
    if ((uint32_t)(millis() - g_last_telem_ms) >= 50) {
        g_last_telem_ms = millis();

        // A slow sweep between -8 and +8 degrees, so the laptop side can tell
        // live data from a stuck value.
        g_fake_angle = 8.0f * sinf((float)millis() * 0.0005f);

        char line[32];
        char abuf[10];
        dtostrf(g_fake_angle, 1, 2, abuf);
        snprintf(line, sizeof(line), "A:%s,F:0\n", abuf);
        Serial1.print(line);
    }

    // ---- liveness --------------------------------------------------------
    static uint32_t s_last = 0;
    if ((uint32_t)(millis() - s_last) >= 3000) {
        s_last = millis();
        Serial.print(F("[alive; "));
        Serial.print(g_bytes);
        Serial.print(F(" bytes received, "));
        Serial.print(g_parser.good_lines());
        Serial.print(F(" good lines, "));
        Serial.print(g_parser.bad_lines());
        Serial.println(F(" bad]"));
    }
}
