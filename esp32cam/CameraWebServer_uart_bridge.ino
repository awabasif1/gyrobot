// ===========================================================================
// reflex-cv - ESP32-CAM camera + WiFi <-> UART bridge
// ===========================================================================
//
// This board is deliberately dumb. It has no idea the robot balances. It does
// exactly three things:
//
//   1. serves the camera as an MJPEG stream the laptop can open,
//   2. accepts  GET /cmd?lean=<deg>&turn=<x>  and writes the agreed ASCII line
//      onto its UART, where the Arduino Mega is listening,
//   3. accepts  GET /telemetry  and returns the last line the Mega sent back.
//
// Keeping it dumb is the point. All the perception lives on the laptop (which
// has the GPU) and all the real-time control lives on the Mega (which has
// nothing else to do). This board is a wire with an antenna.
//
// ---------------------------------------------------------------------------
// WHY THERE ARE TWO HTTP SERVERS
// ---------------------------------------------------------------------------
// An MJPEG stream handler never returns - it sits in a loop pushing frames for
// as long as the client is connected. esp_http_server processes requests on a
// single task per instance, so if /stream and /cmd were served by the same
// instance, /cmd would be queued behind the stream and never answered. The
// robot would get the video and no commands at all.
//
// So: control on port 80, stream on port 81. This is the same split the stock
// CameraWebServer example makes, for the same reason.
//
//   stream     http://<ip>:81/stream     <- give this to --stream
//   commands   http://<ip>/cmd?...       <- the laptop finds this itself
//   telemetry  http://<ip>/telemetry
//   status     http://<ip>/status        <- plain text, for debugging by hand
//   index      http://<ip>/              <- a page showing the live stream
//
// ---------------------------------------------------------------------------
// BOARD SETUP (Arduino IDE)
// ---------------------------------------------------------------------------
//   Board:          "AI Thinker ESP32-CAM"
//   Partition:      "Huge APP (3MB No OTA/1MB SPIFFS)"   <- needed; the
//                   default partition scheme is too small for the camera
//                   libraries plus WiFi and the build fails at link time
//   PSRAM:          Enabled
//   Upload speed:   460800 (drop to 115200 if uploads fail)
//
// ---------------------------------------------------------------------------
// WIRING TO THE MEGA
// ---------------------------------------------------------------------------
//   ESP32 GPIO14 (TX) ----------------------------> Mega RX1 (pin 19)
//   ESP32 GPIO13 (RX) <--[ 1k / 2k DIVIDER ]------- Mega TX1 (pin 18)
//   GND  ------------------------------------------ GND
//
//   The divider on the Mega->ESP32 direction is mandatory: the Mega drives 5V
//   and the ESP32's pins are 3.3V. 1k from Mega TX to ESP32 RX, 2k from ESP32
//   RX to ground. The other direction needs nothing - 3.3V reads as HIGH on a
//   5V AVR.
//
//   Why GPIO13/14: on an AI-Thinker ESP32-CAM almost every pin is taken by the
//   camera. 13 and 14 are SD-card pins, free whenever the SD card is unused,
//   and - unlike GPIO12 and GPIO15 - they are not boot strapping pins, so
//   whatever the Mega happens to be driving at power-on cannot change how the
//   ESP32 boots. If you need the SD card, move these and expect to hunt for
//   two pins that are genuinely free.
//
//   *** POWER: feed this board 5V from the UBEC, not from the Mega's 5V pin.
//   The camera pulls ~250mA in bursts when it initialises and the Mega's
//   regulator browns out, which shows up as "Camera init failed with error
//   0x105" - a power fault that reads like a software fault. ***
// ===========================================================================

#include <Arduino.h>
#include <WiFi.h>
#include <ESPmDNS.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"

// ---------------------------------------------------------------------------
// CONFIGURE THESE
// ---------------------------------------------------------------------------
static const char* WIFI_SSID = "YOUR_SSID";
static const char* WIFI_PASS = "YOUR_PASSWORD";

// A fixed address is worth the two minutes it takes to set up: with DHCP the
// board's IP changes and every command line on the laptop has to change with
// it, usually in the middle of a demo. Set USE_STATIC_IP to 0 to use DHCP.
#define USE_STATIC_IP 1
static const IPAddress STATIC_IP (192, 168, 1,  50);
static const IPAddress GATEWAY   (192, 168, 1,   1);
static const IPAddress SUBNET    (255, 255, 255, 0);
static const IPAddress DNS1      (192, 168, 1,   1);

// Also advertises itself as reflex-cam.local, which survives an address change.
static const char* MDNS_NAME = "reflex-cam";

// UART to the Mega.
static const int  UART_RX_PIN = 13;
static const int  UART_TX_PIN = 14;
static const long UART_BAUD   = 115200;

// Starting resolution. VGA (640x480) letterboxes cleanly into a 640x640 model
// input and keeps JPEGs around 15-25KB, which is about as much as 2.4GHz WiFi
// will carry at 20+ fps in a room with other traffic. /res can change it live.
static const framesize_t START_FRAMESIZE = FRAMESIZE_VGA;
static const int         START_QUALITY   = 12;   // 10 = better, 20 = smaller

// ---------------------------------------------------------------------------
// AI-Thinker ESP32-CAM pin map - do not change for this board
// ---------------------------------------------------------------------------
#define PWDN_GPIO_NUM  32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM   0
#define SIOD_GPIO_NUM  26
#define SIOC_GPIO_NUM  27
#define Y9_GPIO_NUM    35
#define Y8_GPIO_NUM    34
#define Y7_GPIO_NUM    39
#define Y6_GPIO_NUM    36
#define Y5_GPIO_NUM    21
#define Y4_GPIO_NUM    19
#define Y3_GPIO_NUM    18
#define Y2_GPIO_NUM     5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM  23
#define PCLK_GPIO_NUM  22

// ===========================================================================
// Shared state
// ===========================================================================
// Telemetry is written by loop() (which drains the UART) and read by an HTTP
// handler running in a different task, so access is guarded. A spinlock rather
// than a mutex: the critical section is a 32-byte memcpy, far cheaper than a
// context switch.
static portMUX_TYPE g_telem_mux = portMUX_INITIALIZER_UNLOCKED;
static char         g_telem_line[48] = "A:0.00,F:1";
static uint32_t     g_telem_age_ms   = 0;
static uint32_t     g_telem_count    = 0;

static volatile uint32_t g_cmd_count   = 0;
static volatile uint32_t g_cmd_bad     = 0;
static volatile uint32_t g_frames_sent = 0;

static httpd_handle_t g_ctrl_server   = NULL;
static httpd_handle_t g_stream_server = NULL;

// UART line assembly for the Mega's telemetry.
static char    g_rx_buf[48];
static uint8_t g_rx_len = 0;

// ===========================================================================
// MJPEG streaming
// ===========================================================================
#define PART_BOUNDARY "reflexcvframeboundary"
static const char* STREAM_CONTENT_TYPE =
    "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* STREAM_PART =
    "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %lld\r\n\r\n";

static esp_err_t stream_handler(httpd_req_t* req) {
    esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    // Without this, some proxies and players buffer the stream and add latency.
    httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store");
    httpd_resp_set_hdr(req, "X-Framerate", "30");

    char part_buf[96];

    while (true) {
        camera_fb_t* fb = esp_camera_fb_get();
        if (!fb) {
            Serial.println("stream: frame grab failed");
            res = ESP_FAIL;
            break;
        }

        if (fb->format != PIXFORMAT_JPEG) {
            // Should not happen: the sensor is configured for JPEG. Bail
            // rather than silently streaming raw bytes a client cannot decode.
            esp_camera_fb_return(fb);
            res = ESP_FAIL;
            break;
        }

        const int hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART,
                                  (unsigned)fb->len,
                                  (long long)esp_timer_get_time());

        res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
        if (res == ESP_OK) res = httpd_resp_send_chunk(req, part_buf, hlen);
        if (res == ESP_OK) res = httpd_resp_send_chunk(req, (const char*)fb->buf,
                                                       fb->len);
        esp_camera_fb_return(fb);

        if (res != ESP_OK) break;   // client disconnected - normal, not an error
        ++g_frames_sent;

        // No delay here on purpose: the frame rate is set by how fast the
        // sensor and the TCP socket can go. A delay would just add latency.
    }
    return res;
}

// ===========================================================================
// Control endpoints
// ===========================================================================

// GET /cmd?lean=<deg>&turn=<x>
//
// Forwards the agreed ASCII line to the Mega verbatim. The line format is
// fixed by mcu/SERIAL_PROTOCOL.md and must match what balance_core.h's
// CommandParser accepts.
//
// This handler does NOT clamp or validate the numbers beyond parsing them -
// the Mega clamps them itself, because the Mega is the board that would be
// damaged by a bad value and should not trust anything arriving over a
// network.
static esp_err_t cmd_handler(httpd_req_t* req) {
    char query[96];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK) {
        ++g_cmd_bad;
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "expected ?lean=<deg>&turn=<x>");
        return ESP_FAIL;
    }

    char lean_s[24], turn_s[24];
    if (httpd_query_key_value(query, "lean", lean_s, sizeof(lean_s)) != ESP_OK ||
        httpd_query_key_value(query, "turn", turn_s, sizeof(turn_s)) != ESP_OK) {
        ++g_cmd_bad;
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "need both lean and turn");
        return ESP_FAIL;
    }

    const float lean = atof(lean_s);
    const float turn = atof(turn_s);

    // Re-formatted rather than pasted through, so a malformed query string can
    // never inject extra characters or newlines into the Mega's UART stream.
    char line[32];
    snprintf(line, sizeof(line), "L:%.2f,T:%.2f\n", lean, turn);
    Serial1.print(line);

    ++g_cmd_count;

    httpd_resp_set_type(req, "text/plain");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_sendstr(req, "OK");
    return ESP_OK;
}

// GET /telemetry  ->  "A:<angle>,F:<0|1>"
static esp_err_t telemetry_handler(httpd_req_t* req) {
    char copy[48];
    uint32_t age;
    portENTER_CRITICAL(&g_telem_mux);
    strncpy(copy, g_telem_line, sizeof(copy) - 1);
    copy[sizeof(copy) - 1] = '\0';
    age = millis() - g_telem_age_ms;
    portEXIT_CRITICAL(&g_telem_mux);

    char body[80];
    snprintf(body, sizeof(body), "%s,AGE:%lu", copy, (unsigned long)age);

    httpd_resp_set_type(req, "text/plain");
    httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
    httpd_resp_sendstr(req, body);
    return ESP_OK;
}

// GET /status - plain text diagnostics, meant to be read by a human with curl
static esp_err_t status_handler(httpd_req_t* req) {
    char copy[48];
    portENTER_CRITICAL(&g_telem_mux);
    strncpy(copy, g_telem_line, sizeof(copy) - 1);
    copy[sizeof(copy) - 1] = '\0';
    portEXIT_CRITICAL(&g_telem_mux);

    sensor_t* s = esp_camera_sensor_get();

    char body[384];
    snprintf(body, sizeof(body),
             "reflex-cv esp32cam bridge\n"
             "uptime_s     %lu\n"
             "wifi         %s  rssi %d dBm  ip %s\n"
             "framesize    %d   quality %d\n"
             "frames_sent  %lu\n"
             "cmds_ok      %lu\n"
             "cmds_bad     %lu\n"
             "telem_lines  %lu\n"
             "telem_last   %s\n"
             "heap_free    %lu\n"
             "psram_free   %lu\n",
             (unsigned long)(millis() / 1000),
             WiFi.status() == WL_CONNECTED ? "connected" : "DOWN",
             (int)WiFi.RSSI(), WiFi.localIP().toString().c_str(),
             s ? (int)s->status.framesize : -1, s ? (int)s->status.quality : -1,
             (unsigned long)g_frames_sent, (unsigned long)g_cmd_count,
             (unsigned long)g_cmd_bad, (unsigned long)g_telem_count, copy,
             (unsigned long)ESP.getFreeHeap(),
             (unsigned long)ESP.getFreePsram());

    httpd_resp_set_type(req, "text/plain");
    httpd_resp_sendstr(req, body);
    return ESP_OK;
}

// GET /res?size=qvga|vga|svga|xga
//
// Lets the resolution be changed without reflashing, which is how you fill in
// the latency table for more than one frame size. NOTE: the laptop pipeline
// sizes its GPU buffers from the first frame it sees, so restart it after
// changing this.
static esp_err_t res_handler(httpd_req_t* req) {
    char query[64], size_s[16];
    if (httpd_req_get_url_query_str(req, query, sizeof(query)) != ESP_OK ||
        httpd_query_key_value(query, "size", size_s, sizeof(size_s)) != ESP_OK) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST,
                            "?size=qvga|vga|svga|xga");
        return ESP_FAIL;
    }

    framesize_t fs;
    if      (!strcmp(size_s, "qvga")) fs = FRAMESIZE_QVGA;  // 320x240
    else if (!strcmp(size_s, "vga"))  fs = FRAMESIZE_VGA;   // 640x480
    else if (!strcmp(size_s, "svga")) fs = FRAMESIZE_SVGA;  // 800x600
    else if (!strcmp(size_s, "xga"))  fs = FRAMESIZE_XGA;   // 1024x768
    else {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "unknown size");
        return ESP_FAIL;
    }

    sensor_t* s = esp_camera_sensor_get();
    if (!s || s->set_framesize(s, fs) != 0) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR,
                            "set_framesize failed");
        return ESP_FAIL;
    }

    httpd_resp_set_type(req, "text/plain");
    httpd_resp_sendstr(req,
                       "OK - restart the laptop pipeline to pick up the new size");
    return ESP_OK;
}

// GET / - a page showing the stream, so the link can be checked from a phone
// before any C++ is involved.
static esp_err_t index_handler(httpd_req_t* req) {
    static const char page[] =
        "<!doctype html><html><head><meta name=viewport "
        "content='width=device-width,initial-scale=1'>"
        "<title>reflex-cv camera</title>"
        "<style>body{background:#111;color:#eee;font-family:system-ui,sans-serif;"
        "margin:0;padding:12px;text-align:center}img{max-width:100%;height:auto;"
        "border:1px solid #333;border-radius:6px}a{color:#6cf}"
        "code{background:#222;padding:2px 5px;border-radius:3px}</style>"
        "</head><body><h3>reflex-cv camera bridge</h3>"
        "<img src=\"\" id=s>"
        "<p><a href=/status>/status</a> &middot; "
        "<a href=/telemetry>/telemetry</a></p>"
        "<p style='font-size:.85em;color:#888'>commands: "
        "<code>/cmd?lean=0.0&amp;turn=0.0</code></p>"
        "<script>document.getElementById('s').src="
        "window.location.protocol+'//'+window.location.hostname+':81/stream';"
        "</script></body></html>";

    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, page, strlen(page));
}

// ===========================================================================
// Server startup
// ===========================================================================
static void start_servers() {
    httpd_config_t cfg = HTTPD_DEFAULT_CONFIG();
    cfg.server_port = 80;
    cfg.ctrl_port   = 32768;
    // The laptop keeps one connection alive for commands; the browser may open
    // another. 7 is plenty and costs little.
    cfg.max_open_sockets = 7;
    cfg.lru_purge_enable = true;

    httpd_uri_t index_uri  = { "/",          HTTP_GET, index_handler,     NULL };
    httpd_uri_t cmd_uri    = { "/cmd",       HTTP_GET, cmd_handler,       NULL };
    httpd_uri_t telem_uri  = { "/telemetry", HTTP_GET, telemetry_handler, NULL };
    httpd_uri_t status_uri = { "/status",    HTTP_GET, status_handler,    NULL };
    httpd_uri_t res_uri    = { "/res",       HTTP_GET, res_handler,       NULL };

    if (httpd_start(&g_ctrl_server, &cfg) == ESP_OK) {
        httpd_register_uri_handler(g_ctrl_server, &index_uri);
        httpd_register_uri_handler(g_ctrl_server, &cmd_uri);
        httpd_register_uri_handler(g_ctrl_server, &telem_uri);
        httpd_register_uri_handler(g_ctrl_server, &status_uri);
        httpd_register_uri_handler(g_ctrl_server, &res_uri);
        Serial.println("control server up on :80");
    } else {
        Serial.println("FAILED to start control server on :80");
    }

    // Separate instance, separate task - see the note at the top of the file.
    httpd_config_t scfg = HTTPD_DEFAULT_CONFIG();
    scfg.server_port = 81;
    scfg.ctrl_port   = 32769;
    scfg.max_open_sockets = 3;
    scfg.lru_purge_enable = true;

    httpd_uri_t stream_uri = { "/stream", HTTP_GET, stream_handler, NULL };
    if (httpd_start(&g_stream_server, &scfg) == ESP_OK) {
        httpd_register_uri_handler(g_stream_server, &stream_uri);
        Serial.println("stream server up on :81/stream");
    } else {
        Serial.println("FAILED to start stream server on :81");
    }
}

static bool start_camera() {
    camera_config_t c;
    c.ledc_channel = LEDC_CHANNEL_0;
    c.ledc_timer   = LEDC_TIMER_0;
    c.pin_d0 = Y2_GPIO_NUM;   c.pin_d1 = Y3_GPIO_NUM;
    c.pin_d2 = Y4_GPIO_NUM;   c.pin_d3 = Y5_GPIO_NUM;
    c.pin_d4 = Y6_GPIO_NUM;   c.pin_d5 = Y7_GPIO_NUM;
    c.pin_d6 = Y8_GPIO_NUM;   c.pin_d7 = Y9_GPIO_NUM;
    c.pin_xclk  = XCLK_GPIO_NUM;
    c.pin_pclk  = PCLK_GPIO_NUM;
    c.pin_vsync = VSYNC_GPIO_NUM;
    c.pin_href  = HREF_GPIO_NUM;
    c.pin_sccb_sda = SIOD_GPIO_NUM;
    c.pin_sccb_scl = SIOC_GPIO_NUM;
    c.pin_pwdn  = PWDN_GPIO_NUM;
    c.pin_reset = RESET_GPIO_NUM;
    c.xclk_freq_hz = 20000000;
    c.pixel_format = PIXFORMAT_JPEG;   // the sensor's own JPEG encoder; doing
                                       // this on the ESP32's CPU instead would
                                       // cap the frame rate at single digits
    c.frame_size   = START_FRAMESIZE;
    c.jpeg_quality = START_QUALITY;
    c.fb_count     = 2;
    c.fb_location  = CAMERA_FB_IN_PSRAM;

    // CAMERA_GRAB_LATEST, not CAMERA_GRAB_WHEN_EMPTY: with two frame buffers
    // and a consumer that is sometimes slow, WHEN_EMPTY hands out whichever
    // frame has been sitting around longest. The whole system is built to
    // minimise the age of the frame being acted on, so always take the newest
    // and let the older one be overwritten.
    c.grab_mode = CAMERA_GRAB_LATEST;

    if (!psramFound()) {
        // Without PSRAM there is nowhere to put two VGA JPEG buffers, so fall
        // back to something that fits in internal RAM rather than failing to
        // initialise with an error that looks like a hardware fault.
        Serial.println("WARNING: no PSRAM detected - falling back to QVGA.");
        Serial.println("  On an AI-Thinker board this usually means PSRAM is");
        Serial.println("  disabled in Tools > PSRAM, not that it is missing.");
        c.frame_size   = FRAMESIZE_QVGA;
        c.fb_location  = CAMERA_FB_IN_DRAM;
        c.fb_count     = 1;
        c.jpeg_quality = 14;
    }

    const esp_err_t err = esp_camera_init(&c);
    if (err != ESP_OK) {
        Serial.printf("Camera init failed with error 0x%x\n", err);
        Serial.println("  0x105 (ESP_ERR_NOT_FOUND) is almost always POWER:");
        Serial.println("  feed the board 5V from the UBEC, not from the Mega.");
        Serial.println("  Also check the ribbon connector is fully seated.");
        return false;
    }

    sensor_t* s = esp_camera_sensor_get();
    if (s) {
        // The OV2640 ships mirrored and slightly washed out for this use.
        s->set_vflip(s, 0);
        s->set_hmirror(s, 0);
        s->set_brightness(s, 1);
        s->set_saturation(s, 0);
    }
    return true;
}

static void connect_wifi() {
    WiFi.mode(WIFI_STA);

#if USE_STATIC_IP
    if (!WiFi.config(STATIC_IP, GATEWAY, SUBNET, DNS1)) {
        Serial.println("WiFi.config failed - falling back to DHCP");
    }
#endif

    WiFi.begin(WIFI_SSID, WIFI_PASS);

    // *** The single most important line in this file for latency. ***
    // WiFi modem sleep parks the radio between packets to save power. It saves
    // milliamps the robot does not need and costs 100-200ms of random latency
    // on the command path, which is catastrophic for a control loop. It is ON
    // by default.
    WiFi.setSleep(false);

    Serial.print("connecting to ");
    Serial.print(WIFI_SSID);
    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) {
        delay(400);
        Serial.print('.');
    }
    Serial.println();

    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi FAILED - check SSID/password, and that the network");
        Serial.println("is 2.4GHz (the ESP32 cannot see a 5GHz-only SSID).");
        return;
    }

    Serial.print("ip      ");
    Serial.println(WiFi.localIP());
    Serial.print("rssi    ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm   (weaker than -70 will stutter)");

    if (MDNS.begin(MDNS_NAME)) {
        MDNS.addService("http", "tcp", 80);
        Serial.printf("mdns    http://%s.local/\n", MDNS_NAME);
    }

    Serial.println();
    Serial.println("give these to the laptop:");
    Serial.printf("  --stream http://%s:81/stream\n",
                  WiFi.localIP().toString().c_str());
    Serial.printf("  (commands go to http://%s/cmd automatically)\n",
                  WiFi.localIP().toString().c_str());
}

// ===========================================================================
// setup / loop
// ===========================================================================
void setup() {
    Serial.begin(115200);
    Serial.setDebugOutput(false);
    delay(200);
    Serial.println("\n\n=== reflex-cv esp32cam bridge ===");

    // UART to the Mega. Started before the camera so that if the camera fails
    // the Mega still gets its zero command and is not left holding a stale one.
    Serial1.begin(UART_BAUD, SERIAL_8N1, UART_RX_PIN, UART_TX_PIN);
    Serial1.print("L:0.00,T:0.00\n");
    Serial.printf("uart    rx=GPIO%d tx=GPIO%d @ %ld baud\n",
                  UART_RX_PIN, UART_TX_PIN, UART_BAUD);

    if (!start_camera()) {
        // Keep going anyway: the command bridge is still useful without the
        // camera, and a board that reboots in a loop is harder to diagnose
        // than one that stays up and says what is wrong on /status.
        Serial.println("continuing without the camera - /cmd still works");
    }

    connect_wifi();
    start_servers();

    Serial.println("ready");
}

void loop() {
    // ---- drain the Mega's telemetry --------------------------------------
    while (Serial1.available()) {
        const char ch = (char)Serial1.read();
        if (ch == '\r') continue;

        if (ch == '\n') {
            if (g_rx_len > 0) {
                g_rx_buf[g_rx_len] = '\0';
                portENTER_CRITICAL(&g_telem_mux);
                strncpy(g_telem_line, g_rx_buf, sizeof(g_telem_line) - 1);
                g_telem_line[sizeof(g_telem_line) - 1] = '\0';
                g_telem_age_ms = millis();
                ++g_telem_count;
                portEXIT_CRITICAL(&g_telem_mux);
            }
            g_rx_len = 0;
        } else if (g_rx_len < sizeof(g_rx_buf) - 1) {
            g_rx_buf[g_rx_len++] = ch;
        } else {
            g_rx_len = 0;   // overlong line: drop it rather than truncating
        }
    }

    // ---- WiFi watchdog ---------------------------------------------------
    // The robot moves around and will drop off the network. Reconnecting here
    // means it recovers on its own instead of needing a power cycle mid-demo.
    static uint32_t s_last_check = 0;
    if (millis() - s_last_check > 5000) {
        s_last_check = millis();
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("wifi dropped - reconnecting");
            WiFi.disconnect();
            WiFi.begin(WIFI_SSID, WIFI_PASS);
            WiFi.setSleep(false);
        }
    }

    // Yield to the IDF tasks (WiFi, both HTTP servers). Without a delay here
    // this loop starves lower-priority system tasks on the same core.
    delay(2);
}
