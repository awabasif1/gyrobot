#pragma once
// ---------------------------------------------------------------------------
// The laptop -> ESP32-CAM command link.
//
// THE ONE IDEA IN THIS FILE: the vision loop must never block on WiFi.
//
// A naive implementation calls an HTTP GET inline, once per frame. When the
// link hiccups - and a 2.4GHz link in a room full of laptops hiccups
// constantly - that call stalls for anywhere up to its timeout, the vision
// loop stops dead, frames pile up in the capture buffer, and the latency the
// profiler reports becomes meaningless. Worse, the commands that eventually
// get through describe where the target was a second ago.
//
// So: a worker thread owns the socket. post_command() only drops the newest
// command into a one-slot mailbox and returns immediately. If the network is
// slower than the vision loop, intermediate commands are overwritten and never
// sent, which is exactly right - a stale steering command has negative value.
// Dropped commands are counted so the behaviour is visible rather than
// invisible.
//
// Degradation is by design: the Arduino Mega keeps balancing on its last
// received command regardless of what happens here, so a dead link makes the
// robot coast and stop, not fall over.
// ---------------------------------------------------------------------------

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>

namespace reflex {

// Last telemetry line the Mega sent back up through the bridge.
struct Telemetry {
    float  angle_deg = 0.f;
    bool   fallen    = false;
    bool   valid     = false;  // false until a telemetry read has succeeded
    double age_s     = 0.0;    // how stale this reading is
};

struct LinkStats {
    unsigned long sent      = 0;  // commands that reached the ESP32 with a 200
    unsigned long dropped   = 0;  // superseded before they could be sent
    unsigned long failed    = 0;  // send attempts that errored or timed out
    unsigned long reconnects = 0;
    double        last_rtt_ms = 0.0;
    bool          connected   = false;
};

// Pieces of an http://host:port/path URL.
struct HttpUrl {
    std::string host;
    int         port = 80;
    std::string path = "/";
};

// Parses http://host[:port][/path]. Returns false on anything it does not
// understand (https, missing host, non-numeric port).
bool parse_http_url(const std::string& url, HttpUrl* out);

class Esp32Link {
public:
    // host/port of the ESP32-CAM's HTTP server. `send_hz` caps how often
    // commands actually go out, independent of how fast frames arrive - the
    // balance loop runs 10-25x faster than any vision update, so flooding the
    // bridge buys nothing and costs airtime.
    Esp32Link(std::string host, int port, double send_hz = 20.0,
              double telemetry_hz = 2.0, int timeout_ms = 150);
    ~Esp32Link();

    Esp32Link(const Esp32Link&)            = delete;
    Esp32Link& operator=(const Esp32Link&) = delete;

    void start();
    void stop();   // joins the worker; does NOT send anything

    // Non-blocking. Overwrites any command not yet sent.
    void post_command(float lean_deg, float turn);

    // Blocking, best-effort, with its own short timeout. Called on shutdown so
    // the robot is not left holding a turn command after Ctrl-C. This is a
    // safety behaviour, not a nicety: the Mega holds its last command
    // indefinitely.
    bool send_stop_blocking();

    LinkStats  stats() const;
    Telemetry  telemetry() const;

    // Enabled unless --no-send was passed. When disabled, commands are shaped,
    // logged and counted but never transmitted - the way to bench-test the
    // whole vision pipeline with no robot powered on.
    void set_enabled(bool e) { enabled_.store(e); }
    bool enabled() const { return enabled_.load(); }

private:
    void worker();
    bool ensure_connected();
    void disconnect();
    // Sends one request on the persistent connection. Returns false (and drops
    // the connection) on any error, so the next attempt reconnects.
    bool http_get(const std::string& path, std::string* body);
    bool send_command_now(float lean_deg, float turn);
    void poll_telemetry();

    std::string host_;
    int         port_;
    double      send_interval_s_;
    double      telemetry_interval_s_;
    int         timeout_ms_;

    std::thread             thread_;
    mutable std::mutex      mu_;
    std::condition_variable cv_;
    std::atomic<bool>       running_{false};
    std::atomic<bool>       enabled_{true};

    // one-slot mailbox
    bool  pending_      = false;
    float pending_lean_ = 0.f;
    float pending_turn_ = 0.f;

    LinkStats stats_{};
    Telemetry telem_{};

    int socket_fd_ = -1;   // SOCKET on Windows, but int is wide enough
};

} // namespace reflex
