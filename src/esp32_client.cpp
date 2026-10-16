#include "esp32_client.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>   // atoi/atof/atol - included explicitly rather than relying
                     // on them arriving transitively, which differs on MSVC
#include <cstring>
#include <iostream>
#include <sstream>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#define RCV_CLOSESOCK closesocket
#define RCV_SOCKERR   WSAGetLastError()
using rcv_sock_len = int;
#else
#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/select.h>
#include <sys/socket.h>
#include <unistd.h>
#define RCV_CLOSESOCK ::close
#define RCV_SOCKERR   errno
using rcv_sock_len = socklen_t;
#endif

namespace reflex {

using Clock = std::chrono::steady_clock;

namespace {

// Winsock needs explicit global init. Done once via a static local so there is
// no ordering dependency on main().
void ensure_sockets_ready() {
#ifdef _WIN32
    static bool done = [] {
        WSADATA wsa;
        return WSAStartup(MAKEWORD(2, 2), &wsa) == 0;
    }();
    (void)done;
#endif
}

void set_blocking(int fd, bool blocking) {
#ifdef _WIN32
    u_long mode = blocking ? 0 : 1;
    ioctlsocket(static_cast<SOCKET>(fd), FIONBIO, &mode);
#else
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return;
    fcntl(fd, F_SETFL, blocking ? (flags & ~O_NONBLOCK) : (flags | O_NONBLOCK));
#endif
}

void set_rw_timeout(int fd, int timeout_ms) {
#ifdef _WIN32
    DWORD tv = static_cast<DWORD>(timeout_ms);
    setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char*>(&tv), sizeof(tv));
    setsockopt(static_cast<SOCKET>(fd), SOL_SOCKET, SO_SNDTIMEO,
               reinterpret_cast<const char*>(&tv), sizeof(tv));
#else
    timeval tv{};
    tv.tv_sec  = timeout_ms / 1000;
    tv.tv_usec = (timeout_ms % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

// Connect with a hard upper bound on how long it can take. The default TCP
// connect timeout is tens of seconds, which on a robot means the command
// thread disappears for half a minute when the ESP32 reboots.
int connect_to(const std::string& host, int port, int timeout_ms) {
    ensure_sockets_ready();

    addrinfo hints{};
    hints.ai_family   = AF_INET;      // the ESP32-CAM is IPv4-only
    hints.ai_socktype = SOCK_STREAM;

    const std::string port_s = std::to_string(port);
    addrinfo* res = nullptr;
    if (getaddrinfo(host.c_str(), port_s.c_str(), &hints, &res) != 0 || !res) {
        return -1;
    }

    int fd = static_cast<int>(
        ::socket(res->ai_family, res->ai_socktype, res->ai_protocol));
    if (fd < 0) {
        freeaddrinfo(res);
        return -1;
    }

    set_blocking(fd, false);
    const int rc = ::connect(fd, res->ai_addr,
                             static_cast<rcv_sock_len>(res->ai_addrlen));
    freeaddrinfo(res);

    if (rc != 0) {
        // Non-blocking connect returns immediately; wait for writability.
        fd_set wfds;
        FD_ZERO(&wfds);
        FD_SET(static_cast<unsigned>(fd), &wfds);
        timeval tv{};
        tv.tv_sec  = timeout_ms / 1000;
        tv.tv_usec = (timeout_ms % 1000) * 1000;

        if (::select(fd + 1, nullptr, &wfds, nullptr, &tv) <= 0) {
            RCV_CLOSESOCK(fd);
            return -1;
        }
        int err = 0;
        rcv_sock_len len = sizeof(err);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR,
                         reinterpret_cast<char*>(&err), &len) != 0 ||
            err != 0) {
            RCV_CLOSESOCK(fd);
            return -1;
        }
    }

    set_blocking(fd, true);
    set_rw_timeout(fd, timeout_ms);

    // Nagle batches small writes, which is the opposite of what a stream of
    // tiny steering commands wants: it would add up to 40ms of delay waiting
    // for more data that never comes.
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY,
               reinterpret_cast<const char*>(&one), sizeof(one));

    return fd;
}

bool send_all(int fd, const char* data, size_t len) {
    size_t off = 0;
    while (off < len) {
#ifdef _WIN32
        const int n = ::send(static_cast<SOCKET>(fd), data + off,
                             static_cast<int>(len - off), 0);
#else
        const ssize_t n = ::send(fd, data + off, len - off, MSG_NOSIGNAL);
#endif
        if (n <= 0) return false;
        off += static_cast<size_t>(n);
    }
    return true;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

// Issues one HTTP/1.1 request on an already-connected socket and reads the
// response. Keep-alive is used (the connection is reused across calls),
// because one TCP handshake per frame would both add latency and churn through
// the ESP32's small socket pool.
bool request_on(int fd, const std::string& host, const std::string& path,
                int* status_out, std::string* body_out) {
    std::ostringstream req;
    req << "GET " << path << " HTTP/1.1\r\n"
        << "Host: " << host << "\r\n"
        << "Connection: keep-alive\r\n"
        << "User-Agent: reflex-cv\r\n\r\n";
    const std::string s = req.str();
    if (!send_all(fd, s.data(), s.size())) return false;

    std::string buf;
    char chunk[1024];
    size_t header_end = std::string::npos;

    // --- headers ---
    while (true) {
#ifdef _WIN32
        const int n = ::recv(static_cast<SOCKET>(fd), chunk, sizeof(chunk), 0);
#else
        const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
#endif
        if (n <= 0) return false;
        buf.append(chunk, static_cast<size_t>(n));
        header_end = buf.find("\r\n\r\n");
        if (header_end != std::string::npos) break;
        if (buf.size() > 16384) return false;   // runaway / not HTTP
    }

    // --- status line ---
    int status = 0;
    {
        const size_t sp = buf.find(' ');
        if (sp == std::string::npos) return false;
        status = std::atoi(buf.c_str() + sp + 1);
    }
    if (status_out) *status_out = status;

    // --- content-length, if any ---
    const std::string headers = lower(buf.substr(0, header_end));
    long content_length = -1;
    const size_t cl = headers.find("content-length:");
    if (cl != std::string::npos) {
        content_length = std::atol(headers.c_str() + cl + 15);
    }

    std::string body = buf.substr(header_end + 4);
    if (content_length >= 0) {
        while (body.size() < static_cast<size_t>(content_length)) {
#ifdef _WIN32
            const int n = ::recv(static_cast<SOCKET>(fd), chunk, sizeof(chunk), 0);
#else
            const ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
#endif
            if (n <= 0) break;
            body.append(chunk, static_cast<size_t>(n));
        }
        body.resize(std::min(body.size(), static_cast<size_t>(content_length)));
    }
    if (body_out) *body_out = body;
    return status >= 200 && status < 300;
}

} // namespace

bool parse_http_url(const std::string& url, HttpUrl* out) {
    if (!out) return false;
    const std::string prefix = "http://";
    if (url.compare(0, prefix.size(), prefix) != 0) return false;

    const std::string rest = url.substr(prefix.size());
    const size_t slash = rest.find('/');
    const std::string authority = rest.substr(0, slash);
    out->path = (slash == std::string::npos) ? "/" : rest.substr(slash);
    if (authority.empty()) return false;

    const size_t colon = authority.find(':');
    if (colon == std::string::npos) {
        out->host = authority;
        out->port = 80;
    } else {
        out->host = authority.substr(0, colon);
        const std::string p = authority.substr(colon + 1);
        if (p.empty() || !std::all_of(p.begin(), p.end(), [](unsigned char c) {
                return std::isdigit(c) != 0;
            })) {
            return false;
        }
        out->port = std::atoi(p.c_str());
    }
    return !out->host.empty() && out->port > 0 && out->port < 65536;
}

Esp32Link::Esp32Link(std::string host, int port, double send_hz,
                     double telemetry_hz, int timeout_ms)
    : host_(std::move(host)),
      port_(port),
      send_interval_s_(send_hz > 0.0 ? 1.0 / send_hz : 0.05),
      telemetry_interval_s_(telemetry_hz > 0.0 ? 1.0 / telemetry_hz : 0.5),
      timeout_ms_(timeout_ms) {}

Esp32Link::~Esp32Link() {
    stop();
    disconnect();
}

void Esp32Link::start() {
    if (running_.exchange(true)) return;
    thread_ = std::thread(&Esp32Link::worker, this);
}

void Esp32Link::stop() {
    if (!running_.exchange(false)) return;
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
}

void Esp32Link::post_command(float lean_deg, float turn) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (pending_) ++stats_.dropped;   // superseded before it was sent
        pending_lean_ = lean_deg;
        pending_turn_ = turn;
        pending_      = true;
    }
    cv_.notify_one();
}

LinkStats Esp32Link::stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    return stats_;
}

Telemetry Esp32Link::telemetry() const {
    std::lock_guard<std::mutex> lk(mu_);
    return telem_;
}

bool Esp32Link::ensure_connected() {
    if (socket_fd_ >= 0) return true;
    socket_fd_ = connect_to(host_, port_, timeout_ms_);
    std::lock_guard<std::mutex> lk(mu_);
    if (socket_fd_ >= 0) {
        ++stats_.reconnects;
        stats_.connected = true;
        return true;
    }
    stats_.connected = false;
    return false;
}

void Esp32Link::disconnect() {
    if (socket_fd_ >= 0) {
        RCV_CLOSESOCK(socket_fd_);
        socket_fd_ = -1;
    }
    std::lock_guard<std::mutex> lk(mu_);
    stats_.connected = false;
}

bool Esp32Link::http_get(const std::string& path, std::string* body) {
    if (!ensure_connected()) return false;
    int status = 0;
    if (!request_on(socket_fd_, host_, path, &status, body)) {
        // Any transport error invalidates the keep-alive connection; drop it so
        // the next attempt starts clean rather than reusing a half-closed one.
        disconnect();
        return false;
    }
    return true;
}

bool Esp32Link::send_command_now(float lean_deg, float turn) {
    // NOTE: "%.2f" honours the C locale's decimal separator. main() must not
    // call setlocale(LC_NUMERIC, "") - a comma here would make the ESP32's
    // atof() read 1,25 as 1 and the robot would quietly under-steer.
    char path[96];
    std::snprintf(path, sizeof(path), "/cmd?lean=%.2f&turn=%.2f",
                  static_cast<double>(lean_deg), static_cast<double>(turn));

    const auto t0 = Clock::now();
    const bool ok = http_get(path, nullptr);
    const double rtt =
        std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    std::lock_guard<std::mutex> lk(mu_);
    stats_.last_rtt_ms = rtt;
    if (ok) ++stats_.sent; else ++stats_.failed;
    return ok;
}

void Esp32Link::poll_telemetry() {
    std::string body;
    if (!http_get("/telemetry", &body)) return;

    // Expected: "A:<angle>,F:<0|1>"
    float angle = 0.f;
    int   fall  = 0;
    const size_t a = body.find("A:");
    const size_t f = body.find("F:");
    if (a == std::string::npos || f == std::string::npos) return;
    angle = static_cast<float>(std::atof(body.c_str() + a + 2));
    fall  = std::atoi(body.c_str() + f + 2);

    std::lock_guard<std::mutex> lk(mu_);
    telem_.angle_deg = angle;
    telem_.fallen    = (fall != 0);
    telem_.valid     = true;
    telem_.age_s     = 0.0;
}

bool Esp32Link::send_stop_blocking() {
    // Uses its own short-lived connection so this is safe whether or not the
    // worker thread is still alive and holding socket_fd_.
    const int fd = connect_to(host_, port_, timeout_ms_);
    if (fd < 0) return false;
    int status = 0;
    const bool ok = request_on(fd, host_, "/cmd?lean=0.00&turn=0.00", &status,
                               nullptr);
    RCV_CLOSESOCK(fd);
    return ok;
}

void Esp32Link::worker() {
    auto next_send  = Clock::now();
    auto last_telem = Clock::now() - std::chrono::seconds(10);

    while (running_.load()) {
        float lean = 0.f, turn = 0.f;
        bool  have = false;
        {
            std::unique_lock<std::mutex> lk(mu_);
            // Paces outgoing commands at send_hz regardless of how fast the
            // vision loop calls post_command().
            cv_.wait_until(lk, next_send,
                           [this] { return !running_.load(); });
            if (!running_.load()) break;
            if (pending_) {
                lean    = pending_lean_;
                turn    = pending_turn_;
                pending_ = false;
                have    = true;
            }
        }

        next_send = Clock::now() +
                    std::chrono::duration_cast<Clock::duration>(
                        std::chrono::duration<double>(send_interval_s_));

        if (have && enabled_.load()) send_command_now(lean, turn);

        const double since_telem =
            std::chrono::duration<double>(Clock::now() - last_telem).count();
        if (enabled_.load() && since_telem >= telemetry_interval_s_) {
            poll_telemetry();
            last_telem = Clock::now();
        } else {
            std::lock_guard<std::mutex> lk(mu_);
            telem_.age_s = since_telem;
        }
    }

    disconnect();
}

} // namespace reflex
