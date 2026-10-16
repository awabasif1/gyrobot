#pragma once
// ---------------------------------------------------------------------------
// Wall-clock companion to stage_profiler.cuh.
//
// CUDA events are the right tool for GPU work and the wrong tool for
// everything else: they can only time regions that are actually enqueued on a
// stream. Three stages of this pipeline are not GPU work at all -
//
//   * waiting for a frame to arrive over WiFi,
//   * JPEG decode on the CPU inside OpenCV,
//   * box decoding / NMS / command shaping on the CPU,
//
// - and the project's latency table has a row for each. Timing those with CUDA
// events would silently report nonsense, so they are measured here with
// steady_clock instead, and main.cpp prints the two tables together.
//
// steady_clock, not system_clock: the latter can jump backwards when NTP
// corrects the laptop's clock mid-run, which would produce negative durations.
// ---------------------------------------------------------------------------

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

class HostProfiler {
public:
    using Clock = std::chrono::steady_clock;

    void start(const std::string& stage) { active_[stage] = Clock::now(); }

    void stop(const std::string& stage) {
        auto it = active_.find(stage);
        if (it == active_.end()) return;   // stop without start: ignore
        const double ms =
            std::chrono::duration<double, std::milli>(Clock::now() - it->second)
                .count();
        history_[stage].push_back(ms);
        active_.erase(it);
    }

    // Record a duration measured elsewhere (used for the end-to-end row).
    void add_sample(const std::string& stage, double ms) {
        history_[stage].push_back(ms);
    }

    struct Stats {
        double mean = 0, min = 0, max = 0, p95 = 0;
        size_t n = 0;
    };

    Stats stats(const std::string& stage) const {
        Stats s;
        auto it = history_.find(stage);
        if (it == history_.end() || it->second.empty()) return s;

        std::vector<double> v = it->second;
        s.n = v.size();
        double sum = 0;
        for (double x : v) sum += x;
        s.mean = sum / static_cast<double>(v.size());
        std::sort(v.begin(), v.end());
        s.min = v.front();
        s.max = v.back();
        // p95 matters more than max for a control loop: max is usually one
        // outlier from startup, while p95 is what the robot actually lives
        // with frame after frame.
        s.p95 = v[static_cast<size_t>(0.95 * (v.size() - 1))];
        return s;
    }

    std::vector<std::string> stages() const {
        std::vector<std::string> out;
        out.reserve(history_.size());
        for (const auto& kv : history_) out.push_back(kv.first);
        return out;
    }

    void report(std::ostream& os = std::cout,
                const std::string& title = "Per-stage latency, host timed (ms)") const {
        os << std::fixed << std::setprecision(3);
        os << "\n---- " << title << " ----\n";
        os << std::left << std::setw(22) << "stage" << std::right
           << std::setw(10) << "mean" << std::setw(10) << "min"
           << std::setw(10) << "p95" << std::setw(10) << "max"
           << std::setw(8) << "n" << "\n";
        for (const auto& kv : history_) {
            if (kv.second.empty()) continue;
            const Stats s = stats(kv.first);
            os << std::left << std::setw(22) << kv.first << std::right
               << std::setw(10) << s.mean << std::setw(10) << s.min
               << std::setw(10) << s.p95 << std::setw(10) << s.max
               << std::setw(8) << s.n << "\n";
        }
        os << std::string(70, '-') << "\n";
    }

    // Markdown row dump, so the README's latency table can be filled in by
    // copy-paste instead of by retyping numbers off a terminal.
    void report_markdown(std::ostream& os) const {
        os << std::fixed << std::setprecision(2);
        os << "\n| Stage | Mean (ms) | Min (ms) | p95 (ms) | Max (ms) | n |\n";
        os << "|---|---|---|---|---|---|\n";
        for (const auto& kv : history_) {
            if (kv.second.empty()) continue;
            const Stats s = stats(kv.first);
            os << "| " << kv.first << " | " << s.mean << " | " << s.min
               << " | " << s.p95 << " | " << s.max << " | " << s.n << " |\n";
        }
    }

    void write_csv(std::ostream& os) const {
        os << "stage,sample_index,ms\n";
        for (const auto& kv : history_) {
            for (size_t i = 0; i < kv.second.size(); ++i) {
                os << kv.first << "," << i << "," << kv.second[i] << "\n";
            }
        }
    }

private:
    std::map<std::string, Clock::time_point>  active_;
    std::map<std::string, std::vector<double>> history_;
};
