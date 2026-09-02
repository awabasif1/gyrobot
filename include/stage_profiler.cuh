#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <iomanip>
#include "cuda_utils.cuh"

// Wall-clock (std::chrono) timing on the host is not trustworthy for GPU
// work: it includes CPU-side launch overhead and doesn't wait for async
// kernels/copies to actually finish unless you insert a sync, and a sync
// biases the very number you're trying to measure. CUDA events are recorded
// directly into the stream, so cudaEventElapsedTime() measures true
// GPU-side elapsed time between two points, independent of CPU scheduling.
class StageProfiler {
public:
    // Call once per named stage per frame/batch, wrapping the region you
    // want timed. Usage:
    //   profiler.start("preprocess");
    //   launch_preprocess_kernel(...);
    //   profiler.stop("preprocess");
    void start(const std::string& stage) {
        auto& ev = events_[stage];
        if (ev.start == nullptr) {
            CUDA_CHECK(cudaEventCreate(&ev.start));
            CUDA_CHECK(cudaEventCreate(&ev.stop));
        }
        CUDA_CHECK(cudaEventRecord(ev.start));
    }

    void stop(const std::string& stage) {
        auto it = events_.find(stage);
        if (it == events_.end()) {
            throw std::runtime_error("StageProfiler::stop called before start for stage: " + stage);
        }
        CUDA_CHECK(cudaEventRecord(it->second.stop));
        CUDA_CHECK(cudaEventSynchronize(it->second.stop)); // only syncs THIS stage's stop event
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, it->second.start, it->second.stop));
        history_[stage].push_back(ms);
    }

    // Prints mean/min/max per stage plus the implied end-to-end total.
    // This is the table that goes straight into the README.
    void report(std::ostream& os = std::cout) const {
        os << std::fixed << std::setprecision(3);
        os << "\n---- Per-stage latency (ms) ----\n";
        os << std::left << std::setw(16) << "stage"
           << std::right << std::setw(10) << "mean"
           << std::setw(10) << "min"
           << std::setw(10) << "max"
           << std::setw(10) << "n" << "\n";

        double total_mean = 0.0;
        for (const auto& [stage, samples] : history_) {
            if (samples.empty()) continue;
            double sum = 0.0, mn = samples[0], mx = samples[0];
            for (double v : samples) { sum += v; mn = std::min(mn, v); mx = std::max(mx, v); }
            double mean = sum / samples.size();
            total_mean += mean;
            os << std::left << std::setw(16) << stage
               << std::right << std::setw(10) << mean
               << std::setw(10) << mn
               << std::setw(10) << mx
               << std::setw(10) << samples.size() << "\n";
        }
        os << std::left << std::setw(16) << "TOTAL (sum of means)"
           << std::right << std::setw(10) << total_mean << "\n";
        os << "---------------------------------\n";
    }

    ~StageProfiler() {
        for (auto& [name, ev] : events_) {
            if (ev.start) cudaEventDestroy(ev.start);
            if (ev.stop) cudaEventDestroy(ev.stop);
        }
    }

private:
    struct EventPair { cudaEvent_t start = nullptr, stop = nullptr; };
    std::unordered_map<std::string, EventPair> events_;
    std::unordered_map<std::string, std::vector<double>> history_;
};
