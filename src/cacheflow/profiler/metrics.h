#pragma once

#include "cacheflow/common.h"
#include <atomic>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>

namespace cacheflow {

// ── Running statistics (Welford's online algorithm) ─────────────────────────

class RunningStats {
public:
    void update(double x) {
        ++count_;
        double delta = x - mean_;
        mean_ += delta / static_cast<double>(count_);
        double delta2 = x - mean_;
        m2_ += delta * delta2;
        min_ = std::min(min_, x);
        max_ = std::max(max_, x);
    }

    uint64_t count()    const { return count_; }
    double   mean()     const { return mean_; }
    double   variance() const {
        return count_ > 1 ? m2_ / static_cast<double>(count_ - 1) : 0.0;
    }
    double stddev() const { return std::sqrt(variance()); }
    double min()    const { return min_; }
    double max()    const { return max_; }

    void reset() {
        count_ = 0; mean_ = 0; m2_ = 0;
        min_ = 1e30; max_ = -1e30;
    }

private:
    uint64_t count_ = 0;
    double   mean_  = 0.0;
    double   m2_    = 0.0;
    double   min_   = 1e30;
    double   max_   = -1e30;
};

// ── Histogram for latency distribution ──────────────────────────────────────

class Histogram {
public:
    Histogram(double min_val, double max_val, size_t num_bins)
        : min_val_(min_val)
        , max_val_(max_val)
        , bin_width_((max_val - min_val) / static_cast<double>(num_bins))
        , bins_(num_bins, 0)
    {}

    void record(double val) {
        if (val < min_val_) { ++underflow_; return; }
        if (val >= max_val_) { ++overflow_; return; }
        size_t idx = static_cast<size_t>((val - min_val_) / bin_width_);
        if (idx >= bins_.size()) idx = bins_.size() - 1;
        ++bins_[idx];
        ++total_;
    }

    double percentile(double p) const {
        if (total_ == 0) return 0.0;
        uint64_t target = static_cast<uint64_t>(
            p * static_cast<double>(total_) / 100.0);
        uint64_t cum = 0;
        for (size_t i = 0; i < bins_.size(); ++i) {
            cum += bins_[i];
            if (cum >= target)
                return min_val_ + (static_cast<double>(i) + 0.5) * bin_width_;
        }
        return max_val_;
    }

    void reset() {
        std::fill(bins_.begin(), bins_.end(), 0);
        underflow_ = overflow_ = total_ = 0;
    }

    const std::vector<uint64_t>& bins() const { return bins_; }
    uint64_t total()     const { return total_; }
    uint64_t underflow() const { return underflow_; }
    uint64_t overflow()  const { return overflow_; }
    double   bin_width() const { return bin_width_; }
    double   min_val()   const { return min_val_; }

private:
    double               min_val_;
    double               max_val_;
    double               bin_width_;
    std::vector<uint64_t> bins_;
    uint64_t             underflow_ = 0;
    uint64_t             overflow_  = 0;
    uint64_t             total_     = 0;
};

// ── Throughput meter ────────────────────────────────────────────────────────

class ThroughputMeter {
public:
    void record_tokens(uint64_t count) {
        total_tokens_ += count;
    }

    void record_interval(double ms) {
        total_time_ms_ += ms;
        ++intervals_;
    }

    double tokens_per_sec() const {
        if (total_time_ms_ <= 0.0) return 0.0;
        return static_cast<double>(total_tokens_) /
               (total_time_ms_ / 1000.0);
    }

    double avg_interval_ms() const {
        if (intervals_ == 0) return 0.0;
        return total_time_ms_ / static_cast<double>(intervals_);
    }

    uint64_t total_tokens() const { return total_tokens_; }
    double   total_time_ms() const { return total_time_ms_; }

    void reset() {
        total_tokens_ = 0; total_time_ms_ = 0; intervals_ = 0;
    }

private:
    uint64_t total_tokens_  = 0;
    double   total_time_ms_ = 0.0;
    uint64_t intervals_     = 0;
};

// ── Aggregate metrics snapshot ──────────────────────────────────────────────

struct MetricsSnapshot {
    double   throughput_tok_per_sec   = 0.0;
    double   avg_latency_ms          = 0.0;
    double   p50_latency_ms          = 0.0;
    double   p90_latency_ms          = 0.0;
    double   p95_latency_ms          = 0.0;
    double   p99_latency_ms          = 0.0;
    double   latency_stddev_ms       = 0.0;
    double   avg_ttft_ms             = 0.0;
    double   avg_tpot_ms             = 0.0;
    float    gpu_cache_usage         = 0.0f;
    float    cpu_cache_usage         = 0.0f;
    float    prefix_cache_hit_rate   = 0.0f;
    float    internal_fragmentation  = 0.0f;
    float    external_fragmentation  = 0.0f;
    size_t   num_running_seqs        = 0;
    size_t   num_waiting_seqs        = 0;
    size_t   num_swapped_seqs        = 0;
    uint64_t total_tokens_generated  = 0;
    uint64_t total_requests_served   = 0;
};

}  // namespace cacheflow
