#pragma once

#include "cacheflow/common.h"
#include "cacheflow/profiler/metrics.h"
#include "cacheflow/profiler/trace.h"
#include <memory>
#include <string>
#include <unordered_map>

namespace cacheflow {

class BlockManager;
class PrefixCache;
class Scheduler;

// ── System-level profiler ───────────────────────────────────────────────────

struct ProfilerConfig {
    bool   enable_tracing  = true;
    bool   enable_metrics  = true;
    std::string trace_path = "cacheflow_trace.json";
    double histogram_min   = 0.0;
    double histogram_max   = 500.0;
    size_t histogram_bins  = 500;
};

class Profiler {
public:
    using Config = ProfilerConfig;

    explicit Profiler(const Config& config = Config{});
    ~Profiler();

    Profiler(const Profiler&) = delete;
    Profiler& operator=(const Profiler&) = delete;

    void set_block_manager(BlockManager* bm)   { block_manager_ = bm; }
    void set_prefix_cache(PrefixCache* pc)      { prefix_cache_ = pc; }
    void set_scheduler(Scheduler* sched)        { scheduler_ = sched; }

    void begin_step(size_t num_tokens);
    void end_step(double step_time_ms);
    void mark(const std::string& label);

    void record_request_start(SeqId id);
    void record_first_token(SeqId id);
    void record_request_end(SeqId id, size_t output_tokens);

    void record_prefill(SeqId id, size_t prompt_len, double time_ms);
    void record_decode(SeqId id, double time_ms);

    MetricsSnapshot snapshot() const;

    RunningStats& step_latency()      { return step_latency_; }
    RunningStats& ttft_stats()        { return ttft_stats_; }
    RunningStats& tpot_stats()        { return tpot_stats_; }
    Histogram&    latency_histogram() { return latency_hist_; }
    ThroughputMeter& throughput()     { return throughput_; }

    TraceWriter* trace_writer() {
        return trace_writer_ ? trace_writer_.get() : nullptr;
    }

    void reset();
    void dump_summary(std::ostream& os) const;
    void dump_csv(const std::string& path) const;

private:
    Config config_;

    RunningStats    step_latency_;
    RunningStats    ttft_stats_;
    RunningStats    tpot_stats_;
    Histogram       latency_hist_;
    ThroughputMeter throughput_;

    struct RequestRecord {
        TimePoint start;
        TimePoint first_token;
        bool      got_first_token = false;
    };
    std::unordered_map<SeqId, RequestRecord> active_requests_;

    std::unique_ptr<TraceWriter> trace_writer_;
    TimePoint step_start_;

    BlockManager* block_manager_ = nullptr;
    PrefixCache*  prefix_cache_  = nullptr;
    Scheduler*    scheduler_     = nullptr;
    uint64_t      total_requests_ = 0;
};

}  // namespace cacheflow
