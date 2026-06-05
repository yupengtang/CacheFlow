#include "cacheflow/profiler/profiler.h"
#include "cacheflow/kv_cache/block_manager.h"
#include "cacheflow/kv_cache/prefix_cache.h"
#include "cacheflow/scheduler/scheduler.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

namespace cacheflow {

Profiler::Profiler(const Config& config)
    : config_(config)
    , latency_hist_(config.histogram_min, config.histogram_max,
                    config.histogram_bins)
{
    if (config_.enable_tracing) {
        trace_writer_ = std::make_unique<TraceWriter>(config_.trace_path);
    }
}

Profiler::~Profiler() = default;

// ── Step lifecycle ──────────────────────────────────────────────────────────

void Profiler::begin_step(size_t num_tokens) {
    step_start_ = Clock::now();
    throughput_.record_tokens(num_tokens);

    if (trace_writer_) {
        TraceEvent ev;
        ev.name     = "decode_step";
        ev.category = "engine";
        ev.phase    = TraceEvent::Phase::BEGIN;
        ev.timestamp = step_start_;
        ev.args.push_back({"num_tokens", std::to_string(num_tokens)});
        trace_writer_->add_event(std::move(ev));
    }
}

void Profiler::end_step(double step_time_ms) {
    step_latency_.update(step_time_ms);
    latency_hist_.record(step_time_ms);
    throughput_.record_interval(step_time_ms);

    if (trace_writer_) {
        TraceEvent ev;
        ev.name     = "decode_step";
        ev.category = "engine";
        ev.phase    = TraceEvent::Phase::END;
        ev.timestamp = Clock::now();
        trace_writer_->add_event(std::move(ev));

        if (block_manager_) {
            trace_writer_->add_counter("gpu_cache_usage", "utilization",
                                       block_manager_->gpu_cache_usage());
        }
    }
}

void Profiler::mark(const std::string& label) {
    if (!trace_writer_) return;
    TraceEvent ev;
    ev.name      = label;
    ev.category  = "mark";
    ev.phase     = TraceEvent::Phase::INSTANT;
    ev.timestamp = Clock::now();
    trace_writer_->add_event(std::move(ev));
}

// ── Request lifecycle ───────────────────────────────────────────────────────

void Profiler::record_request_start(SeqId id) {
    RequestRecord rec;
    rec.start = Clock::now();
    active_requests_[id] = rec;
}

void Profiler::record_first_token(SeqId id) {
    auto it = active_requests_.find(id);
    if (it == active_requests_.end()) return;

    it->second.first_token = Clock::now();
    it->second.got_first_token = true;

    double ttft_ms = Duration(
        it->second.first_token - it->second.start).count();
    ttft_stats_.update(ttft_ms);
}

void Profiler::record_request_end(SeqId id, size_t output_tokens) {
    auto it = active_requests_.find(id);
    if (it == active_requests_.end()) return;

    auto now = Clock::now();

    if (it->second.got_first_token && output_tokens > 1) {
        double decode_ms = Duration(
            now - it->second.first_token).count();
        double tpot = decode_ms / static_cast<double>(output_tokens - 1);
        tpot_stats_.update(tpot);
    }

    ++total_requests_;
    active_requests_.erase(it);

    if (trace_writer_) {
        trace_writer_->add_counter("requests_completed", "count",
                                   static_cast<double>(total_requests_));
    }
}

void Profiler::record_prefill(SeqId id, size_t prompt_len,
                              double time_ms) {
    if (trace_writer_) {
        TraceEvent ev;
        ev.name     = "prefill";
        ev.category = "decode";
        ev.phase    = TraceEvent::Phase::COMPLETE;
        ev.timestamp = Clock::now();
        ev.duration = Duration(time_ms);
        ev.args.push_back({"seq_id", std::to_string(id)});
        ev.args.push_back({"prompt_len", std::to_string(prompt_len)});
        trace_writer_->add_event(std::move(ev));
    }
}

void Profiler::record_decode(SeqId id, double time_ms) {
    if (trace_writer_) {
        TraceEvent ev;
        ev.name     = "decode";
        ev.category = "decode";
        ev.phase    = TraceEvent::Phase::COMPLETE;
        ev.timestamp = Clock::now();
        ev.duration = Duration(time_ms);
        ev.args.push_back({"seq_id", std::to_string(id)});
        trace_writer_->add_event(std::move(ev));
    }
}

// ── Snapshot ────────────────────────────────────────────────────────────────

MetricsSnapshot Profiler::snapshot() const {
    MetricsSnapshot snap;
    snap.throughput_tok_per_sec  = throughput_.tokens_per_sec();
    snap.avg_latency_ms          = step_latency_.mean();
    snap.latency_stddev_ms       = step_latency_.stddev();
    snap.p50_latency_ms          = latency_hist_.percentile(50);
    snap.p90_latency_ms          = latency_hist_.percentile(90);
    snap.p95_latency_ms          = latency_hist_.percentile(95);
    snap.p99_latency_ms          = latency_hist_.percentile(99);
    snap.avg_ttft_ms             = ttft_stats_.mean();
    snap.avg_tpot_ms             = tpot_stats_.mean();
    snap.total_tokens_generated  = throughput_.total_tokens();
    snap.total_requests_served   = total_requests_;

    if (block_manager_) {
        snap.gpu_cache_usage = block_manager_->gpu_cache_usage();
        snap.cpu_cache_usage = block_manager_->cpu_cache_usage();
        auto frag = block_manager_->compute_fragmentation(DeviceType::CUDA);
        snap.internal_fragmentation = frag.internal_fragmentation;
        snap.external_fragmentation = frag.external_fragmentation;
    }

    if (prefix_cache_) {
        snap.prefix_cache_hit_rate = prefix_cache_->hit_rate();
    }

    if (scheduler_) {
        snap.num_running_seqs = scheduler_->num_running();
        snap.num_waiting_seqs = scheduler_->num_waiting();
        snap.num_swapped_seqs = scheduler_->num_swapped();
    }

    return snap;
}

void Profiler::reset() {
    step_latency_.reset();
    ttft_stats_.reset();
    tpot_stats_.reset();
    latency_hist_.reset();
    throughput_.reset();
    active_requests_.clear();
    total_requests_ = 0;
}

// ── Reporting ───────────────────────────────────────────────────────────────

void Profiler::dump_summary(std::ostream& os) const {
    auto snap = snapshot();
    os << std::fixed << std::setprecision(2);
    os << "\n╔══════════════════════════════════════════════════╗\n";
    os << "║          CacheFlow Performance Summary           ║\n";
    os << "╠══════════════════════════════════════════════════╣\n";
    os << "║ Throughput:    " << std::setw(10)
       << snap.throughput_tok_per_sec << " tok/s           ║\n";
    os << "║ Avg Latency:   " << std::setw(10)
       << snap.avg_latency_ms << " ms              ║\n";
    os << "║ P50 Latency:   " << std::setw(10)
       << snap.p50_latency_ms << " ms              ║\n";
    os << "║ P90 Latency:   " << std::setw(10)
       << snap.p90_latency_ms << " ms              ║\n";
    os << "║ P95 Latency:   " << std::setw(10)
       << snap.p95_latency_ms << " ms              ║\n";
    os << "║ P99 Latency:   " << std::setw(10)
       << snap.p99_latency_ms << " ms              ║\n";
    os << "║ Latency σ:     " << std::setw(10)
       << snap.latency_stddev_ms << " ms              ║\n";
    os << "║ Avg TTFT:      " << std::setw(10)
       << snap.avg_ttft_ms << " ms              ║\n";
    os << "║ Avg TPOT:      " << std::setw(10)
       << snap.avg_tpot_ms << " ms              ║\n";
    os << "╠══════════════════════════════════════════════════╣\n";
    os << "║ GPU Cache:     " << std::setw(10)
       << snap.gpu_cache_usage * 100 << " %               ║\n";
    os << "║ Int. Frag:     " << std::setw(10)
       << snap.internal_fragmentation * 100 << " %               ║\n";
    os << "║ Ext. Frag:     " << std::setw(10)
       << snap.external_fragmentation * 100 << " %               ║\n";
    os << "║ Prefix Hit:    " << std::setw(10)
       << snap.prefix_cache_hit_rate * 100 << " %               ║\n";
    os << "╠══════════════════════════════════════════════════╣\n";
    os << "║ Running:       " << std::setw(10)
       << snap.num_running_seqs << "                  ║\n";
    os << "║ Waiting:       " << std::setw(10)
       << snap.num_waiting_seqs << "                  ║\n";
    os << "║ Swapped:       " << std::setw(10)
       << snap.num_swapped_seqs << "                  ║\n";
    os << "║ Total Served:  " << std::setw(10)
       << snap.total_requests_served << "                  ║\n";
    os << "║ Total Tokens:  " << std::setw(10)
       << snap.total_tokens_generated << "                  ║\n";
    os << "╚══════════════════════════════════════════════════╝\n";
}

void Profiler::dump_csv(const std::string& path) const {
    std::ofstream f(path);
    if (!f.is_open()) return;

    f << "metric,value\n";
    auto snap = snapshot();
    f << "throughput_tok_per_sec," << snap.throughput_tok_per_sec << "\n";
    f << "avg_latency_ms," << snap.avg_latency_ms << "\n";
    f << "p50_latency_ms," << snap.p50_latency_ms << "\n";
    f << "p90_latency_ms," << snap.p90_latency_ms << "\n";
    f << "p95_latency_ms," << snap.p95_latency_ms << "\n";
    f << "p99_latency_ms," << snap.p99_latency_ms << "\n";
    f << "latency_stddev_ms," << snap.latency_stddev_ms << "\n";
    f << "avg_ttft_ms," << snap.avg_ttft_ms << "\n";
    f << "avg_tpot_ms," << snap.avg_tpot_ms << "\n";
    f << "gpu_cache_usage," << snap.gpu_cache_usage << "\n";
    f << "cpu_cache_usage," << snap.cpu_cache_usage << "\n";
    f << "internal_fragmentation," << snap.internal_fragmentation << "\n";
    f << "external_fragmentation," << snap.external_fragmentation << "\n";
    f << "prefix_cache_hit_rate," << snap.prefix_cache_hit_rate << "\n";
    f << "num_running_seqs," << snap.num_running_seqs << "\n";
    f << "num_waiting_seqs," << snap.num_waiting_seqs << "\n";
    f << "num_swapped_seqs," << snap.num_swapped_seqs << "\n";
    f << "total_requests_served," << snap.total_requests_served << "\n";
    f << "total_tokens_generated," << snap.total_tokens_generated << "\n";
}

}  // namespace cacheflow
