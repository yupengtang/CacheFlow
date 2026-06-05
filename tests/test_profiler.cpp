#include "cacheflow/profiler/profiler.h"
#include "cacheflow/profiler/metrics.h"
#include <cassert>
#include <iostream>
#include <cmath>
#include <thread>
#include <chrono>
#include <sstream>

using namespace cacheflow;

static void test_running_stats() {
    RunningStats stats;
    stats.update(10.0);
    stats.update(20.0);
    stats.update(30.0);

    assert(stats.count() == 3);
    assert(std::abs(stats.mean() - 20.0) < 0.01);
    assert(stats.min() == 10.0);
    assert(stats.max() == 30.0);
    assert(stats.stddev() > 0.0);

    stats.reset();
    assert(stats.count() == 0);

    std::cout << "  PASS: test_running_stats\n";
}

static void test_histogram() {
    Histogram hist(0.0, 100.0, 100);

    for (int i = 0; i < 100; ++i)
        hist.record(static_cast<double>(i));

    assert(hist.total() == 100);
    [[maybe_unused]] double p50 = hist.percentile(50);
    assert(p50 > 45.0 && p50 < 55.0);

    [[maybe_unused]] double p99 = hist.percentile(99);
    assert(p99 > 95.0);

    hist.record(-5.0);
    assert(hist.underflow() == 1);
    hist.record(200.0);
    assert(hist.overflow() == 1);

    std::cout << "  PASS: test_histogram\n";
}

static void test_throughput_meter() {
    ThroughputMeter meter;

    meter.record_tokens(100);
    meter.record_interval(10.0);
    meter.record_tokens(200);
    meter.record_interval(20.0);

    assert(meter.total_tokens() == 300);
    [[maybe_unused]] double tps = meter.tokens_per_sec();
    assert(tps > 0.0);
    assert(std::abs(tps - 10000.0) < 100.0);

    meter.reset();
    assert(meter.total_tokens() == 0);

    std::cout << "  PASS: test_throughput_meter\n";
}

static void test_profiler_lifecycle() {
    Profiler::Config cfg;
    cfg.enable_tracing = false;
    cfg.enable_metrics = true;
    Profiler prof(cfg);

    prof.begin_step(32);
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    prof.end_step(1.5);

    prof.begin_step(64);
    prof.end_step(2.0);

    [[maybe_unused]] auto snap = prof.snapshot();
    assert(snap.throughput_tok_per_sec > 0);
    assert(snap.avg_latency_ms > 0);

    std::cout << "  PASS: test_profiler_lifecycle\n";
}

static void test_request_tracking() {
    Profiler::Config cfg;
    cfg.enable_tracing = false;
    Profiler prof(cfg);

    prof.record_request_start(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    prof.record_first_token(1);
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    prof.record_request_end(1, 10);

    assert(prof.ttft_stats().count() == 1);
    assert(prof.ttft_stats().mean() > 0);
    assert(prof.tpot_stats().count() == 1);

    std::cout << "  PASS: test_request_tracking\n";
}

static void test_dump_summary() {
    Profiler::Config cfg;
    cfg.enable_tracing = false;
    Profiler prof(cfg);

    prof.begin_step(100);
    prof.end_step(5.0);

    std::ostringstream oss;
    prof.dump_summary(oss);
    std::string output = oss.str();
    assert(output.find("CacheFlow") != std::string::npos);
    assert(output.find("Throughput") != std::string::npos);

    std::cout << "  PASS: test_dump_summary\n";
}

static void test_profiler_reset() {
    Profiler::Config cfg;
    cfg.enable_tracing = false;
    Profiler prof(cfg);

    prof.begin_step(50);
    prof.end_step(3.0);

    prof.reset();
    [[maybe_unused]] auto snap = prof.snapshot();
    assert(snap.total_tokens_generated == 0);
    assert(snap.avg_latency_ms == 0.0);

    std::cout << "  PASS: test_profiler_reset\n";
}

int main() {
    std::cout << "=== Profiler Tests ===\n";
    test_running_stats();
    test_histogram();
    test_throughput_meter();
    test_profiler_lifecycle();
    test_request_tracking();
    test_dump_summary();
    test_profiler_reset();
    std::cout << "All profiler tests passed.\n";
    return 0;
}
