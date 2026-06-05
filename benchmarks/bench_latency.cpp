#include "bench_config.h"
#include "cacheflow/cacheflow.h"
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>
#include <algorithm>

using namespace cacheflow;
using namespace cacheflow::bench;

static BenchResult run_latency_bench(
    const BenchConfig& cfg, size_t target_output_len) {

    ModelConfig model = cfg.model;
    CacheConfig cache = cfg.cache;
    SchedulerConfig sched = cfg.scheduler;
    sched.max_num_seqs = 1;

    BlockManager block_manager(cache, model);
    PoolManager pool_manager(model.block_bytes(),
                             cache.num_gpu_blocks, cache.num_cpu_blocks);
    Scheduler scheduler(sched, &block_manager);
    BatchManager batch_manager(model, sched);
    BatchDecodeEngine engine(model, cache, &block_manager, &pool_manager);

    Profiler profiler;
    profiler.set_block_manager(&block_manager);
    engine.set_profiler(&profiler);

    std::mt19937 rng(42);
    std::uniform_int_distribution<TokenId> dist(1, 31999);

    std::vector<double> per_request_latencies;
    std::vector<double> ttft_values;
    std::vector<double> tpot_values;

    for (size_t r = 0; r < cfg.num_requests; ++r) {
        std::vector<TokenId> prompt(cfg.prompt_len);
        for (auto& t : prompt) t = dist(rng);

        SamplingParams sp;
        sp.max_tokens = static_cast<int32_t>(target_output_len);

        auto group = std::make_unique<SequenceGroup>(
            static_cast<SeqId>(r), prompt, sp);

        auto req_start = Clock::now();
        profiler.record_request_start(static_cast<SeqId>(r));
        scheduler.add_request(std::move(group));

        TimePoint first_tok_time;
        bool got_first = false;
        size_t tokens_generated = 0;

        while (scheduler.has_pending()) {
            auto sched_out = scheduler.schedule();
            if (sched_out.is_empty()) break;

            auto batch = batch_manager.build_batch(sched_out);
            auto result = engine.step(batch);
            batch_manager.update_after_decode(sched_out, result.new_tokens);

            if (!got_first && !result.new_tokens.empty()) {
                first_tok_time = Clock::now();
                got_first = true;
                profiler.record_first_token(static_cast<SeqId>(r));
            }
            tokens_generated += result.new_tokens.size();
            scheduler.free_finished_requests();
        }

        auto req_end = Clock::now();
        double req_ms = Duration(req_end - req_start).count();
        per_request_latencies.push_back(req_ms);

        if (got_first) {
            double ttft = Duration(first_tok_time - req_start).count();
            ttft_values.push_back(ttft);

            if (tokens_generated > 1) {
                double decode_ms = Duration(
                    req_end - first_tok_time).count();
                tpot_values.push_back(
                    decode_ms / static_cast<double>(tokens_generated - 1));
            }
        }

        profiler.record_request_end(
            static_cast<SeqId>(r), tokens_generated);
    }

    std::sort(per_request_latencies.begin(), per_request_latencies.end());

    auto pct = [&](double p) -> double {
        size_t idx = static_cast<size_t>(
            p / 100.0 * static_cast<double>(per_request_latencies.size()));
        idx = std::min(idx, per_request_latencies.size() - 1);
        return per_request_latencies[idx];
    };

    double sum = 0;
    for (auto v : per_request_latencies) sum += v;
    double avg = sum / static_cast<double>(per_request_latencies.size());

    double var = 0;
    for (auto v : per_request_latencies) var += (v - avg) * (v - avg);
    var /= static_cast<double>(per_request_latencies.size());

    double avg_ttft = 0;
    for (auto v : ttft_values) avg_ttft += v;
    if (!ttft_values.empty())
        avg_ttft /= static_cast<double>(ttft_values.size());

    double avg_tpot = 0;
    for (auto v : tpot_values) avg_tpot += v;
    if (!tpot_values.empty())
        avg_tpot /= static_cast<double>(tpot_values.size());

    BenchResult result;
    result.benchmark_name   = "latency_output" +
                              std::to_string(target_output_len);
    result.avg_latency_ms   = avg;
    result.p50_latency_ms   = pct(50);
    result.p90_latency_ms   = pct(90);
    result.p95_latency_ms   = pct(95);
    result.p99_latency_ms   = pct(99);
    result.latency_stddev_ms = std::sqrt(var);
    result.avg_ttft_ms      = avg_ttft;
    result.avg_tpot_ms      = avg_tpot;
    result.num_requests     = cfg.num_requests;

    auto snap = profiler.snapshot();
    result.throughput_tok_per_sec = snap.throughput_tok_per_sec;
    result.gpu_cache_utilization  = snap.gpu_cache_usage;
    result.fragmentation          = snap.internal_fragmentation;

    return result;
}

int main(int argc, char** argv) {
    auto cfg = BenchConfig::from_args(argc, argv);
    std::cout << "CacheFlow Latency Benchmark\n";
    std::cout << "  Requests:   " << cfg.num_requests << "\n";
    std::cout << "  Prompt Len: " << cfg.prompt_len << "\n\n";

    std::vector<size_t> output_lens = {32, 64, 128, 256, 512, 1024};
    std::vector<BenchResult> all_results;

    for (size_t olen : output_lens) {
        std::cout << "Output length: " << olen << " tokens...\n";
        auto r = run_latency_bench(cfg, olen);
        r.print();
        all_results.push_back(std::move(r));
    }

    std::string csv_path = cfg.output_dir + "/latency_results.csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "output_len,avg_ms,p50_ms,p90_ms,p95_ms,p99_ms,"
               "stddev_ms,ttft_ms,tpot_ms,throughput_tok_s\n";
        for (size_t i = 0; i < output_lens.size(); ++i) {
            auto& r = all_results[i];
            csv << output_lens[i] << "," << r.avg_latency_ms << ","
                << r.p50_latency_ms << "," << r.p90_latency_ms << ","
                << r.p95_latency_ms << "," << r.p99_latency_ms << ","
                << r.latency_stddev_ms << "," << r.avg_ttft_ms << ","
                << r.avg_tpot_ms << "," << r.throughput_tok_per_sec
                << "\n";
        }
        std::cout << "Results saved to " << csv_path << "\n";
    }

    return 0;
}
