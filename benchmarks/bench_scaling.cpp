#include "bench_config.h"
#include "cacheflow/cacheflow.h"
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>

using namespace cacheflow;
using namespace cacheflow::bench;

static BenchResult run_scaling_point(const BenchConfig& cfg,
                                     size_t concurrency) {
    ModelConfig model = cfg.model;
    CacheConfig cache = cfg.cache;
    SchedulerConfig sched = cfg.scheduler;
    sched.max_num_seqs = concurrency;
    sched.max_num_batched_tokens =
        concurrency * (cfg.prompt_len + cfg.output_len);

    BlockManager block_manager(cache, model);
    PoolManager pool_manager(model.block_bytes(),
                             cache.num_gpu_blocks, cache.num_cpu_blocks);
    Scheduler scheduler(sched, &block_manager);
    BatchManager batch_manager(model, sched);
    BatchDecodeEngine engine(model, cache, &block_manager, &pool_manager);

    Profiler profiler;
    profiler.set_block_manager(&block_manager);
    profiler.set_scheduler(&scheduler);
    engine.set_profiler(&profiler);

    std::mt19937 rng(42);
    std::uniform_int_distribution<TokenId> dist(1, 31999);

    SeqId next_id = 0;
    size_t submitted = 0;
    size_t completed = 0;
    size_t target = cfg.num_requests;

    while (completed < target) {
        while (submitted < target &&
               scheduler.num_running() + scheduler.num_waiting()
                   < concurrency) {
            std::vector<TokenId> prompt(cfg.prompt_len);
            for (auto& t : prompt) t = dist(rng);

            SamplingParams sp;
            sp.max_tokens = static_cast<int32_t>(cfg.output_len);
            auto group = std::make_unique<SequenceGroup>(
                next_id, std::move(prompt), sp);
            profiler.record_request_start(next_id);
            scheduler.add_request(std::move(group));
            ++next_id;
            ++submitted;
        }

        auto sched_out = scheduler.schedule();
        if (sched_out.is_empty()) break;

        auto batch = batch_manager.build_batch(sched_out);
        auto step_result = engine.step(batch);
        batch_manager.update_after_decode(sched_out, step_result.new_tokens);

        for (auto* grp : sched_out.scheduled_groups) {
            if (grp->all_finished()) {
                profiler.record_request_end(
                    grp->request_id,
                    grp->sequences[0].output_len());
                ++completed;
            }
        }
        scheduler.free_finished_requests();
    }

    auto snap = profiler.snapshot();
    BenchResult result;
    result.benchmark_name        = "scaling_c" + std::to_string(concurrency);
    result.throughput_tok_per_sec = snap.throughput_tok_per_sec;
    result.avg_latency_ms        = snap.avg_latency_ms;
    result.p50_latency_ms        = snap.p50_latency_ms;
    result.p90_latency_ms        = snap.p90_latency_ms;
    result.p95_latency_ms        = snap.p95_latency_ms;
    result.p99_latency_ms        = snap.p99_latency_ms;
    result.latency_stddev_ms     = snap.latency_stddev_ms;
    result.avg_ttft_ms           = snap.avg_ttft_ms;
    result.avg_tpot_ms           = snap.avg_tpot_ms;
    result.gpu_cache_utilization = snap.gpu_cache_usage;
    result.fragmentation         = snap.internal_fragmentation;
    result.num_requests          = target;
    result.concurrent_level      = concurrency;
    return result;
}

int main(int argc, char** argv) {
    auto cfg = BenchConfig::from_args(argc, argv);
    std::cout << "CacheFlow Scaling Benchmark\n";
    std::cout << "  Requests per point: " << cfg.num_requests << "\n";
    std::cout << "  Prompt Len:         " << cfg.prompt_len << "\n";
    std::cout << "  Output Len:         " << cfg.output_len << "\n\n";

    std::vector<size_t> concurrency_levels =
        {1, 2, 4, 8, 12, 16};
    std::vector<BenchResult> results;

    for (size_t c : concurrency_levels) {
        std::cout << "Concurrency: " << c << "...\n";
        double total_tput = 0;
        BenchResult best;
        for (size_t iter = 0; iter < cfg.num_iterations; ++iter) {
            auto r = run_scaling_point(cfg, c);
            total_tput += r.throughput_tok_per_sec;
            if (iter == 0 || r.throughput_tok_per_sec >
                             best.throughput_tok_per_sec)
                best = r;
        }
        best.throughput_tok_per_sec =
            total_tput / static_cast<double>(cfg.num_iterations);
        best.print();
        results.push_back(std::move(best));
    }

    std::cout << "\n══ Scaling Summary ══\n";
    std::cout << std::setw(12) << "Concurrent"
              << std::setw(15) << "Throughput"
              << std::setw(12) << "Avg Lat"
              << std::setw(12) << "P99 Lat"
              << std::setw(12) << "Speedup"
              << "\n";
    double baseline = results[0].throughput_tok_per_sec;
    for (auto& r : results) {
        std::cout << std::setw(12) << r.concurrent_level
                  << std::setw(12) << std::fixed << std::setprecision(1)
                  << r.throughput_tok_per_sec << " t/s"
                  << std::setw(10) << std::setprecision(2)
                  << r.avg_latency_ms << " ms"
                  << std::setw(10) << r.p99_latency_ms << " ms"
                  << std::setw(10) << std::setprecision(2)
                  << r.throughput_tok_per_sec / baseline << "x"
                  << "\n";
    }

    std::string csv_path = cfg.output_dir + "/scaling_results.csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "concurrency,throughput_tok_s,avg_latency_ms,p50_ms,"
               "p90_ms,p95_ms,p99_ms,stddev_ms,gpu_util,fragmentation\n";
        for (auto& r : results) {
            csv << r.concurrent_level << ","
                << r.throughput_tok_per_sec << ","
                << r.avg_latency_ms << ","
                << r.p50_latency_ms << ","
                << r.p90_latency_ms << ","
                << r.p95_latency_ms << ","
                << r.p99_latency_ms << ","
                << r.latency_stddev_ms << ","
                << r.gpu_cache_utilization << ","
                << r.fragmentation << "\n";
        }
        std::cout << "Results saved to " << csv_path << "\n";
    }

    return 0;
}
