#include "bench_config.h"
#include "cacheflow/cacheflow.h"
#include <iostream>
#include <fstream>
#include <thread>
#include <future>
#include <random>
#include <iomanip>

using namespace cacheflow;
using namespace cacheflow::bench;

static std::vector<std::vector<TokenId>>
generate_prompts(size_t num, size_t len, float shared_prefix_ratio) {
    std::mt19937 rng(42);
    std::uniform_int_distribution<TokenId> dist(1, 31999);

    std::vector<TokenId> shared_prefix;
    size_t shared_len = static_cast<size_t>(
        static_cast<float>(len) * shared_prefix_ratio);
    for (size_t i = 0; i < shared_len; ++i)
        shared_prefix.push_back(dist(rng));

    std::vector<std::vector<TokenId>> prompts(num);
    for (size_t i = 0; i < num; ++i) {
        prompts[i] = shared_prefix;
        for (size_t j = shared_len; j < len; ++j)
            prompts[i].push_back(dist(rng));
    }
    return prompts;
}

static BenchResult run_throughput_bench(const BenchConfig& cfg) {
    ModelConfig model = cfg.model;
    CacheConfig cache = cfg.cache;
    SchedulerConfig sched = cfg.scheduler;
    sched.max_num_seqs = cfg.num_concurrent;
    sched.max_num_batched_tokens =
        cfg.num_concurrent * (cfg.prompt_len + cfg.output_len);

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

    auto prompts = generate_prompts(
        cfg.num_requests, cfg.prompt_len, cfg.shared_prefix_ratio);

    SeqId next_id = 0;
    size_t submitted = 0;
    size_t completed = 0;

    while (completed < cfg.num_requests) {
        while (submitted < cfg.num_requests &&
               scheduler.num_running() + scheduler.num_waiting()
                   < cfg.num_concurrent) {
            SamplingParams sp;
            sp.max_tokens = static_cast<int32_t>(cfg.output_len);
            auto group = std::make_unique<SequenceGroup>(
                next_id, prompts[submitted], sp);
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
            for (auto& seq : grp->sequences) {
                if (seq.output_len() == 1) {
                    profiler.record_first_token(seq.seq_id);
                }
            }
        }

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
    result.benchmark_name        = "throughput";
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
    result.prefix_cache_hit_rate = snap.prefix_cache_hit_rate;
    result.num_requests          = cfg.num_requests;
    result.concurrent_level      = cfg.num_concurrent;
    return result;
}

int main(int argc, char** argv) {
    auto cfg = BenchConfig::from_args(argc, argv);
    std::cout << "CacheFlow Throughput Benchmark\n";
    std::cout << "  Requests:     " << cfg.num_requests << "\n";
    std::cout << "  Prompt Len:   " << cfg.prompt_len << "\n";
    std::cout << "  Output Len:   " << cfg.output_len << "\n";
    std::cout << "  Concurrent:   " << cfg.num_concurrent << "\n";
    std::cout << "  Iterations:   " << cfg.num_iterations << "\n\n";

    std::vector<BenchResult> results;
    for (size_t iter = 0; iter < cfg.num_iterations; ++iter) {
        std::cout << "Iteration " << (iter + 1) << "/"
                  << cfg.num_iterations << "...\n";
        auto r = run_throughput_bench(cfg);
        r.benchmark_name = "throughput_iter" + std::to_string(iter + 1);
        r.print();
        results.push_back(std::move(r));
    }

    double avg_tput = 0;
    for (auto& r : results)
        avg_tput += r.throughput_tok_per_sec;
    avg_tput /= static_cast<double>(results.size());

    std::cout << "\n══ Average Throughput: " << std::fixed
              << std::setprecision(1) << avg_tput << " tok/s ══\n";

    std::string csv_path = cfg.output_dir + "/throughput_results.csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "iteration,throughput_tok_s,avg_latency_ms,p50_ms,p90_ms,"
               "p95_ms,p99_ms,stddev_ms,ttft_ms,tpot_ms\n";
        for (size_t i = 0; i < results.size(); ++i) {
            auto& r = results[i];
            csv << (i + 1) << "," << r.throughput_tok_per_sec << ","
                << r.avg_latency_ms << "," << r.p50_latency_ms << ","
                << r.p90_latency_ms << "," << r.p95_latency_ms << ","
                << r.p99_latency_ms << "," << r.latency_stddev_ms << ","
                << r.avg_ttft_ms << "," << r.avg_tpot_ms << "\n";
        }
        std::cout << "Results saved to " << csv_path << "\n";
    }

    return 0;
}
