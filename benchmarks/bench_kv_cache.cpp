#include "bench_config.h"
#include "cacheflow/cacheflow.h"
#include <iostream>
#include <fstream>
#include <random>
#include <iomanip>

using namespace cacheflow;
using namespace cacheflow::bench;

struct KVCacheResult {
    size_t block_size;
    double alloc_throughput_ops;
    double free_throughput_ops;
    float  internal_fragmentation;
    float  external_fragmentation;
    float  prefix_hit_rate;
    double avg_alloc_ns;
    double avg_free_ns;
};

static KVCacheResult run_allocation_bench(
    const BenchConfig& cfg, size_t block_size) {

    ModelConfig model = cfg.model;
    CacheConfig cache = cfg.cache;
    cache.block_size = block_size;

    BlockManager bm(cache, model);
    KVCacheResult result{};
    result.block_size = block_size;

    size_t num_ops = std::min(cache.num_gpu_blocks / 2, size_t(10000));

    auto t0 = Clock::now();
    std::vector<std::pair<SeqId, BlockId>> allocated;
    for (size_t i = 0; i < num_ops; ++i) {
        SeqId sid = static_cast<SeqId>(i);
        BlockId bid = bm.allocate_block(sid, DeviceType::CUDA);
        if (bid != INVALID_BLOCK)
            allocated.emplace_back(sid, bid);
    }
    auto t1 = Clock::now();
    double alloc_ms = Duration(t1 - t0).count();
    result.alloc_throughput_ops = static_cast<double>(allocated.size()) /
                                  (alloc_ms / 1000.0);
    result.avg_alloc_ns = (alloc_ms * 1e6) /
                           static_cast<double>(allocated.size());

    auto frag = bm.compute_fragmentation(DeviceType::CUDA);
    result.internal_fragmentation = frag.internal_fragmentation;

    auto t2 = Clock::now();
    for (size_t i = 0; i < allocated.size(); i += 2) {
        bm.free_block(allocated[i].second, DeviceType::CUDA);
    }
    auto t3 = Clock::now();
    double free_ms = Duration(t3 - t2).count();
    size_t freed = (allocated.size() + 1) / 2;
    result.free_throughput_ops = static_cast<double>(freed) /
                                 (free_ms / 1000.0);
    result.avg_free_ns = (free_ms * 1e6) / static_cast<double>(freed);

    auto frag2 = bm.compute_fragmentation(DeviceType::CUDA);
    result.external_fragmentation = frag2.external_fragmentation;

    return result;
}

static KVCacheResult run_prefix_cache_bench(
    const BenchConfig& cfg, float shared_ratio) {

    ModelConfig model = cfg.model;
    CacheConfig cache = cfg.cache;
    cache.enable_prefix_cache = true;

    BlockManager bm(cache, model);
    PrefixCache pc(cache, &bm);

    std::mt19937 rng(42);
    std::uniform_int_distribution<TokenId> dist(1, 31999);

    size_t prefix_len = static_cast<size_t>(
        static_cast<float>(cfg.prompt_len) * shared_ratio);
    size_t num_blocks = prefix_len / cache.block_size;

    std::vector<TokenId> shared_prefix(prefix_len);
    for (auto& t : shared_prefix) t = dist(rng);

    std::vector<BlockId> prefix_blocks;
    for (size_t i = 0; i < num_blocks; ++i) {
        BlockId bid = bm.allocate_block(0, DeviceType::CUDA);
        prefix_blocks.push_back(bid);
    }
    pc.insert(shared_prefix, prefix_blocks);

    size_t num_lookups = cfg.num_requests;
    for (size_t i = 0; i < num_lookups; ++i) {
        std::vector<TokenId> prompt = shared_prefix;
        for (size_t j = prefix_len; j < cfg.prompt_len; ++j)
            prompt.push_back(dist(rng));

        auto lookup = pc.lookup(prompt);
    }

    KVCacheResult result{};
    result.block_size = cache.block_size;
    result.prefix_hit_rate = pc.hit_rate();
    return result;
}

int main(int argc, char** argv) {
    auto cfg = BenchConfig::from_args(argc, argv);
    std::cout << "CacheFlow KV-Cache Benchmark\n\n";

    std::cout << "── Allocation Performance ──\n";
    std::vector<size_t> block_sizes = {8, 16, 32, 64};
    std::vector<KVCacheResult> alloc_results;

    for (size_t bs : block_sizes) {
        std::cout << "Block size: " << bs << "...\n";
        auto r = run_allocation_bench(cfg, bs);
        std::cout << "  Alloc: " << std::fixed << std::setprecision(0)
                  << r.alloc_throughput_ops << " ops/s ("
                  << std::setprecision(1) << r.avg_alloc_ns << " ns/op)\n";
        std::cout << "  Free:  " << std::setprecision(0)
                  << r.free_throughput_ops << " ops/s ("
                  << std::setprecision(1) << r.avg_free_ns << " ns/op)\n";
        std::cout << "  Int Frag: "
                  << std::setprecision(2)
                  << r.internal_fragmentation * 100 << "%\n";
        std::cout << "  Ext Frag: "
                  << r.external_fragmentation * 100 << "%\n";
        alloc_results.push_back(r);
    }

    std::cout << "\n── Prefix Cache Hit Rates ──\n";
    std::vector<float> shared_ratios = {0.0f, 0.25f, 0.5f, 0.75f, 1.0f};
    std::vector<KVCacheResult> prefix_results;

    for (float sr : shared_ratios) {
        auto r = run_prefix_cache_bench(cfg, sr);
        std::cout << "  Shared " << std::fixed << std::setprecision(0)
                  << sr * 100 << "%: hit rate = "
                  << std::setprecision(2)
                  << r.prefix_hit_rate * 100 << "%\n";
        prefix_results.push_back(r);
    }

    std::string csv_path = cfg.output_dir + "/kv_cache_results.csv";
    std::ofstream csv(csv_path);
    if (csv.is_open()) {
        csv << "block_size,alloc_ops_s,free_ops_s,alloc_ns,"
               "free_ns,int_frag,ext_frag\n";
        for (auto& r : alloc_results) {
            csv << r.block_size << ","
                << r.alloc_throughput_ops << ","
                << r.free_throughput_ops << ","
                << r.avg_alloc_ns << ","
                << r.avg_free_ns << ","
                << r.internal_fragmentation << ","
                << r.external_fragmentation << "\n";
        }
        std::cout << "Results saved to " << csv_path << "\n";
    }

    return 0;
}
