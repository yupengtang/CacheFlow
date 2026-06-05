#pragma once

#include "cacheflow/common.h"
#include <string>
#include <vector>
#include <iostream>
#include <getopt.h>

namespace cacheflow {
namespace bench {

struct BenchConfig {
    ModelConfig    model;
    CacheConfig    cache;
    SchedulerConfig scheduler;

    size_t num_requests        = 100;
    size_t warmup_requests     = 10;
    size_t prompt_len          = 128;
    size_t output_len          = 256;
    size_t num_concurrent      = 1;
    size_t num_iterations      = 3;
    bool   enable_prefix_cache = true;
    float  shared_prefix_ratio = 0.0f;
    std::string output_dir     = "results";
    std::string output_format  = "csv";
    bool   verbose             = false;

    static BenchConfig from_args(int argc, char** argv) {
        BenchConfig cfg;
        static struct option long_opts[] = {
            {"num-requests",     required_argument, nullptr, 'n'},
            {"prompt-len",       required_argument, nullptr, 'p'},
            {"output-len",       required_argument, nullptr, 'o'},
            {"concurrent",       required_argument, nullptr, 'c'},
            {"iterations",       required_argument, nullptr, 'i'},
            {"gpu-blocks",       required_argument, nullptr, 'g'},
            {"block-size",       required_argument, nullptr, 'b'},
            {"output-dir",       required_argument, nullptr, 'd'},
            {"prefix-cache",     no_argument,       nullptr, 'x'},
            {"no-prefix-cache",  no_argument,       nullptr, 'X'},
            {"shared-prefix",    required_argument, nullptr, 's'},
            {"verbose",          no_argument,       nullptr, 'v'},
            {"help",             no_argument,       nullptr, 'h'},
            {nullptr, 0, nullptr, 0}
        };

        int opt;
        while ((opt = getopt_long(argc, argv, "n:p:o:c:i:g:b:d:s:xXvh",
                                  long_opts, nullptr)) != -1) {
            switch (opt) {
            case 'n': cfg.num_requests = std::stoull(optarg); break;
            case 'p': cfg.prompt_len   = std::stoull(optarg); break;
            case 'o': cfg.output_len   = std::stoull(optarg); break;
            case 'c': cfg.num_concurrent = std::stoull(optarg); break;
            case 'i': cfg.num_iterations = std::stoull(optarg); break;
            case 'g': cfg.cache.num_gpu_blocks = std::stoull(optarg); break;
            case 'b': cfg.cache.block_size     = std::stoull(optarg); break;
            case 'd': cfg.output_dir = optarg; break;
            case 'x': cfg.enable_prefix_cache = true; break;
            case 'X': cfg.enable_prefix_cache = false; break;
            case 's': cfg.shared_prefix_ratio = std::stof(optarg); break;
            case 'v': cfg.verbose = true; break;
            case 'h':
                print_usage();
                std::exit(0);
            default: break;
            }
        }
        return cfg;
    }

    static void print_usage() {
        std::cout << R"(
CacheFlow Benchmark Suite
Usage: bench_<type> [options]

Options:
  -n, --num-requests <N>     Number of requests (default: 100)
  -p, --prompt-len <N>       Prompt length in tokens (default: 128)
  -o, --output-len <N>       Output length in tokens (default: 256)
  -c, --concurrent <N>       Number of concurrent requests (default: 1)
  -i, --iterations <N>       Number of iterations (default: 3)
  -g, --gpu-blocks <N>       Number of GPU cache blocks (default: 8192)
  -b, --block-size <N>       Tokens per cache block (default: 16)
  -d, --output-dir <path>    Output directory (default: results)
  -x, --prefix-cache         Enable prefix caching
  -X, --no-prefix-cache      Disable prefix caching
  -s, --shared-prefix <0-1>  Fraction of shared prefix (default: 0)
  -v, --verbose              Verbose output
  -h, --help                 Show this help
)";
    }
};

struct BenchResult {
    std::string benchmark_name;
    double throughput_tok_per_sec   = 0.0;
    double avg_latency_ms          = 0.0;
    double p50_latency_ms          = 0.0;
    double p90_latency_ms          = 0.0;
    double p95_latency_ms          = 0.0;
    double p99_latency_ms          = 0.0;
    double latency_stddev_ms       = 0.0;
    double avg_ttft_ms             = 0.0;
    double avg_tpot_ms             = 0.0;
    float  gpu_cache_utilization   = 0.0f;
    float  fragmentation           = 0.0f;
    float  prefix_cache_hit_rate   = 0.0f;
    size_t num_requests            = 0;
    size_t concurrent_level        = 0;

    void print(std::ostream& os = std::cout) const {
        os << "\n── " << benchmark_name << " ──\n";
        os << "  Throughput:      " << throughput_tok_per_sec << " tok/s\n";
        os << "  Avg Latency:     " << avg_latency_ms << " ms\n";
        os << "  P50 Latency:     " << p50_latency_ms << " ms\n";
        os << "  P90 Latency:     " << p90_latency_ms << " ms\n";
        os << "  P95 Latency:     " << p95_latency_ms << " ms\n";
        os << "  P99 Latency:     " << p99_latency_ms << " ms\n";
        os << "  Latency StdDev:  " << latency_stddev_ms << " ms\n";
        os << "  Avg TTFT:        " << avg_ttft_ms << " ms\n";
        os << "  Avg TPOT:        " << avg_tpot_ms << " ms\n";
        os << "  GPU Cache Usage: " << gpu_cache_utilization * 100 << " %\n";
        os << "  Fragmentation:   " << fragmentation * 100 << " %\n";
        os << "  Prefix Hit Rate: " << prefix_cache_hit_rate * 100 << " %\n";
    }
};

}  // namespace bench
}  // namespace cacheflow
