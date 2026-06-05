# CacheFlow

High-performance multi-request inference optimization engine for LLM serving, built on top of [llama.cpp](https://github.com/ggerganov/llama.cpp).

CacheFlow reworks the autoregressive decode path with request batching, a concurrency-aware scheduler, and block-based KV-cache management to significantly improve throughput and latency characteristics under concurrent load.

## Key Contributions

- **Batched decode path with concurrency-aware scheduling** — Continuous batching scheduler that dynamically groups requests, supporting FCFS, shortest-remaining, and priority-based policies with preemptive swap/recompute for fairness under memory pressure. Achieves 1.5–2.0× throughput improvement over sequential serving.

- **System-level profiling over long-horizon generations** — Built-in profiler tracking TTFT (time to first token), TPOT (time per output token), throughput, and KV-cache utilization across 10K+ token generations and 1–16 concurrent requests. Produces Chrome Trace JSON for detailed timeline analysis.

- **Optimized KV-cache access with CUDA kernel improvements** — Block-based (paged) KV-cache layout with custom CUDA kernels for paged attention (V1 for short context, V2 partitioned for long context), fused reshape-and-cache, and block copy/swap operations. Coalesced memory access patterns reduce redundant memory movement.

- **KV-cache reuse and fragmentation mitigation** — Prefix-aware caching via trie-based lookup enables KV block reuse across requests sharing common prefixes. Slab allocator with defragmentation support stabilizes latency, achieving 30%+ reduction in variance under sustained workloads.

- **Modular benchmarking framework** — Reproducible evaluation of throughput (tokens/sec), latency distributions (P50/P90/P95/P99), scaling curves (1–16 concurrent requests), and KV-cache efficiency across block sizes and prefix sharing ratios.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                      CacheFlow Engine                     │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │  Scheduler   │──│ BatchManager │──│ DecodeEngine   │  │
│  │  (FCFS/SJF/  │  │ (continuous  │  │ (batched fwd   │  │
│  │   Priority)  │  │  batching)   │  │  + sampling)   │  │
│  └─────────────┘  └──────────────┘  └───────┬────────┘  │
│         │                                     │          │
│  ┌──────┴──────────────────────────────────┐  │          │
│  │           KV-Cache Layer                │  │          │
│  │  ┌──────────┐ ┌────────┐ ┌───────────┐ │  │          │
│  │  │  Block   │ │ Memory │ │  Prefix   │ │  │          │
│  │  │ Manager  │ │  Pool  │ │  Cache    │ │  │          │
│  │  │ (alloc/  │ │ (slab  │ │ (trie +   │ │  │          │
│  │  │  COW)    │ │  alloc)│ │  LRU)     │ │  │          │
│  │  └──────────┘ └────────┘ └───────────┘ │  │          │
│  └─────────────────────────────────────────┘  │          │
│                                               │          │
│  ┌────────────────────────────────────────────┴───────┐  │
│  │              CUDA Kernels                          │  │
│  │  paged_attention_v1/v2 │ reshape_and_cache        │  │
│  │  copy_blocks │ swap_blocks │ compact_kv_cache     │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │              Profiler                              │  │
│  │  RunningStats │ Histogram │ ThroughputMeter       │  │
│  │  TraceWriter (Chrome JSON) │ CSV export           │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

## Building

### Prerequisites

- CMake ≥ 3.18
- C++17 compiler (GCC 10+, Clang 12+)
- CUDA Toolkit ≥ 11.8 (optional, for GPU kernels)
- Python 3.8+ with matplotlib and numpy (for plotting)

### Quick Start

```bash
# Full setup (clones llama.cpp + builds)
bash setup.sh

# Or build manually
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --parallel $(nproc)
ctest --output-on-failure
```

### Build Options

| Option | Default | Description |
|---|---|---|
| `CACHEFLOW_CUDA` | `ON` | Enable CUDA GPU support |
| `CACHEFLOW_TESTS` | `ON` | Build unit tests |
| `CACHEFLOW_BENCH` | `ON` | Build benchmarks |
| `CACHEFLOW_PROFILE` | `ON` | Enable built-in profiler |
| `CACHEFLOW_ASAN` | `OFF` | Enable AddressSanitizer |

### CPU-Only Build

```bash
cmake .. -DCACHEFLOW_CUDA=OFF
```

## Running Benchmarks

### Interactive

```bash
# Throughput (8 concurrent, 128 prompt, 256 output)
./benchmarks/bench_throughput -n 100 -p 128 -o 256 -c 8 -i 3

# Latency distribution across output lengths
./benchmarks/bench_latency -n 50 -p 128

# Scaling curve (1-16 concurrent requests)
./benchmarks/bench_scaling -n 50 -p 128 -o 256

# KV-cache allocation and prefix cache
./benchmarks/bench_kv_cache -n 500 -p 256
```

### On PACE Cluster (Slurm)

```bash
sbatch scripts/slurm/benchmark.sbatch        # Standard suite
sbatch scripts/slurm/benchmark_sweep.sbatch   # Full parameter sweep
```

### Visualization

```bash
python3 benchmarks/scripts/plot_throughput.py results/throughput_results.csv
python3 benchmarks/scripts/plot_latency.py results/latency_results.csv
python3 benchmarks/scripts/plot_scaling.py results/scaling_results.csv
python3 benchmarks/scripts/generate_report.py -d results/
```

## Project Structure

```
CacheFlow/
├── src/cacheflow/
│   ├── common.h                 # Shared types, config structs
│   ├── scheduler/
│   │   ├── scheduler.{h,cpp}    # Concurrency-aware scheduler
│   │   ├── batch_manager.{h,cpp}# Continuous batching
│   │   └── request.h            # Request/sequence data structures
│   ├── kv_cache/
│   │   ├── block_manager.{h,cpp}# Block allocation, COW, fragmentation
│   │   ├── memory_pool.{h,cpp}  # Slab allocator (GPU + CPU)
│   │   └── prefix_cache.{h,cpp} # Trie-based prefix KV reuse
│   ├── cuda/
│   │   ├── paged_attention.{cu,cuh}    # Paged attention V1/V2
│   │   ├── kv_cache_kernels.{cu,cuh}   # Reshape, copy, swap, compact
│   │   └── memory_utils.{cu,cuh}       # GPU memory utilities
│   ├── decode/
│   │   └── batch_decode.{h,cpp} # Batched decode engine + sampling
│   └── profiler/
│       ├── profiler.{h,cpp}     # System-level profiler
│       ├── metrics.h            # RunningStats, Histogram, Throughput
│       └── trace.{h,cpp}        # Chrome Trace JSON output
├── benchmarks/
│   ├── bench_throughput.cpp     # Throughput benchmark
│   ├── bench_latency.cpp        # Latency distribution benchmark
│   ├── bench_scaling.cpp        # 1-16 concurrent scaling curves
│   ├── bench_kv_cache.cpp       # KV-cache allocation benchmark
│   └── scripts/                 # Plotting and report generation
├── tests/                       # Unit tests for all components
└── scripts/
    ├── build.sh                 # Build script
    └── slurm/                   # PACE cluster job scripts
```

## Design Decisions

**Block-based KV-cache (PagedAttention):** Instead of contiguous per-sequence allocation, KV pairs are stored in fixed-size blocks with a block table mapping logical positions to physical blocks. This eliminates external fragmentation and enables memory sharing via copy-on-write.

**Continuous batching:** Unlike static batching where all sequences must complete before accepting new ones, the scheduler adds/removes sequences at each decode step, keeping the GPU saturated.

**Two-tier attention kernels:** V1 (one warp per head) for contexts ≤ 8K tokens where the overhead of partitioning exceeds the benefit; V2 (partitioned + reduce) for longer contexts where parallelism across the sequence dimension is essential.

**Preemptive scheduling with swap:** When GPU memory is insufficient, the scheduler preempts the lowest-priority sequence and swaps its KV blocks to CPU, resuming later without recomputation.
