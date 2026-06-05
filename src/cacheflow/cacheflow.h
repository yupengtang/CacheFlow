#pragma once

// CacheFlow: High-Performance Multi-Request Inference Engine
//
// A set of optimizations for llama.cpp targeting multi-request serving:
//   - Continuous batching with concurrency-aware scheduling
//   - Block-based KV-cache management (PagedAttention-inspired)
//   - CUDA kernel-level KV-cache access optimization
//   - System-level profiling and benchmarking

#include "cacheflow/common.h"
#include "cacheflow/scheduler/request.h"
#include "cacheflow/scheduler/scheduler.h"
#include "cacheflow/scheduler/batch_manager.h"
#include "cacheflow/kv_cache/block_manager.h"
#include "cacheflow/kv_cache/memory_pool.h"
#include "cacheflow/kv_cache/prefix_cache.h"
#include "cacheflow/decode/batch_decode.h"
#include "cacheflow/profiler/profiler.h"

namespace cacheflow {

struct EngineConfig {
    ModelConfig      model;
    SchedulerConfig  scheduler;
    CacheConfig      cache;
};

}  // namespace cacheflow
