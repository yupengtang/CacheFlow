#pragma once

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <atomic>
#include <limits>

namespace cacheflow {

// ── Compile-time configuration ──────────────────────────────────────────────

constexpr size_t KV_BLOCK_SIZE       = 16;
constexpr size_t MAX_SEQ_LEN         = 32768;
constexpr size_t MAX_BATCH_TOKENS    = 4096;
constexpr size_t MAX_CONCURRENT_REQS = 256;
constexpr size_t DEFAULT_GPU_BLOCKS  = 8192;
constexpr size_t DEFAULT_CPU_BLOCKS  = 16384;

// ── Aliases ─────────────────────────────────────────────────────────────────

using TokenId    = int32_t;
using SeqId      = uint64_t;
using BlockId    = int32_t;
using Clock      = std::chrono::steady_clock;
using TimePoint  = Clock::time_point;
using Duration   = std::chrono::duration<double, std::milli>;

constexpr BlockId INVALID_BLOCK = -1;

// ── Enumerations ────────────────────────────────────────────────────────────

enum class RequestStatus : uint8_t {
    WAITING,
    RUNNING,
    PREEMPTED,
    SWAPPED,
    FINISHED_STOPPED,
    FINISHED_LENGTH,
    FINISHED_EOS,
    FINISHED_ABORT,
};

inline bool is_finished(RequestStatus s) {
    return s >= RequestStatus::FINISHED_STOPPED;
}

enum class SchedulePolicy : uint8_t {
    FCFS,
    SHORTEST_REMAINING,
    PRIORITY,
};

enum class PreemptionMode : uint8_t {
    RECOMPUTE,
    SWAP,
};

enum class DeviceType : uint8_t {
    CPU,
    CUDA,
};

// ── Lightweight span (C++17, before std::span) ──────────────────────────────

template <typename T>
struct Span {
    T*     data;
    size_t size;

    T&       operator[](size_t i)       { return data[i]; }
    const T& operator[](size_t i) const { return data[i]; }
    T*       begin()       { return data; }
    T*       end()         { return data + size; }
    const T* begin() const { return data; }
    const T* end()   const { return data + size; }
};

// ── Model configuration ─────────────────────────────────────────────────────

struct ModelConfig {
    int32_t n_layers    = 32;
    int32_t n_heads     = 32;
    int32_t n_kv_heads  = 32;
    int32_t head_dim    = 128;
    int32_t vocab_size  = 32000;
    int32_t max_seq_len = 4096;

    size_t kv_cache_elem_size() const {
        return static_cast<size_t>(head_dim) * n_kv_heads;
    }

    size_t block_bytes() const {
        return 2ULL * n_layers * kv_cache_elem_size() * KV_BLOCK_SIZE * sizeof(float);
    }
};

// ── Scheduler configuration ─────────────────────────────────────────────────

struct SchedulerConfig {
    size_t          max_num_seqs          = 256;
    size_t          max_num_batched_tokens = 4096;
    size_t          max_paddings           = 256;
    SchedulePolicy  policy                = SchedulePolicy::FCFS;
    PreemptionMode  preemption_mode       = PreemptionMode::SWAP;
    float           gpu_memory_utilization = 0.90f;
    size_t          swap_space_bytes       = 4ULL * 1024 * 1024 * 1024;
};

// ── Cache configuration ─────────────────────────────────────────────────────

struct CacheConfig {
    size_t num_gpu_blocks     = DEFAULT_GPU_BLOCKS;
    size_t num_cpu_blocks     = DEFAULT_CPU_BLOCKS;
    size_t block_size         = KV_BLOCK_SIZE;
    bool   enable_prefix_cache = true;
    float  watermark_low      = 0.10f;
    float  watermark_high     = 0.90f;
};

// ── Token-level timing ──────────────────────────────────────────────────────

struct TokenTiming {
    TimePoint queued;
    TimePoint scheduled;
    TimePoint decoded;
};

}  // namespace cacheflow
