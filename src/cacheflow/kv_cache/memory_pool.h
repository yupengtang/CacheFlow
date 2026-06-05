#pragma once

#include "cacheflow/common.h"
#include <cstdlib>
#include <vector>
#include <memory>
#include <mutex>

namespace cacheflow {

// ── Slab allocator for KV-cache blocks ──────────────────────────────────────
//
// Pre-allocates a contiguous memory region and serves fixed-size block
// allocations from it, eliminating per-block malloc overhead and reducing
// fragmentation.  The GPU variant uses cudaMalloc; CPU uses aligned_alloc.

class MemoryPool {
public:
    MemoryPool(size_t block_size_bytes, size_t num_blocks, DeviceType device);
    ~MemoryPool();

    MemoryPool(const MemoryPool&) = delete;
    MemoryPool& operator=(const MemoryPool&) = delete;

    void* allocate();
    void  deallocate(void* ptr);
    void  reset();

    size_t block_size_bytes() const { return block_size_bytes_; }
    size_t num_blocks()       const { return num_blocks_; }
    size_t num_free()         const;
    size_t num_used()         const;
    float  utilization()      const;
    void*  base_ptr()         const { return base_; }
    DeviceType device()       const { return device_; }

    BlockId ptr_to_block_id(const void* ptr) const;
    void*   block_id_to_ptr(BlockId id) const;

private:
    size_t       block_size_bytes_;
    size_t       num_blocks_;
    DeviceType   device_;
    void*        base_ = nullptr;
    std::vector<void*> free_stack_;
    mutable std::mutex mu_;
};

// ── Unified pool manager (GPU + CPU) ────────────────────────────────────────

class PoolManager {
public:
    PoolManager(size_t block_size_bytes,
                size_t num_gpu_blocks,
                size_t num_cpu_blocks);

    MemoryPool& gpu_pool() { return *gpu_pool_; }
    MemoryPool& cpu_pool() { return *cpu_pool_; }
    const MemoryPool& gpu_pool() const { return *gpu_pool_; }
    const MemoryPool& cpu_pool() const { return *cpu_pool_; }

    MemoryPool& pool_for(DeviceType device) {
        return device == DeviceType::CUDA ? *gpu_pool_ : *cpu_pool_;
    }

private:
    std::unique_ptr<MemoryPool> gpu_pool_;
    std::unique_ptr<MemoryPool> cpu_pool_;
};

}  // namespace cacheflow
