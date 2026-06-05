#include "cacheflow/kv_cache/memory_pool.h"
#include <stdexcept>
#include <cassert>
#include <cstring>

#ifdef CACHEFLOW_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace cacheflow {

MemoryPool::MemoryPool(size_t block_size_bytes, size_t num_blocks,
                       DeviceType device)
    : block_size_bytes_(block_size_bytes)
    , num_blocks_(num_blocks)
    , device_(device)
{
    size_t total_bytes = block_size_bytes_ * num_blocks_;
    if (total_bytes == 0) return;

    if (device_ == DeviceType::CUDA) {
#ifdef CACHEFLOW_USE_CUDA
        cudaError_t err = cudaMalloc(&base_, total_bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaMalloc failed: ") +
                cudaGetErrorString(err));
        }
        cudaMemset(base_, 0, total_bytes);
#else
        base_ = std::aligned_alloc(64, total_bytes);
        if (!base_) throw std::bad_alloc();
        std::memset(base_, 0, total_bytes);
#endif
    } else {
        base_ = std::aligned_alloc(64, total_bytes);
        if (!base_) throw std::bad_alloc();
        std::memset(base_, 0, total_bytes);
    }

    free_stack_.reserve(num_blocks_);
    for (size_t i = num_blocks_; i > 0; --i) {
        void* ptr = static_cast<char*>(base_) +
                    (i - 1) * block_size_bytes_;
        free_stack_.push_back(ptr);
    }
}

MemoryPool::~MemoryPool() {
    if (!base_) return;

    if (device_ == DeviceType::CUDA) {
#ifdef CACHEFLOW_USE_CUDA
        cudaFree(base_);
#else
        std::free(base_);
#endif
    } else {
        std::free(base_);
    }
}

void* MemoryPool::allocate() {
    std::lock_guard<std::mutex> lock(mu_);
    if (free_stack_.empty()) return nullptr;
    void* ptr = free_stack_.back();
    free_stack_.pop_back();
    return ptr;
}

void MemoryPool::deallocate(void* ptr) {
    if (!ptr) return;
    std::lock_guard<std::mutex> lock(mu_);
    free_stack_.push_back(ptr);
}

void MemoryPool::reset() {
    std::lock_guard<std::mutex> lock(mu_);
    free_stack_.clear();
    free_stack_.reserve(num_blocks_);
    for (size_t i = num_blocks_; i > 0; --i) {
        void* ptr = static_cast<char*>(base_) +
                    (i - 1) * block_size_bytes_;
        free_stack_.push_back(ptr);
    }
}

size_t MemoryPool::num_free() const {
    std::lock_guard<std::mutex> lock(mu_);
    return free_stack_.size();
}

size_t MemoryPool::num_used() const {
    return num_blocks_ - num_free();
}

float MemoryPool::utilization() const {
    if (num_blocks_ == 0) return 0.0f;
    return static_cast<float>(num_used()) /
           static_cast<float>(num_blocks_);
}

BlockId MemoryPool::ptr_to_block_id(const void* ptr) const {
    auto offset = static_cast<const char*>(ptr) -
                  static_cast<const char*>(base_);
    return static_cast<BlockId>(
        static_cast<size_t>(offset) / block_size_bytes_);
}

void* MemoryPool::block_id_to_ptr(BlockId id) const {
    return static_cast<char*>(base_) +
           static_cast<size_t>(id) * block_size_bytes_;
}

// ── PoolManager ─────────────────────────────────────────────────────────────

PoolManager::PoolManager(size_t block_size_bytes,
                         size_t num_gpu_blocks,
                         size_t num_cpu_blocks)
    : gpu_pool_(std::make_unique<MemoryPool>(
          block_size_bytes, num_gpu_blocks, DeviceType::CUDA))
    , cpu_pool_(std::make_unique<MemoryPool>(
          block_size_bytes, num_cpu_blocks, DeviceType::CPU))
{}

}  // namespace cacheflow
