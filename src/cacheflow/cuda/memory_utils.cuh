#pragma once

#include <cstdint>
#include <cstddef>

#ifdef CACHEFLOW_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace cacheflow {
namespace cuda {

// ── GPU memory utilities ────────────────────────────────────────────────────

#ifdef CACHEFLOW_USE_CUDA

size_t get_gpu_free_memory();
size_t get_gpu_total_memory();
float  get_gpu_memory_utilization();

void async_memcpy_h2d(void* dst, const void* src, size_t bytes,
                      cudaStream_t stream = nullptr);
void async_memcpy_d2h(void* dst, const void* src, size_t bytes,
                      cudaStream_t stream = nullptr);
void async_memcpy_d2d(void* dst, const void* src, size_t bytes,
                      cudaStream_t stream = nullptr);

void launch_memset_blocks(
    float*         cache,
    const int32_t* block_ids,
    int32_t        num_blocks,
    int32_t        block_size_bytes,
    cudaStream_t   stream = nullptr);

void synchronize_stream(cudaStream_t stream);
void synchronize_device();

#endif  // CACHEFLOW_USE_CUDA

}  // namespace cuda
}  // namespace cacheflow
