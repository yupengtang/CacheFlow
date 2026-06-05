#include "cacheflow/cuda/memory_utils.cuh"

#ifdef CACHEFLOW_USE_CUDA

#include <cstdio>

namespace cacheflow {
namespace cuda {

size_t get_gpu_free_memory() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return free_bytes;
}

size_t get_gpu_total_memory() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return total_bytes;
}

float get_gpu_memory_utilization() {
    size_t free_bytes, total_bytes;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    return 1.0f - static_cast<float>(free_bytes) /
                   static_cast<float>(total_bytes);
}

void async_memcpy_h2d(void* dst, const void* src, size_t bytes,
                      cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream);
}

void async_memcpy_d2h(void* dst, const void* src, size_t bytes,
                      cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream);
}

void async_memcpy_d2d(void* dst, const void* src, size_t bytes,
                      cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream);
}

// ── Memset blocks: zero out specific blocks in the KV-cache ─────────────────

__global__ void memset_blocks_kernel(
    float*         __restrict__ cache,
    const int32_t* __restrict__ block_ids,
    int32_t        block_size_elems)
{
    const int block_idx = blockIdx.x;
    const int bid       = block_ids[block_idx];
    const int base      = bid * block_size_elems;

    for (int i = threadIdx.x; i < block_size_elems; i += blockDim.x)
        cache[base + i] = 0.0f;
}

void launch_memset_blocks(
    float* cache, const int32_t* block_ids,
    int32_t num_blocks, int32_t block_size_bytes,
    cudaStream_t stream)
{
    int block_size_elems = block_size_bytes / sizeof(float);
    dim3 grid(num_blocks);
    dim3 block(256);
    memset_blocks_kernel<<<grid, block, 0, stream>>>(
        cache, block_ids, block_size_elems);
}

void synchronize_stream(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
}

void synchronize_device() {
    cudaDeviceSynchronize();
}

}  // namespace cuda
}  // namespace cacheflow

#endif  // CACHEFLOW_USE_CUDA
