#include "cacheflow/cuda/kv_cache_kernels.cuh"

#ifdef CACHEFLOW_USE_CUDA

namespace cacheflow {
namespace cuda {

// ── Reshape and cache: write new K/V into the paged KV-cache ────────────────
//
// Each token's K and V vectors are written to their assigned slot in the
// block-structured cache.  slot_mapping[i] gives the flat slot index
// (block_id * block_size + offset_within_block) for token i.
//
// Memory access: each thread handles one element of head_dim, achieving
// coalesced writes to the cache.

__global__ void reshape_and_cache_kernel(
    const float*   __restrict__ key,
    const float*   __restrict__ value,
    float*         __restrict__ key_cache,
    float*         __restrict__ value_cache,
    const int32_t* __restrict__ slot_mapping,
    int32_t        num_tokens,
    int32_t        num_kv_heads,
    int32_t        head_dim,
    int32_t        block_size)
{
    const int token_idx = blockIdx.x;
    const int head_idx  = blockIdx.y;
    const int d         = threadIdx.x;

    if (token_idx >= num_tokens || d >= head_dim) return;

    const int slot = slot_mapping[token_idx];
    const int block_idx = slot / block_size;
    const int block_off = slot % block_size;

    const int src_off = (token_idx * num_kv_heads + head_idx) * head_dim + d;
    const int dst_off = ((block_idx * num_kv_heads + head_idx) * block_size +
                         block_off) * head_dim + d;

    key_cache[dst_off]   = key[src_off];
    value_cache[dst_off] = value[src_off];
}

void launch_reshape_and_cache(
    const float* key, const float* value,
    float* key_cache, float* value_cache,
    const int32_t* slot_mapping,
    int32_t num_tokens, int32_t num_kv_heads,
    int32_t head_dim, int32_t block_size,
    cudaStream_t stream)
{
    dim3 grid(num_tokens, num_kv_heads);
    dim3 block(head_dim);
    reshape_and_cache_kernel<<<grid, block, 0, stream>>>(
        key, value, key_cache, value_cache, slot_mapping,
        num_tokens, num_kv_heads, head_dim, block_size);
}

// ── Copy blocks: duplicate KV-cache blocks (for COW / prefix sharing) ───────

__global__ void copy_blocks_kernel(
    float*         __restrict__ key_cache,
    float*         __restrict__ value_cache,
    const int64_t* __restrict__ block_mapping,
    int32_t        num_kv_heads,
    int32_t        head_dim,
    int32_t        block_size)
{
    const int pair_idx = blockIdx.x;
    const int64_t src_block = block_mapping[pair_idx * 2];
    const int64_t dst_block = block_mapping[pair_idx * 2 + 1];

    const int elems_per_block = num_kv_heads * block_size * head_dim;
    const int tid = threadIdx.x + blockIdx.y * blockDim.x;

    if (tid < elems_per_block) {
        key_cache[dst_block * elems_per_block + tid] =
            key_cache[src_block * elems_per_block + tid];
        value_cache[dst_block * elems_per_block + tid] =
            value_cache[src_block * elems_per_block + tid];
    }
}

void launch_copy_blocks(
    float* key_cache, float* value_cache,
    const int64_t* block_mapping, int32_t num_pairs,
    int32_t num_kv_heads, int32_t head_dim, int32_t block_size,
    cudaStream_t stream)
{
    int elems_per_block = num_kv_heads * block_size * head_dim;
    int threads = 256;
    int blocks_y = (elems_per_block + threads - 1) / threads;

    dim3 grid(num_pairs, blocks_y);
    dim3 block(threads);
    copy_blocks_kernel<<<grid, block, 0, stream>>>(
        key_cache, value_cache, block_mapping,
        num_kv_heads, head_dim, block_size);
}

// ── Swap blocks: async transfer between GPU and CPU KV-cache ────────────────

__global__ void swap_blocks_kernel(
    float*         __restrict__ dst_cache,
    const float*   __restrict__ src_cache,
    const int64_t* __restrict__ block_mapping,
    int32_t        block_size_elems)
{
    const int pair_idx = blockIdx.x;
    const int64_t src_block = block_mapping[pair_idx * 2];
    const int64_t dst_block = block_mapping[pair_idx * 2 + 1];

    for (int i = threadIdx.x; i < block_size_elems; i += blockDim.x) {
        dst_cache[dst_block * block_size_elems + i] =
            src_cache[src_block * block_size_elems + i];
    }
}

void launch_swap_blocks(
    float* src_cache, float* dst_cache,
    const int64_t* block_mapping, int32_t num_pairs,
    int32_t block_size_bytes, cudaStream_t stream)
{
    int block_size_elems = block_size_bytes / sizeof(float);
    dim3 grid(num_pairs);
    dim3 block(256);
    swap_blocks_kernel<<<grid, block, 0, stream>>>(
        dst_cache, src_cache, block_mapping, block_size_elems);
}

// ── Compact KV-cache: defragment by moving blocks to contiguous locations ───
//
// After many allocations and frees, the physical layout becomes fragmented.
// This kernel copies blocks from scattered source positions to contiguous
// destination positions, reducing the working set's TLB footprint and
// improving memory access locality for subsequent attention passes.

__global__ void compact_kv_cache_kernel(
    float*         __restrict__ key_cache,
    float*         __restrict__ value_cache,
    const int32_t* __restrict__ src_ids,
    const int32_t* __restrict__ dst_ids,
    int32_t        num_kv_heads,
    int32_t        head_dim,
    int32_t        block_size)
{
    const int block_idx     = blockIdx.x;
    const int src           = src_ids[block_idx];
    const int dst           = dst_ids[block_idx];
    if (src == dst) return;

    const int elems_per_block = num_kv_heads * block_size * head_dim;

    for (int i = threadIdx.x + blockIdx.y * blockDim.x;
         i < elems_per_block;
         i += blockDim.x * gridDim.y)
    {
        int src_off = src * elems_per_block + i;
        int dst_off = dst * elems_per_block + i;
        key_cache[dst_off]   = key_cache[src_off];
        value_cache[dst_off] = value_cache[src_off];
    }
}

void launch_compact_kv_cache(
    float* key_cache, float* value_cache,
    const int32_t* src_block_ids, const int32_t* dst_block_ids,
    int32_t num_blocks, int32_t num_kv_heads,
    int32_t head_dim, int32_t block_size,
    cudaStream_t stream)
{
    int elems_per_block = num_kv_heads * block_size * head_dim;
    int threads = 256;
    int blocks_y = (elems_per_block + threads - 1) / threads;

    dim3 grid(num_blocks, blocks_y);
    dim3 block(threads);
    compact_kv_cache_kernel<<<grid, block, 0, stream>>>(
        key_cache, value_cache, src_block_ids, dst_block_ids,
        num_kv_heads, head_dim, block_size);
}

}  // namespace cuda
}  // namespace cacheflow

#endif  // CACHEFLOW_USE_CUDA
