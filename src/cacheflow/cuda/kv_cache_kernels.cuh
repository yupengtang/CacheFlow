#pragma once

#include <cstdint>
#include <cstddef>

#ifdef CACHEFLOW_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace cacheflow {
namespace cuda {

// ── KV-cache management kernels ─────────────────────────────────────────────

#ifdef CACHEFLOW_USE_CUDA

void launch_reshape_and_cache(
    const float* key,              // [num_tokens, num_kv_heads, head_dim]
    const float* value,            // [num_tokens, num_kv_heads, head_dim]
    float*       key_cache,        // [num_blocks, num_kv_heads, block_size, head_dim]
    float*       value_cache,      // [num_blocks, num_kv_heads, block_size, head_dim]
    const int32_t* slot_mapping,   // [num_tokens]
    int32_t      num_tokens,
    int32_t      num_kv_heads,
    int32_t      head_dim,
    int32_t      block_size,
    cudaStream_t stream = nullptr);

void launch_copy_blocks(
    float*         key_cache,
    float*         value_cache,
    const int64_t* block_mapping,  // [num_pairs, 2]
    int32_t        num_pairs,
    int32_t        num_kv_heads,
    int32_t        head_dim,
    int32_t        block_size,
    cudaStream_t   stream = nullptr);

void launch_swap_blocks(
    float*         src_cache,
    float*         dst_cache,
    const int64_t* block_mapping,
    int32_t        num_pairs,
    int32_t        block_size_bytes,
    cudaStream_t   stream = nullptr);

void launch_compact_kv_cache(
    float*          key_cache,
    float*          value_cache,
    const int32_t*  src_block_ids,
    const int32_t*  dst_block_ids,
    int32_t         num_blocks,
    int32_t         num_kv_heads,
    int32_t         head_dim,
    int32_t         block_size,
    cudaStream_t    stream = nullptr);

#endif  // CACHEFLOW_USE_CUDA

}  // namespace cuda
}  // namespace cacheflow
