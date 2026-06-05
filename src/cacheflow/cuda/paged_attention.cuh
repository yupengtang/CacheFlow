#pragma once

#include <cstdint>
#include <cstddef>

#ifdef CACHEFLOW_USE_CUDA
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#endif

namespace cacheflow {
namespace cuda {

// ── Paged attention kernel interface ────────────────────────────────────────
//
// Computes scaled dot-product attention where K and V are stored in
// non-contiguous memory blocks (pages).  The block_table maps each
// sequence's logical KV positions to physical block IDs, enabling
// the PagedAttention memory layout.
//
// Two variants:
//   V1: one warp per query head (small context, low latency)
//   V2: partition across warps + reduce (large context, high throughput)

struct PagedAttentionParams {
    const float*   query;           // [num_seqs, num_heads, head_dim]
    const float*   key_cache;       // [num_blocks, num_kv_heads, block_size, head_dim]
    const float*   value_cache;     // [num_blocks, num_kv_heads, block_size, head_dim]
    float*         output;          // [num_seqs, num_heads, head_dim]
    const int32_t* block_tables;    // [num_seqs, max_blocks_per_seq]
    const int32_t* context_lens;    // [num_seqs]
    float          scale;
    int32_t        num_seqs;
    int32_t        num_heads;
    int32_t        num_kv_heads;
    int32_t        head_dim;
    int32_t        block_size;
    int32_t        max_context_len;
    int32_t        max_blocks_per_seq;
};

#ifdef CACHEFLOW_USE_CUDA

void launch_paged_attention_v1(
    const PagedAttentionParams& params,
    cudaStream_t stream = nullptr);

void launch_paged_attention_v2(
    const PagedAttentionParams& params,
    float* exp_sums,       // [num_seqs, num_heads, max_num_partitions]
    float* max_logits,     // [num_seqs, num_heads, max_num_partitions]
    float* tmp_output,     // [num_seqs, num_heads, max_num_partitions, head_dim]
    int32_t num_partitions,
    cudaStream_t stream = nullptr);

#endif  // CACHEFLOW_USE_CUDA

}  // namespace cuda
}  // namespace cacheflow
