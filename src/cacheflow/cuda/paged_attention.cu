#include "cacheflow/cuda/paged_attention.cuh"

#ifdef CACHEFLOW_USE_CUDA

#include <cfloat>
#include <cstdio>

namespace cacheflow {
namespace cuda {

// ── Warp-level primitives ───────────────────────────────────────────────────

static constexpr int WARP_SIZE = 32;

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}

// ── Paged Attention V1 ─────────────────────────────────────────────────────
//
// One thread block per (sequence, head) pair.  Each warp computes attention
// over a contiguous range of KV positions, then warps reduce across the
// block.  Suited for short-to-medium contexts (< 8K tokens).
//
// Memory access pattern: the block_table indirection means K/V reads are
// non-sequential in physical memory, but within a block the access is
// coalesced along head_dim (threads in a warp read consecutive floats).

template <int HEAD_DIM, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v1_kernel(
    float*         __restrict__ output,
    const float*   __restrict__ query,
    const float*   __restrict__ key_cache,
    const float*   __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    float          scale,
    int32_t        num_heads,
    int32_t        num_kv_heads,
    int32_t        max_blocks_per_seq)
{
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int warp_idx = threadIdx.x / WARP_SIZE;
    const int lane_idx = threadIdx.x % WARP_SIZE;

    const int num_warps  = NUM_THREADS / WARP_SIZE;
    const int kv_head    = head_idx / (num_heads / num_kv_heads);
    const int ctx_len    = context_lens[seq_idx];
    const int num_blocks = (ctx_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const float* q = query + (seq_idx * num_heads + head_idx) * HEAD_DIM;

    __shared__ float smem_qk[NUM_THREADS];
    __shared__ float smem_max[NUM_THREADS / WARP_SIZE];
    __shared__ float smem_sum[NUM_THREADS / WARP_SIZE];

    float qk_max = -FLT_MAX;

    extern __shared__ float smem_out[];

    for (int d = threadIdx.x; d < HEAD_DIM; d += NUM_THREADS)
        smem_out[d] = 0.0f;

    for (int block_idx = warp_idx; block_idx < num_blocks;
         block_idx += num_warps) {

        const int phys_block = block_tables[
            seq_idx * max_blocks_per_seq + block_idx];

        for (int tok_off = 0; tok_off < BLOCK_SIZE; ++tok_off) {
            const int tok_pos = block_idx * BLOCK_SIZE + tok_off;
            if (tok_pos >= ctx_len) break;

            const float* k = key_cache +
                ((phys_block * num_kv_heads + kv_head) * BLOCK_SIZE +
                 tok_off) * HEAD_DIM;

            float dot = 0.0f;
            for (int d = lane_idx; d < HEAD_DIM; d += WARP_SIZE)
                dot += q[d] * k[d];
            dot = warp_reduce_sum(dot) * scale;

            if (lane_idx == 0)
                smem_qk[block_idx * BLOCK_SIZE + tok_off] = dot;
            qk_max = fmaxf(qk_max, dot);
        }
    }

    qk_max = warp_reduce_max(qk_max);
    if (lane_idx == 0) smem_max[warp_idx] = qk_max;
    __syncthreads();

    if (warp_idx == 0) {
        float val = (lane_idx < num_warps) ? smem_max[lane_idx] : -FLT_MAX;
        val = warp_reduce_max(val);
        if (lane_idx == 0) smem_max[0] = val;
    }
    __syncthreads();
    qk_max = smem_max[0];

    float exp_sum = 0.0f;
    for (int block_idx = warp_idx; block_idx < num_blocks;
         block_idx += num_warps) {
        for (int tok_off = 0; tok_off < BLOCK_SIZE; ++tok_off) {
            int tok_pos = block_idx * BLOCK_SIZE + tok_off;
            if (tok_pos >= ctx_len) break;

            float& qk_val = smem_qk[block_idx * BLOCK_SIZE + tok_off];
            float  w = __expf(qk_val - qk_max);
            qk_val = w;
            if (lane_idx == 0) exp_sum += w;
        }
    }

    exp_sum = warp_reduce_sum(exp_sum);
    if (lane_idx == 0) smem_sum[warp_idx] = exp_sum;
    __syncthreads();

    if (warp_idx == 0) {
        float val = (lane_idx < num_warps) ? smem_sum[lane_idx] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_idx == 0) smem_sum[0] = val;
    }
    __syncthreads();
    float inv_sum = 1.0f / (smem_sum[0] + 1e-8f);

    float local_out[HEAD_DIM / WARP_SIZE + 1] = {};

    for (int block_idx = warp_idx; block_idx < num_blocks;
         block_idx += num_warps) {
        const int phys_block = block_tables[
            seq_idx * max_blocks_per_seq + block_idx];

        for (int tok_off = 0; tok_off < BLOCK_SIZE; ++tok_off) {
            int tok_pos = block_idx * BLOCK_SIZE + tok_off;
            if (tok_pos >= ctx_len) break;

            float w = smem_qk[block_idx * BLOCK_SIZE + tok_off] * inv_sum;

            const float* v = value_cache +
                ((phys_block * num_kv_heads + kv_head) * BLOCK_SIZE +
                 tok_off) * HEAD_DIM;

            int idx = 0;
            for (int d = lane_idx; d < HEAD_DIM; d += WARP_SIZE, ++idx)
                local_out[idx] += w * v[d];
        }
    }

    int idx = 0;
    for (int d = lane_idx; d < HEAD_DIM; d += WARP_SIZE, ++idx)
        atomicAdd(&smem_out[d], local_out[idx]);

    __syncthreads();

    float* out = output + (seq_idx * num_heads + head_idx) * HEAD_DIM;
    for (int d = threadIdx.x; d < HEAD_DIM; d += NUM_THREADS)
        out[d] = smem_out[d];
}

// ── Paged Attention V2 ─────────────────────────────────────────────────────
//
// Partitioned across the context length dimension.  Each partition computes
// a partial softmax (local max + exp sum + weighted V sum), then a final
// reduce kernel merges partitions using the log-sum-exp correction.
// Suited for long contexts (> 8K tokens).

template <int HEAD_DIM, int BLOCK_SIZE, int NUM_THREADS, int PARTITION_SIZE>
__global__ void paged_attention_v2_kernel(
    float*         __restrict__ exp_sums,
    float*         __restrict__ max_logits,
    float*         __restrict__ tmp_output,
    const float*   __restrict__ query,
    const float*   __restrict__ key_cache,
    const float*   __restrict__ value_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ context_lens,
    float          scale,
    int32_t        num_heads,
    int32_t        num_kv_heads,
    int32_t        max_blocks_per_seq,
    int32_t        num_partitions)
{
    const int seq_idx     = blockIdx.x;
    const int head_idx    = blockIdx.y;
    const int part_idx    = blockIdx.z;
    const int warp_idx    = threadIdx.x / WARP_SIZE;
    const int lane_idx    = threadIdx.x % WARP_SIZE;
    const int kv_head     = head_idx / (num_heads / num_kv_heads);

    const int ctx_len    = context_lens[seq_idx];
    const int start_tok  = part_idx * PARTITION_SIZE;
    if (start_tok >= ctx_len) return;
    const int end_tok    = min(start_tok + PARTITION_SIZE, ctx_len);

    const float* q = query + (seq_idx * num_heads + head_idx) * HEAD_DIM;

    float local_max = -FLT_MAX;
    float local_sum = 0.0f;
    float local_out[HEAD_DIM / WARP_SIZE + 1] = {};

    const int start_block = start_tok / BLOCK_SIZE;
    const int end_block   = (end_tok + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int blk = start_block + warp_idx; blk < end_block;
         blk += (NUM_THREADS / WARP_SIZE)) {
        const int phys = block_tables[
            seq_idx * max_blocks_per_seq + blk];

        for (int off = 0; off < BLOCK_SIZE; ++off) {
            int pos = blk * BLOCK_SIZE + off;
            if (pos < start_tok || pos >= end_tok) continue;

            const float* k = key_cache +
                ((phys * num_kv_heads + kv_head) * BLOCK_SIZE + off) *
                HEAD_DIM;

            float dot = 0.0f;
            for (int d = lane_idx; d < HEAD_DIM; d += WARP_SIZE)
                dot += q[d] * k[d];
            dot = warp_reduce_sum(dot) * scale;

            float old_max = local_max;
            local_max = fmaxf(local_max, dot);
            float correction = __expf(old_max - local_max);
            local_sum = local_sum * correction + __expf(dot - local_max);

            const float* v = value_cache +
                ((phys * num_kv_heads + kv_head) * BLOCK_SIZE + off) *
                HEAD_DIM;
            float w = __expf(dot - local_max);

            int idx = 0;
            for (int d = lane_idx; d < HEAD_DIM; d += WARP_SIZE, ++idx)
                local_out[idx] = local_out[idx] * correction + w * v[d];
        }
    }

    __shared__ float s_max[NUM_THREADS / WARP_SIZE];
    __shared__ float s_sum[NUM_THREADS / WARP_SIZE];

    local_max = warp_reduce_max(local_max);
    if (lane_idx == 0) s_max[warp_idx] = local_max;
    __syncthreads();

    if (warp_idx == 0 && lane_idx == 0) {
        float gmax = -FLT_MAX;
        for (int w = 0; w < NUM_THREADS / WARP_SIZE; ++w)
            gmax = fmaxf(gmax, s_max[w]);
        s_max[0] = gmax;
    }
    __syncthreads();
    float global_max = s_max[0];

    float correction = __expf(local_max - global_max);
    local_sum *= correction;
    for (int i = 0; i < HEAD_DIM / WARP_SIZE + 1; ++i)
        local_out[i] *= correction;

    local_sum = warp_reduce_sum(local_sum);
    if (lane_idx == 0) s_sum[warp_idx] = local_sum;
    __syncthreads();

    if (warp_idx == 0 && lane_idx == 0) {
        float total = 0.0f;
        for (int w = 0; w < NUM_THREADS / WARP_SIZE; ++w)
            total += s_sum[w];
        s_sum[0] = total;
    }
    __syncthreads();

    int flat = (seq_idx * num_heads + head_idx) * num_partitions + part_idx;
    if (threadIdx.x == 0) {
        max_logits[flat] = global_max;
        exp_sums[flat]   = s_sum[0];
    }

    float* out = tmp_output + flat * HEAD_DIM;
    int idx = 0;
    for (int d = lane_idx; d < HEAD_DIM; d += WARP_SIZE, ++idx)
        atomicAdd(&out[d], local_out[idx]);
}

template <int HEAD_DIM>
__global__ void paged_attention_v2_reduce_kernel(
    float*       __restrict__ output,
    const float* __restrict__ exp_sums,
    const float* __restrict__ max_logits,
    const float* __restrict__ tmp_output,
    const int32_t* __restrict__ context_lens,
    int32_t      num_heads,
    int32_t      num_partitions)
{
    const int seq_idx  = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int d        = threadIdx.x;

    if (d >= HEAD_DIM) return;

    const int base = (seq_idx * num_heads + head_idx) * num_partitions;
    const int ctx_len = context_lens[seq_idx];
    const int num_parts = (ctx_len + 511) / 512;

    float global_max = -FLT_MAX;
    for (int p = 0; p < num_parts; ++p)
        global_max = fmaxf(global_max, max_logits[base + p]);

    float acc = 0.0f;
    float total_weight = 0.0f;
    for (int p = 0; p < num_parts; ++p) {
        float correction = __expf(max_logits[base + p] - global_max);
        float weight     = exp_sums[base + p] * correction;
        total_weight    += weight;
        acc += weight * tmp_output[(base + p) * HEAD_DIM + d];
    }

    output[(seq_idx * num_heads + head_idx) * HEAD_DIM + d] =
        acc / (total_weight + 1e-8f);
}

// ── Dispatch functions ──────────────────────────────────────────────────────

void launch_paged_attention_v1(
    const PagedAttentionParams& p,
    cudaStream_t stream)
{
    constexpr int NUM_THREADS = 128;
    dim3 grid(p.num_seqs, p.num_heads);
    dim3 block(NUM_THREADS);
    size_t smem = p.head_dim * sizeof(float) +
                  p.max_context_len * sizeof(float);

    if (p.head_dim == 128 && p.block_size == 16) {
        paged_attention_v1_kernel<128, 16, NUM_THREADS>
            <<<grid, block, smem, stream>>>(
                p.output, p.query, p.key_cache, p.value_cache,
                p.block_tables, p.context_lens, p.scale,
                p.num_heads, p.num_kv_heads, p.max_blocks_per_seq);
    } else if (p.head_dim == 64 && p.block_size == 16) {
        paged_attention_v1_kernel<64, 16, NUM_THREADS>
            <<<grid, block, smem, stream>>>(
                p.output, p.query, p.key_cache, p.value_cache,
                p.block_tables, p.context_lens, p.scale,
                p.num_heads, p.num_kv_heads, p.max_blocks_per_seq);
    }
}

void launch_paged_attention_v2(
    const PagedAttentionParams& p,
    float* exp_sums,
    float* max_logits,
    float* tmp_output,
    int32_t num_partitions,
    cudaStream_t stream)
{
    constexpr int NUM_THREADS    = 128;
    constexpr int PARTITION_SIZE = 512;
    dim3 grid(p.num_seqs, p.num_heads, num_partitions);
    dim3 block(NUM_THREADS);

    if (p.head_dim == 128 && p.block_size == 16) {
        paged_attention_v2_kernel<128, 16, NUM_THREADS, PARTITION_SIZE>
            <<<grid, block, 0, stream>>>(
                exp_sums, max_logits, tmp_output,
                p.query, p.key_cache, p.value_cache,
                p.block_tables, p.context_lens, p.scale,
                p.num_heads, p.num_kv_heads, p.max_blocks_per_seq,
                num_partitions);
    }

    dim3 reduce_grid(p.num_seqs, p.num_heads);
    dim3 reduce_block(p.head_dim);
    paged_attention_v2_reduce_kernel<128>
        <<<reduce_grid, reduce_block, 0, stream>>>(
            p.output, exp_sums, max_logits, tmp_output,
            p.context_lens, p.num_heads, num_partitions);
}

}  // namespace cuda
}  // namespace cacheflow

#endif  // CACHEFLOW_USE_CUDA
