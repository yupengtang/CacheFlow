#include "cacheflow/decode/batch_decode.h"
#include "cacheflow/profiler/profiler.h"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <numeric>
#include <random>

#ifdef CACHEFLOW_USE_CUDA
#include "cacheflow/cuda/paged_attention.cuh"
#include "cacheflow/cuda/kv_cache_kernels.cuh"
#include <cuda_runtime.h>
#endif

namespace cacheflow {

BatchDecodeEngine::BatchDecodeEngine(
    const ModelConfig& model_config,
    const CacheConfig& cache_config,
    BlockManager* block_manager,
    PoolManager* pool_manager)
    : model_config_(model_config)
    , cache_config_(cache_config)
    , block_manager_(block_manager)
    , pool_manager_(pool_manager)
{}

BatchDecodeEngine::~BatchDecodeEngine() = default;

// ── Main decode step ────────────────────────────────────────────────────────

BatchDecodeEngine::StepResult
BatchDecodeEngine::step(const DecodeBatch& batch) {
    StepResult result;
    if (batch.empty()) return result;

    auto t_start = Clock::now();

#ifdef CACHEFLOW_PROFILING
    if (profiler_) profiler_->begin_step(batch.total_tokens);
#endif

    DecodeContext ctx;
    ctx.batch = batch;
    ctx.num_seqs = batch.entries.size();
    ctx.logits.resize(ctx.num_seqs * model_config_.vocab_size, 0.0f);

    prepare_kv_cache_inputs(batch, ctx);

#ifdef CACHEFLOW_PROFILING
    if (profiler_) profiler_->mark("kv_cache_prepared");
#endif

    execute_attention(ctx);

#ifdef CACHEFLOW_PROFILING
    if (profiler_) profiler_->mark("attention_done");
#endif

    std::vector<SamplingParams> default_params(ctx.num_seqs);
    result.new_tokens = sample(ctx.logits, batch, default_params);
    result.tokens_processed = batch.total_tokens;

    auto t_end = Clock::now();
    result.step_time_ms = Duration(t_end - t_start).count();

    ++stats_.total_steps;
    stats_.total_tokens += result.tokens_processed;
    stats_.total_time_ms += result.step_time_ms;
    stats_.avg_step_time_ms =
        stats_.total_time_ms / static_cast<double>(stats_.total_steps);
    stats_.tokens_per_sec =
        static_cast<double>(stats_.total_tokens) /
        (stats_.total_time_ms / 1000.0);

    for (auto& e : batch.entries) {
        if (e.is_prefill)
            ++stats_.total_prefill_tokens;
        else
            ++stats_.total_decode_tokens;
    }

#ifdef CACHEFLOW_PROFILING
    if (profiler_) profiler_->end_step(result.step_time_ms);
#endif

    return result;
}

// ── Prepare KV-cache block tables for the GPU kernel ────────────────────────

void BatchDecodeEngine::prepare_kv_cache_inputs(
    const DecodeBatch& batch, DecodeContext& ctx) {

    size_t max_blocks = 0;
    for (auto& e : batch.entries)
        max_blocks = std::max(max_blocks, e.block_table.size());

    size_t num_seqs = batch.entries.size();
    auto* tables = new int32_t[num_seqs * max_blocks]();
    auto* ctx_lens = new int32_t[num_seqs];
    auto* slots = new int32_t[batch.total_tokens];

    for (size_t i = 0; i < num_seqs; ++i) {
        auto& entry = batch.entries[i];
        ctx_lens[i] = entry.position + 1;
        for (size_t b = 0; b < entry.block_table.size(); ++b)
            tables[i * max_blocks + b] = entry.block_table[b];

        int block_idx = entry.position /
                        static_cast<int>(cache_config_.block_size);
        int block_off = entry.position %
                        static_cast<int>(cache_config_.block_size);
        if (block_idx < static_cast<int>(entry.block_table.size()))
            slots[i] = entry.block_table[block_idx] *
                        static_cast<int>(cache_config_.block_size) +
                        block_off;
    }

    ctx.block_tables  = tables;
    ctx.context_lens  = ctx_lens;
    ctx.slot_mapping  = slots;
    ctx.key_cache_ptr =
        static_cast<float*>(pool_manager_->gpu_pool().base_ptr());
    ctx.value_cache_ptr =
        ctx.key_cache_ptr +
        (pool_manager_->gpu_pool().num_blocks() *
         model_config_.n_kv_heads *
         static_cast<int>(cache_config_.block_size) *
         model_config_.head_dim);
}

// ── Attention execution ─────────────────────────────────────────────────────

void BatchDecodeEngine::execute_attention(DecodeContext& ctx) {
#ifdef CACHEFLOW_USE_CUDA
    cuda::PagedAttentionParams params{};
    params.query          = nullptr;
    params.key_cache      = ctx.key_cache_ptr;
    params.value_cache    = ctx.value_cache_ptr;
    params.output         = nullptr;
    params.block_tables   = ctx.block_tables;
    params.context_lens   = ctx.context_lens;
    params.scale          = 1.0f / std::sqrt(
                                static_cast<float>(model_config_.head_dim));
    params.num_seqs       = static_cast<int32_t>(ctx.num_seqs);
    params.num_heads      = model_config_.n_heads;
    params.num_kv_heads   = model_config_.n_kv_heads;
    params.head_dim       = model_config_.head_dim;
    params.block_size     = static_cast<int32_t>(cache_config_.block_size);

    int max_ctx = 0;
    for (size_t i = 0; i < ctx.num_seqs; ++i)
        max_ctx = std::max(max_ctx, ctx.context_lens[i]);
    params.max_context_len = max_ctx;

    size_t max_blocks = 0;
    for (auto& e : ctx.batch.entries)
        max_blocks = std::max(max_blocks, e.block_table.size());
    params.max_blocks_per_seq = static_cast<int32_t>(max_blocks);

    constexpr int PARTITION_SIZE = 512;
    if (max_ctx <= 8192) {
        cuda::launch_paged_attention_v1(params);
    } else {
        int num_partitions =
            (max_ctx + PARTITION_SIZE - 1) / PARTITION_SIZE;
        size_t part_elems = static_cast<size_t>(params.num_seqs) *
                            params.num_heads * num_partitions;
        std::vector<float> exp_sums(part_elems, 0.0f);
        std::vector<float> max_logits(part_elems, 0.0f);
        std::vector<float> tmp_out(
            part_elems * params.head_dim, 0.0f);
        cuda::launch_paged_attention_v2(
            params, exp_sums.data(), max_logits.data(),
            tmp_out.data(), num_partitions);
    }
#else
    execute_attention_cpu(ctx);
#endif

    delete[] ctx.block_tables;
    delete[] ctx.context_lens;
    delete[] ctx.slot_mapping;
    ctx.block_tables  = nullptr;
    ctx.context_lens  = nullptr;
    ctx.slot_mapping  = nullptr;
}

void BatchDecodeEngine::execute_attention_cpu(DecodeContext& ctx) {
    float scale = 1.0f / std::sqrt(
                      static_cast<float>(model_config_.head_dim));
    int n_heads     = model_config_.n_heads;
    int n_kv_heads  = model_config_.n_kv_heads;
    int head_dim    = model_config_.head_dim;
    int block_size  = static_cast<int>(cache_config_.block_size);

    float* k_cache = ctx.key_cache_ptr;
    float* v_cache = ctx.value_cache_ptr;

    if (!k_cache || !v_cache) return;

    for (size_t s = 0; s < ctx.num_seqs; ++s) {
        auto& entry = ctx.batch.entries[s];
        int ctx_len = entry.position + 1;
        int kv_group = n_heads / n_kv_heads;

        for (int h = 0; h < n_heads; ++h) {
            int kv_h = h / kv_group;
            std::vector<float> scores(ctx_len, 0.0f);
            float max_score = -1e30f;

            for (int t = 0; t < ctx_len; ++t) {
                int blk = t / block_size;
                int off = t % block_size;
                if (blk >= static_cast<int>(entry.block_table.size()))
                    continue;
                int phys = entry.block_table[blk];
                const float* k = k_cache +
                    ((phys * n_kv_heads + kv_h) * block_size + off) *
                    head_dim;

                float dot = 0.0f;
                for (int d = 0; d < head_dim; ++d)
                    dot += k[d] * k[d];
                scores[t] = dot * scale;
                max_score = std::max(max_score, scores[t]);
            }

            float sum = 0.0f;
            for (int t = 0; t < ctx_len; ++t) {
                scores[t] = std::exp(scores[t] - max_score);
                sum += scores[t];
            }
            float inv_sum = 1.0f / (sum + 1e-8f);
            for (int t = 0; t < ctx_len; ++t)
                scores[t] *= inv_sum;
        }
    }
}

// ── Sampling ────────────────────────────────────────────────────────────────

std::vector<TokenId>
BatchDecodeEngine::sample(const std::vector<float>& logits,
                          const DecodeBatch& batch,
                          const std::vector<SamplingParams>& params) {
    std::vector<TokenId> tokens;
    tokens.reserve(batch.entries.size());
    thread_local std::mt19937 rng(42);

    for (size_t i = 0; i < batch.entries.size(); ++i) {
        const float* seq_logits =
            logits.data() + i * model_config_.vocab_size;
        const auto& sp = (i < params.size())
            ? params[i] : SamplingParams{};

        if (sp.temperature < 1e-6f) {
            auto max_it = std::max_element(
                seq_logits, seq_logits + model_config_.vocab_size);
            tokens.push_back(
                static_cast<TokenId>(max_it - seq_logits));
            continue;
        }

        std::vector<std::pair<float, TokenId>> scored(
            model_config_.vocab_size);
        for (int v = 0; v < model_config_.vocab_size; ++v) {
            scored[v] = {seq_logits[v] / sp.temperature,
                         static_cast<TokenId>(v)};
        }

        if (sp.top_k > 0 && sp.top_k < model_config_.vocab_size) {
            std::partial_sort(scored.begin(),
                              scored.begin() + sp.top_k,
                              scored.end(),
                              [](auto& a, auto& b) {
                                  return a.first > b.first;
                              });
            scored.resize(static_cast<size_t>(sp.top_k));
        }

        float max_logit = scored[0].first;
        float sum_exp = 0.0f;
        for (auto& [l, _] : scored) {
            l = std::exp(l - max_logit);
            sum_exp += l;
        }
        for (auto& [l, _] : scored)
            l /= sum_exp;

        if (sp.top_p < 1.0f) {
            std::sort(scored.begin(), scored.end(),
                      [](auto& a, auto& b) {
                          return a.first > b.first;
                      });
            float cumsum = 0.0f;
            size_t cutoff = scored.size();
            for (size_t j = 0; j < scored.size(); ++j) {
                cumsum += scored[j].first;
                if (cumsum >= sp.top_p) {
                    cutoff = j + 1;
                    break;
                }
            }
            scored.resize(cutoff);
        }

        float resum = 0.0f;
        for (auto& [l, _] : scored) resum += l;
        std::uniform_real_distribution<float> dist(0.0f, resum);
        float r = dist(rng);
        float acc = 0.0f;
        TokenId chosen = scored[0].second;
        for (auto& [l, tok] : scored) {
            acc += l;
            if (acc >= r) { chosen = tok; break; }
        }
        tokens.push_back(chosen);
    }

    return tokens;
}

}  // namespace cacheflow
