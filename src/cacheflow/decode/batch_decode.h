#pragma once

#include "cacheflow/common.h"
#include "cacheflow/scheduler/request.h"
#include "cacheflow/scheduler/batch_manager.h"
#include "cacheflow/kv_cache/block_manager.h"
#include "cacheflow/kv_cache/memory_pool.h"
#include <functional>
#include <vector>
#include <memory>

namespace cacheflow {

class Profiler;

// ── Decode context: per-step working state ──────────────────────────────────

struct DecodeContext {
    DecodeBatch                 batch;
    std::vector<float>          logits;
    std::vector<TokenId>        sampled_tokens;

    float*   key_cache_ptr   = nullptr;
    float*   value_cache_ptr = nullptr;
    int32_t* block_tables    = nullptr;
    int32_t* context_lens    = nullptr;
    int32_t* slot_mapping    = nullptr;
    size_t   num_seqs        = 0;
};

// ── Model forward function signature ────────────────────────────────────────
// This abstracts away the actual model forward pass (llama.cpp or standalone).

using ForwardFn = std::function<void(
    const float* input_embeddings,
    float*       output_logits,
    float*       key_out,
    float*       value_out,
    const int32_t* positions,
    int32_t        num_tokens,
    int32_t        num_layers
)>;

// ── Batched decode engine ───────────────────────────────────────────────────

class BatchDecodeEngine {
public:
    BatchDecodeEngine(const ModelConfig& model_config,
                      const CacheConfig& cache_config,
                      BlockManager* block_manager,
                      PoolManager* pool_manager);
    ~BatchDecodeEngine();

    BatchDecodeEngine(const BatchDecodeEngine&) = delete;
    BatchDecodeEngine& operator=(const BatchDecodeEngine&) = delete;

    void set_forward_fn(ForwardFn fn) { forward_fn_ = std::move(fn); }
    void set_profiler(Profiler* profiler) { profiler_ = profiler; }

    struct StepResult {
        std::vector<TokenId> new_tokens;
        double               step_time_ms = 0.0;
        size_t               tokens_processed = 0;
    };

    StepResult step(const DecodeBatch& batch);

    void prepare_kv_cache_inputs(const DecodeBatch& batch,
                                 DecodeContext& ctx);

    std::vector<TokenId> sample(const std::vector<float>& logits,
                                const DecodeBatch& batch,
                                const std::vector<SamplingParams>& params);

    struct EngineStats {
        uint64_t total_steps          = 0;
        uint64_t total_tokens         = 0;
        uint64_t total_prefill_tokens = 0;
        uint64_t total_decode_tokens  = 0;
        double   total_time_ms        = 0.0;
        double   avg_step_time_ms     = 0.0;
        double   tokens_per_sec       = 0.0;
    };
    EngineStats stats() const { return stats_; }
    void reset_stats() { stats_ = {}; }

private:
    void execute_attention(DecodeContext& ctx);
    void execute_attention_cpu(DecodeContext& ctx);

    ModelConfig   model_config_;
    CacheConfig   cache_config_;
    BlockManager* block_manager_;
    PoolManager*  pool_manager_;
    Profiler*     profiler_ = nullptr;
    ForwardFn     forward_fn_;
    EngineStats   stats_;
};

}  // namespace cacheflow
