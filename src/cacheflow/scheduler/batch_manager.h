#pragma once

#include "cacheflow/common.h"
#include "cacheflow/scheduler/request.h"
#include "cacheflow/scheduler/scheduler.h"
#include <vector>
#include <memory>

namespace cacheflow {

// ── Batch: a set of token positions to decode in one forward pass ────────────

struct DecodeBatch {
    struct Entry {
        SeqId              seq_id;
        TokenId            token;
        int32_t            position;
        std::vector<BlockId> block_table;
        bool               is_prefill;
    };

    std::vector<Entry> entries;
    size_t total_tokens = 0;

    void clear() {
        entries.clear();
        total_tokens = 0;
    }

    bool empty() const { return entries.empty(); }

    void add(SeqId sid, TokenId tok, int32_t pos,
             std::vector<BlockId> blocks, bool prefill) {
        entries.push_back({sid, tok, pos, std::move(blocks), prefill});
        ++total_tokens;
    }
};

// ── BatchManager: converts scheduler output to executable batches ───────────

class BatchManager {
public:
    explicit BatchManager(const ModelConfig& model_config,
                          const SchedulerConfig& sched_config);
    ~BatchManager();

    BatchManager(const BatchManager&) = delete;
    BatchManager& operator=(const BatchManager&) = delete;

    DecodeBatch build_batch(const SchedulerOutput& sched_output);
    void update_after_decode(SchedulerOutput& sched_output,
                             const std::vector<TokenId>& new_tokens);

    size_t max_batch_tokens() const { return max_batch_tokens_; }
    size_t current_batch_size() const { return current_batch_.entries.size(); }

    struct BatchStats {
        size_t num_prefill_tokens   = 0;
        size_t num_decode_tokens    = 0;
        size_t num_sequences        = 0;
        size_t padding_tokens       = 0;
        float  gpu_utilization_est  = 0.0f;
    };
    BatchStats last_stats() const { return last_stats_; }

private:
    void compute_padding(DecodeBatch& batch);
    float estimate_gpu_util(const DecodeBatch& batch) const;

    ModelConfig     model_config_;
    size_t          max_batch_tokens_;
    size_t          max_num_seqs_;
    DecodeBatch     current_batch_;
    BatchStats      last_stats_;
};

}  // namespace cacheflow
