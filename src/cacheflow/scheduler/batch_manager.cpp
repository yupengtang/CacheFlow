#include "cacheflow/scheduler/batch_manager.h"
#include <algorithm>
#include <numeric>
#include <cassert>

namespace cacheflow {

BatchManager::BatchManager(const ModelConfig& model_config,
                           const SchedulerConfig& sched_config)
    : model_config_(model_config)
    , max_batch_tokens_(sched_config.max_num_batched_tokens)
    , max_num_seqs_(sched_config.max_num_seqs)
    , last_stats_{} {}

BatchManager::~BatchManager() = default;

DecodeBatch BatchManager::build_batch(const SchedulerOutput& sched_output) {
    current_batch_.clear();
    last_stats_ = {};

    for (auto* group : sched_output.scheduled_groups) {
        for (auto& seq : group->sequences) {
            if (seq.is_finished()) continue;

            bool is_prefill = (seq.output_len() == 0);

            if (is_prefill) {
                for (size_t i = 0; i < seq.prompt_len(); ++i) {
                    current_batch_.add(
                        seq.seq_id,
                        seq.prompt_tokens[i],
                        static_cast<int32_t>(i),
                        seq.logical_blocks,
                        true);
                    ++last_stats_.num_prefill_tokens;
                }
            } else {
                TokenId last_tok = seq.output_tokens.empty()
                    ? seq.prompt_tokens.back()
                    : seq.output_tokens.back();
                current_batch_.add(
                    seq.seq_id,
                    last_tok,
                    static_cast<int32_t>(seq.total_len() - 1),
                    seq.logical_blocks,
                    false);
                ++last_stats_.num_decode_tokens;
            }

            ++last_stats_.num_sequences;
        }
    }

    compute_padding(current_batch_);
    last_stats_.gpu_utilization_est = estimate_gpu_util(current_batch_);
    return current_batch_;
}

void BatchManager::update_after_decode(SchedulerOutput& sched_output,
                                       const std::vector<TokenId>& new_tokens) {
    size_t tok_idx = 0;
    for (auto* group : sched_output.scheduled_groups) {
        for (auto& seq : group->sequences) {
            if (seq.is_finished()) continue;
            if (tok_idx >= new_tokens.size()) break;

            TokenId new_tok = new_tokens[tok_idx++];
            seq.append_token(new_tok);

            if (!group->first_token_time.has_value() && seq.output_len() == 1) {
                group->first_token_time = Clock::now();
            }

            if (static_cast<int32_t>(seq.output_len()) >=
                    group->sampling.max_tokens) {
                seq.status = RequestStatus::FINISHED_LENGTH;
            }
        }
    }
}

void BatchManager::compute_padding(DecodeBatch& batch) {
    if (batch.entries.empty()) return;

    int32_t max_pos = 0;
    for (auto& e : batch.entries)
        max_pos = std::max(max_pos, e.position);

    size_t actual_tokens = batch.entries.size();
    size_t padded_tokens = last_stats_.num_sequences *
                           static_cast<size_t>(max_pos + 1);
    last_stats_.padding_tokens =
        (padded_tokens > actual_tokens) ? (padded_tokens - actual_tokens) : 0;
}

float BatchManager::estimate_gpu_util(const DecodeBatch& batch) const {
    if (batch.empty()) return 0.0f;

    size_t compute_tokens = last_stats_.num_prefill_tokens +
                            last_stats_.num_decode_tokens;
    float utilization = static_cast<float>(compute_tokens) /
                        static_cast<float>(max_batch_tokens_);
    return std::min(utilization, 1.0f);
}

}  // namespace cacheflow
