#pragma once

#include "cacheflow/common.h"
#include <deque>
#include <mutex>
#include <unordered_map>
#include <optional>

namespace cacheflow {

// ── Sampling parameters per request ─────────────────────────────────────────

struct SamplingParams {
    float   temperature   = 0.8f;
    float   top_p         = 0.95f;
    int32_t top_k         = 40;
    float   repeat_penalty = 1.1f;
    int32_t max_tokens    = 512;
    bool    stream        = false;
};

// ── Sequence data: tracks generated tokens and KV blocks for one sequence ──

struct SequenceData {
    SeqId                  seq_id;
    std::vector<TokenId>   prompt_tokens;
    std::vector<TokenId>   output_tokens;
    std::vector<BlockId>   logical_blocks;
    RequestStatus          status = RequestStatus::WAITING;

    size_t prompt_len()  const { return prompt_tokens.size(); }
    size_t output_len()  const { return output_tokens.size(); }
    size_t total_len()   const { return prompt_len() + output_len(); }

    size_t num_blocks_needed(size_t block_size) const {
        return (total_len() + block_size - 1) / block_size;
    }

    void append_token(TokenId tok) {
        output_tokens.push_back(tok);
    }

    bool is_finished() const {
        return cacheflow::is_finished(status);
    }
};

// ── Sequence group: a request can have multiple sequences (beam search) ─────

struct SequenceGroup {
    SeqId                           request_id;
    std::vector<SequenceData>       sequences;
    SamplingParams                  sampling;
    TimePoint                       arrival_time;
    int32_t                         priority = 0;
    std::optional<TimePoint>        first_token_time;
    std::vector<TokenTiming>        token_timings;

    SequenceGroup() = default;
    SequenceGroup(SeqId rid,
                  std::vector<TokenId> prompt,
                  SamplingParams sp,
                  int32_t prio = 0)
        : request_id(rid)
        , sampling(std::move(sp))
        , arrival_time(Clock::now())
        , priority(prio)
    {
        SequenceData seq;
        seq.seq_id        = rid;
        seq.prompt_tokens = std::move(prompt);
        seq.status        = RequestStatus::WAITING;
        sequences.push_back(std::move(seq));
    }

    bool all_finished() const {
        for (auto& s : sequences)
            if (!s.is_finished()) return false;
        return true;
    }

    size_t num_unfinished() const {
        size_t n = 0;
        for (auto& s : sequences)
            if (!s.is_finished()) ++n;
        return n;
    }

    size_t prompt_len() const {
        return sequences.empty() ? 0 : sequences[0].prompt_len();
    }

    size_t max_total_len() const {
        size_t m = 0;
        for (auto& s : sequences)
            m = std::max(m, s.total_len());
        return m;
    }
};

// ── Output structure returned to caller ─────────────────────────────────────

struct RequestOutput {
    SeqId                  request_id;
    std::vector<TokenId>   tokens;
    RequestStatus          finish_reason;
    double                 time_to_first_token_ms = 0.0;
    double                 total_time_ms          = 0.0;
    double                 tokens_per_sec         = 0.0;
};

}  // namespace cacheflow
