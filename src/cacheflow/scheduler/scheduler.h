#pragma once

#include "cacheflow/common.h"
#include "cacheflow/scheduler/request.h"
#include <deque>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

namespace cacheflow {

class BlockManager;

// ── Scheduler output: what the decode engine should execute this step ────────

struct SchedulerOutput {
    std::vector<SequenceGroup*> scheduled_groups;
    std::vector<SequenceGroup*> preempted_groups;
    std::vector<SequenceGroup*> swapped_in_groups;
    size_t num_batched_tokens = 0;

    struct BlockOp {
        SeqId   seq_id;
        BlockId src;
        BlockId dst;
    };
    std::vector<BlockOp> block_copies;
    std::vector<BlockOp> block_swaps_in;
    std::vector<BlockOp> block_swaps_out;

    bool is_empty() const {
        return scheduled_groups.empty();
    }

    size_t num_seqs() const {
        size_t n = 0;
        for (auto* g : scheduled_groups)
            n += g->num_unfinished();
        return n;
    }
};

// ── Concurrency-aware scheduler ─────────────────────────────────────────────

class Scheduler {
public:
    explicit Scheduler(const SchedulerConfig& config,
                       BlockManager* block_manager);
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    void add_request(std::unique_ptr<SequenceGroup> group);
    void abort_request(SeqId request_id);
    SchedulerOutput schedule();
    void free_finished_requests();

    size_t num_waiting()  const;
    size_t num_running()  const;
    size_t num_swapped()  const;
    size_t num_total()    const;
    bool   has_pending()  const;

    const SchedulerConfig& config() const { return config_; }

private:
    void schedule_prefills(SchedulerOutput& output, size_t& budget_tokens,
                           size_t& budget_seqs);
    void schedule_decodes(SchedulerOutput& output, size_t& budget_tokens,
                          size_t& budget_seqs);
    bool try_swap_in(SchedulerOutput& output, size_t& budget_tokens,
                     size_t& budget_seqs);
    void preempt(SchedulerOutput& output);

    void sort_waiting_queue();
    size_t estimate_tokens(const SequenceGroup& group, bool is_prefill) const;

    SchedulerConfig config_;
    BlockManager*   block_manager_;

    mutable std::mutex mu_;
    std::deque<std::unique_ptr<SequenceGroup>> waiting_;
    std::deque<std::unique_ptr<SequenceGroup>> running_;
    std::deque<std::unique_ptr<SequenceGroup>> swapped_;
    std::unordered_map<SeqId, SequenceGroup*>  request_map_;
};

}  // namespace cacheflow
