#include "cacheflow/scheduler/scheduler.h"
#include "cacheflow/kv_cache/block_manager.h"
#include <algorithm>
#include <cassert>

namespace cacheflow {

Scheduler::Scheduler(const SchedulerConfig& config, BlockManager* bm)
    : config_(config), block_manager_(bm) {}

Scheduler::~Scheduler() = default;

// ── Public API ──────────────────────────────────────────────────────────────

void Scheduler::add_request(std::unique_ptr<SequenceGroup> group) {
    std::lock_guard<std::mutex> lock(mu_);
    SeqId rid = group->request_id;
    request_map_[rid] = group.get();
    waiting_.push_back(std::move(group));
}

void Scheduler::abort_request(SeqId request_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = request_map_.find(request_id);
    if (it == request_map_.end()) return;

    SequenceGroup* grp = it->second;
    for (auto& seq : grp->sequences) {
        if (!seq.is_finished()) {
            seq.status = RequestStatus::FINISHED_ABORT;
            block_manager_->free_blocks(seq.seq_id);
        }
    }
    request_map_.erase(it);
}

SchedulerOutput Scheduler::schedule() {
    std::lock_guard<std::mutex> lock(mu_);
    SchedulerOutput output;

    size_t budget_tokens = config_.max_num_batched_tokens;
    size_t budget_seqs   = config_.max_num_seqs;

    schedule_decodes(output, budget_tokens, budget_seqs);

    if (!waiting_.empty()) {
        sort_waiting_queue();
        schedule_prefills(output, budget_tokens, budget_seqs);
    }

    if (!swapped_.empty() && budget_tokens > 0 && budget_seqs > 0) {
        try_swap_in(output, budget_tokens, budget_seqs);
    }

    output.num_batched_tokens =
        config_.max_num_batched_tokens - budget_tokens;
    return output;
}

void Scheduler::free_finished_requests() {
    std::lock_guard<std::mutex> lock(mu_);
    auto remove_finished = [&](std::deque<std::unique_ptr<SequenceGroup>>& q) {
        auto it = q.begin();
        while (it != q.end()) {
            if ((*it)->all_finished()) {
                for (auto& seq : (*it)->sequences)
                    block_manager_->free_blocks(seq.seq_id);
                request_map_.erase((*it)->request_id);
                it = q.erase(it);
            } else {
                ++it;
            }
        }
    };
    remove_finished(running_);
    remove_finished(waiting_);
}

size_t Scheduler::num_waiting() const {
    std::lock_guard<std::mutex> lock(mu_);
    return waiting_.size();
}

size_t Scheduler::num_running() const {
    std::lock_guard<std::mutex> lock(mu_);
    return running_.size();
}

size_t Scheduler::num_swapped() const {
    std::lock_guard<std::mutex> lock(mu_);
    return swapped_.size();
}

size_t Scheduler::num_total() const {
    std::lock_guard<std::mutex> lock(mu_);
    return waiting_.size() + running_.size() + swapped_.size();
}

bool Scheduler::has_pending() const {
    std::lock_guard<std::mutex> lock(mu_);
    return !waiting_.empty() || !running_.empty() || !swapped_.empty();
}

// ── Scheduling phases ───────────────────────────────────────────────────────

void Scheduler::schedule_decodes(SchedulerOutput& output,
                                 size_t& budget_tokens,
                                 size_t& budget_seqs) {
    auto it = running_.begin();
    while (it != running_.end()) {
        SequenceGroup* grp = it->get();
        if (grp->all_finished()) {
            ++it;
            continue;
        }

        size_t tokens_needed = estimate_tokens(*grp, false);

        if (tokens_needed > budget_tokens || budget_seqs == 0) {
            preempt(output);
            break;
        }

        bool can_append = true;
        for (auto& seq : grp->sequences) {
            if (seq.is_finished()) continue;
            size_t blocks_needed = seq.num_blocks_needed(
                block_manager_->block_size());
            size_t blocks_have = seq.logical_blocks.size();
            if (blocks_needed > blocks_have) {
                if (!block_manager_->can_allocate(blocks_needed - blocks_have)) {
                    can_append = false;
                    break;
                }
            }
        }

        if (!can_append) {
            preempt(output);
            break;
        }

        for (auto& seq : grp->sequences) {
            if (seq.is_finished()) continue;
            size_t blocks_needed = seq.num_blocks_needed(
                block_manager_->block_size());
            while (seq.logical_blocks.size() < blocks_needed) {
                BlockId bid = block_manager_->allocate_block(
                    seq.seq_id, DeviceType::CUDA);
                seq.logical_blocks.push_back(bid);
            }
        }

        output.scheduled_groups.push_back(grp);
        budget_tokens -= tokens_needed;
        budget_seqs   -= 1;
        ++it;
    }
}

void Scheduler::schedule_prefills(SchedulerOutput& output,
                                  size_t& budget_tokens,
                                  size_t& budget_seqs) {
    while (!waiting_.empty() && budget_seqs > 0) {
        SequenceGroup* grp = waiting_.front().get();
        size_t tokens_needed = estimate_tokens(*grp, true);

        if (tokens_needed > budget_tokens)
            break;

        size_t blocks_needed = grp->sequences[0].num_blocks_needed(
            block_manager_->block_size());

        if (!block_manager_->can_allocate(blocks_needed))
            break;

        for (auto& seq : grp->sequences) {
            seq.status = RequestStatus::RUNNING;
            for (size_t i = 0; i < blocks_needed; ++i) {
                BlockId bid = block_manager_->allocate_block(
                    seq.seq_id, DeviceType::CUDA);
                seq.logical_blocks.push_back(bid);
            }
        }

        output.scheduled_groups.push_back(grp);
        budget_tokens -= tokens_needed;
        budget_seqs   -= 1;

        running_.push_back(std::move(waiting_.front()));
        waiting_.pop_front();
    }
}

bool Scheduler::try_swap_in(SchedulerOutput& output,
                            size_t& budget_tokens,
                            size_t& budget_seqs) {
    if (swapped_.empty()) return false;

    auto it = swapped_.begin();
    while (it != swapped_.end() && budget_seqs > 0) {
        SequenceGroup* grp = it->get();
        size_t tokens_needed = estimate_tokens(*grp, false);

        if (tokens_needed > budget_tokens)
            break;

        size_t total_blocks = 0;
        for (auto& seq : grp->sequences) {
            if (!seq.is_finished())
                total_blocks += seq.logical_blocks.size();
        }

        if (!block_manager_->can_allocate(total_blocks))
            break;

        for (auto& seq : grp->sequences) {
            if (seq.is_finished()) continue;
            seq.status = RequestStatus::RUNNING;
            for (auto& bid : seq.logical_blocks) {
                BlockId new_bid = block_manager_->allocate_block(
                    seq.seq_id, DeviceType::CUDA);
                output.block_swaps_in.push_back({seq.seq_id, bid, new_bid});
                bid = new_bid;
            }
        }

        output.swapped_in_groups.push_back(grp);
        budget_tokens -= tokens_needed;
        budget_seqs   -= 1;

        running_.push_back(std::move(*it));
        it = swapped_.erase(it);
    }

    return !output.swapped_in_groups.empty();
}

void Scheduler::preempt(SchedulerOutput& output) {
    if (running_.empty()) return;

    auto victim_it = running_.end() - 1;
    SequenceGroup* victim = victim_it->get();

    if (config_.preemption_mode == PreemptionMode::SWAP) {
        for (auto& seq : victim->sequences) {
            if (seq.is_finished()) continue;
            seq.status = RequestStatus::SWAPPED;
            for (auto& bid : seq.logical_blocks) {
                BlockId cpu_bid = block_manager_->allocate_block(
                    seq.seq_id, DeviceType::CPU);
                output.block_swaps_out.push_back({seq.seq_id, bid, cpu_bid});
                block_manager_->free_block(bid, DeviceType::CUDA);
                bid = cpu_bid;
            }
        }
        swapped_.push_back(std::move(*victim_it));
    } else {
        for (auto& seq : victim->sequences) {
            if (seq.is_finished()) continue;
            seq.status = RequestStatus::PREEMPTED;
            block_manager_->free_blocks(seq.seq_id);
            seq.logical_blocks.clear();
            seq.output_tokens.clear();
        }
        waiting_.push_front(std::move(*victim_it));
    }

    output.preempted_groups.push_back(victim);
    running_.erase(victim_it);
}

// ── Helpers ─────────────────────────────────────────────────────────────────

void Scheduler::sort_waiting_queue() {
    switch (config_.policy) {
    case SchedulePolicy::FCFS:
        break;
    case SchedulePolicy::SHORTEST_REMAINING: {
        std::stable_sort(waiting_.begin(), waiting_.end(),
            [](const auto& a, const auto& b) {
                return a->sampling.max_tokens < b->sampling.max_tokens;
            });
        break;
    }
    case SchedulePolicy::PRIORITY: {
        std::stable_sort(waiting_.begin(), waiting_.end(),
            [](const auto& a, const auto& b) {
                return a->priority > b->priority;
            });
        break;
    }
    }
}

size_t Scheduler::estimate_tokens(const SequenceGroup& group,
                                  bool is_prefill) const {
    if (is_prefill) {
        return group.prompt_len();
    }
    return group.num_unfinished();
}

}  // namespace cacheflow
