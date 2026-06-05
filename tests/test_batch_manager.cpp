#include "cacheflow/scheduler/batch_manager.h"
#include "cacheflow/scheduler/scheduler.h"
#include "cacheflow/kv_cache/block_manager.h"
#include <cassert>
#include <iostream>

using namespace cacheflow;

static void test_build_batch_prefill() {
    ModelConfig mcfg;
    SchedulerConfig scfg;
    scfg.max_num_batched_tokens = 4096;
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 1024;

    BlockManager bm(ccfg, mcfg);
    Scheduler sched(scfg, &bm);
    BatchManager batch_mgr(mcfg, scfg);

    std::vector<TokenId> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    SamplingParams sp;
    sp.max_tokens = 4;
    auto grp = std::make_unique<SequenceGroup>(1, prompt, sp);
    sched.add_request(std::move(grp));

    auto out = sched.schedule();
    auto batch = batch_mgr.build_batch(out);

    assert(!batch.empty());
    assert(batch.total_tokens == 8);
    assert(batch.entries[0].is_prefill);

    [[maybe_unused]] auto stats = batch_mgr.last_stats();
    assert(stats.num_prefill_tokens == 8);
    assert(stats.num_decode_tokens == 0);

    std::cout << "  PASS: test_build_batch_prefill\n";
}

static void test_build_batch_decode() {
    ModelConfig mcfg;
    SchedulerConfig scfg;
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 1024;

    BlockManager bm(ccfg, mcfg);
    Scheduler sched(scfg, &bm);
    BatchManager batch_mgr(mcfg, scfg);

    std::vector<TokenId> prompt = {1, 2, 3, 4};
    SamplingParams sp;
    sp.max_tokens = 4;
    auto grp = std::make_unique<SequenceGroup>(1, prompt, sp);
    sched.add_request(std::move(grp));

    auto out = sched.schedule();
    auto batch = batch_mgr.build_batch(out);
    std::vector<TokenId> new_tokens = {100};
    batch_mgr.update_after_decode(out, new_tokens);

    auto out2 = sched.schedule();
    auto batch2 = batch_mgr.build_batch(out2);

    assert(!batch2.empty());
    [[maybe_unused]] auto stats = batch_mgr.last_stats();
    assert(stats.num_decode_tokens == 1);

    std::cout << "  PASS: test_build_batch_decode\n";
}

static void test_update_appends_token() {
    ModelConfig mcfg;
    SchedulerConfig scfg;
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 1024;

    BlockManager bm(ccfg, mcfg);
    Scheduler sched(scfg, &bm);
    BatchManager batch_mgr(mcfg, scfg);

    std::vector<TokenId> prompt = {1, 2, 3, 4};
    SamplingParams sp;
    sp.max_tokens = 10;
    auto grp = std::make_unique<SequenceGroup>(1, prompt, sp);
    sched.add_request(std::move(grp));

    auto out = sched.schedule();
    batch_mgr.build_batch(out);
    batch_mgr.update_after_decode(out, {42});

    [[maybe_unused]] auto* seq = &out.scheduled_groups[0]->sequences[0];
    assert(seq->output_len() == 1);
    assert(seq->output_tokens[0] == 42);

    std::cout << "  PASS: test_update_appends_token\n";
}

static void test_multi_sequence_batch() {
    ModelConfig mcfg;
    SchedulerConfig scfg;
    scfg.max_num_seqs = 4;
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 2048;

    BlockManager bm(ccfg, mcfg);
    Scheduler sched(scfg, &bm);
    BatchManager batch_mgr(mcfg, scfg);

    for (int i = 0; i < 3; ++i) {
        std::vector<TokenId> prompt(16, static_cast<TokenId>(i + 1));
        SamplingParams sp;
        sp.max_tokens = 8;
        sched.add_request(std::make_unique<SequenceGroup>(
            static_cast<SeqId>(i), prompt, sp));
    }

    auto out = sched.schedule();
    auto batch = batch_mgr.build_batch(out);

    [[maybe_unused]] auto stats = batch_mgr.last_stats();
    assert(stats.num_sequences == 3);
    assert(stats.num_prefill_tokens == 48);

    std::cout << "  PASS: test_multi_sequence_batch\n";
}

int main() {
    std::cout << "=== Batch Manager Tests ===\n";
    test_build_batch_prefill();
    test_build_batch_decode();
    test_update_appends_token();
    test_multi_sequence_batch();
    std::cout << "All batch manager tests passed.\n";
    return 0;
}
