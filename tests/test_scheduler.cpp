#include "cacheflow/scheduler/scheduler.h"
#include "cacheflow/scheduler/batch_manager.h"
#include "cacheflow/kv_cache/block_manager.h"
#include <cassert>
#include <iostream>

using namespace cacheflow;

static void test_add_and_schedule() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 1024;
    ccfg.num_cpu_blocks = 512;
    ModelConfig mcfg;
    SchedulerConfig scfg;
    scfg.max_num_seqs = 4;
    scfg.max_num_batched_tokens = 2048;

    BlockManager bm(ccfg, mcfg);
    Scheduler sched(scfg, &bm);

    assert(sched.num_total() == 0);

    std::vector<TokenId> prompt = {1, 2, 3, 4, 5, 6, 7, 8};
    SamplingParams sp;
    sp.max_tokens = 10;

    auto grp = std::make_unique<SequenceGroup>(1, prompt, sp);
    sched.add_request(std::move(grp));
    assert(sched.num_waiting() == 1);
    assert(sched.num_running() == 0);

    auto out = sched.schedule();
    assert(!out.is_empty());
    assert(out.scheduled_groups.size() == 1);
    assert(sched.num_running() == 1);
    assert(sched.num_waiting() == 0);

    std::cout << "  PASS: test_add_and_schedule\n";
}

static void test_multiple_requests() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 2048;
    ModelConfig mcfg;
    SchedulerConfig scfg;
    scfg.max_num_seqs = 8;
    scfg.max_num_batched_tokens = 4096;

    BlockManager bm(ccfg, mcfg);
    Scheduler sched(scfg, &bm);

    for (int i = 0; i < 5; ++i) {
        std::vector<TokenId> prompt(32, static_cast<TokenId>(i + 1));
        SamplingParams sp;
        sp.max_tokens = 16;
        auto grp = std::make_unique<SequenceGroup>(
            static_cast<SeqId>(i), prompt, sp);
        sched.add_request(std::move(grp));
    }

    assert(sched.num_waiting() == 5);

    auto out = sched.schedule();
    assert(out.scheduled_groups.size() == 5);
    assert(sched.num_running() == 5);

    std::cout << "  PASS: test_multiple_requests\n";
}

static void test_priority_scheduling() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 2048;
    ModelConfig mcfg;
    SchedulerConfig scfg;
    scfg.max_num_seqs = 1;
    scfg.max_num_batched_tokens = 4096;
    scfg.policy = SchedulePolicy::PRIORITY;

    BlockManager bm(ccfg, mcfg);
    Scheduler sched(scfg, &bm);

    for (int i = 0; i < 3; ++i) {
        std::vector<TokenId> prompt(16, 1);
        SamplingParams sp;
        sp.max_tokens = 8;
        auto grp = std::make_unique<SequenceGroup>(
            static_cast<SeqId>(i), prompt, sp, i);
        sched.add_request(std::move(grp));
    }

    auto out = sched.schedule();
    assert(out.scheduled_groups.size() == 1);
    assert(out.scheduled_groups[0]->priority == 2);

    std::cout << "  PASS: test_priority_scheduling\n";
}

static void test_abort_request() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 1024;
    ModelConfig mcfg;
    SchedulerConfig scfg;

    BlockManager bm(ccfg, mcfg);
    Scheduler sched(scfg, &bm);

    std::vector<TokenId> prompt(16, 1);
    SamplingParams sp;
    auto grp = std::make_unique<SequenceGroup>(42, prompt, sp);
    sched.add_request(std::move(grp));

    sched.schedule();
    assert(sched.num_running() == 1);

    sched.abort_request(42);
    sched.free_finished_requests();

    std::cout << "  PASS: test_abort_request\n";
}

int main() {
    std::cout << "=== Scheduler Tests ===\n";
    test_add_and_schedule();
    test_multiple_requests();
    test_priority_scheduling();
    test_abort_request();
    std::cout << "All scheduler tests passed.\n";
    return 0;
}
