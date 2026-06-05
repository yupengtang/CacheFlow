#include "cacheflow/kv_cache/block_manager.h"
#include <cassert>
#include <iostream>

using namespace cacheflow;

static void test_basic_alloc_free() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 100;
    ccfg.num_cpu_blocks = 50;
    ModelConfig mcfg;

    BlockManager bm(ccfg, mcfg);
    assert(bm.num_free_gpu_blocks() == 100);
    assert(bm.num_free_cpu_blocks() == 50);

    BlockId b1 = bm.allocate_block(1, DeviceType::CUDA);
    assert(b1 != INVALID_BLOCK);
    assert(bm.num_free_gpu_blocks() == 99);

    [[maybe_unused]] BlockId b2 = bm.allocate_block(1, DeviceType::CUDA);
    assert(b2 != INVALID_BLOCK);
    assert(b2 != b1);
    assert(bm.num_free_gpu_blocks() == 98);

    bm.free_block(b1, DeviceType::CUDA);
    assert(bm.num_free_gpu_blocks() == 99);

    bm.free_blocks(1);
    assert(bm.num_free_gpu_blocks() == 100);

    std::cout << "  PASS: test_basic_alloc_free\n";
}

static void test_can_allocate() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 5;
    ModelConfig mcfg;

    BlockManager bm(ccfg, mcfg);
    assert(bm.can_allocate(5));
    assert(!bm.can_allocate(6));

    for (int i = 0; i < 5; ++i)
        bm.allocate_block(static_cast<SeqId>(i), DeviceType::CUDA);

    assert(!bm.can_allocate(1));

    bm.free_blocks(0);
    assert(bm.can_allocate(1));

    std::cout << "  PASS: test_can_allocate\n";
}

static void test_copy_on_write() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 100;
    ModelConfig mcfg;

    BlockManager bm(ccfg, mcfg);

    BlockId b1 = bm.allocate_block(1, DeviceType::CUDA);
    bm.increment_ref(b1, DeviceType::CUDA);

    [[maybe_unused]] auto* blk = bm.get_block(b1, DeviceType::CUDA);
    assert(blk->ref_count == 2);

    [[maybe_unused]] BlockId b2 = bm.copy_on_write(b1, 2, DeviceType::CUDA);
    assert(b2 != INVALID_BLOCK);
    assert(b2 != b1);

    blk = bm.get_block(b1, DeviceType::CUDA);
    assert(blk->ref_count == 1);

    [[maybe_unused]] auto* blk2 = bm.get_block(b2, DeviceType::CUDA);
    assert(blk2->ref_count == 1);

    std::cout << "  PASS: test_copy_on_write\n";
}

static void test_cow_single_ref() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 100;
    ModelConfig mcfg;

    BlockManager bm(ccfg, mcfg);

    BlockId b1 = bm.allocate_block(1, DeviceType::CUDA);
    [[maybe_unused]] BlockId b2 = bm.copy_on_write(b1, 2, DeviceType::CUDA);
    assert(b2 == b1);

    std::cout << "  PASS: test_cow_single_ref\n";
}

static void test_fragmentation_stats() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 20;
    ModelConfig mcfg;

    BlockManager bm(ccfg, mcfg);

    for (int i = 0; i < 20; ++i)
        bm.allocate_block(static_cast<SeqId>(i), DeviceType::CUDA);

    for (int i = 0; i < 20; i += 2)
        bm.free_blocks(static_cast<SeqId>(i));

    [[maybe_unused]] auto frag = bm.compute_fragmentation(DeviceType::CUDA);
    assert(frag.total_blocks == 20);
    assert(frag.free_blocks == 10);
    assert(frag.used_blocks == 10);
    assert(frag.external_fragmentation > 0.0f);

    std::cout << "  PASS: test_fragmentation_stats\n";
}

static void test_cpu_blocks() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 10;
    ccfg.num_cpu_blocks = 20;
    ModelConfig mcfg;

    BlockManager bm(ccfg, mcfg);

    BlockId cb = bm.allocate_block(1, DeviceType::CPU);
    assert(cb != INVALID_BLOCK);
    assert(bm.num_free_cpu_blocks() == 19);

    bm.free_block(cb, DeviceType::CPU);
    assert(bm.num_free_cpu_blocks() == 20);

    std::cout << "  PASS: test_cpu_blocks\n";
}

int main() {
    std::cout << "=== Block Manager Tests ===\n";
    test_basic_alloc_free();
    test_can_allocate();
    test_copy_on_write();
    test_cow_single_ref();
    test_fragmentation_stats();
    test_cpu_blocks();
    std::cout << "All block manager tests passed.\n";
    return 0;
}
