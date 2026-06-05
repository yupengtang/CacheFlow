#include "cacheflow/kv_cache/prefix_cache.h"
#include "cacheflow/kv_cache/block_manager.h"
#include <cassert>
#include <iostream>

using namespace cacheflow;

static void test_insert_and_lookup() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 100;
    ccfg.block_size = 4;
    ModelConfig mcfg;

    BlockManager bm(ccfg, mcfg);
    PrefixCache pc(ccfg, &bm);

    std::vector<TokenId> tokens = {10, 20, 30, 40, 50, 60, 70, 80};
    std::vector<BlockId> blocks;
    for (int i = 0; i < 2; ++i) {
        blocks.push_back(bm.allocate_block(0, DeviceType::CUDA));
    }

    pc.insert(tokens, blocks);

    auto result = pc.lookup(tokens);
    assert(result.matched_tokens == 8);
    assert(result.cached_blocks.size() == 2);

    std::cout << "  PASS: test_insert_and_lookup\n";
}

static void test_partial_match() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 100;
    ccfg.block_size = 4;
    ModelConfig mcfg;

    BlockManager bm(ccfg, mcfg);
    PrefixCache pc(ccfg, &bm);

    std::vector<TokenId> tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    std::vector<BlockId> blocks;
    for (int i = 0; i < 2; ++i)
        blocks.push_back(bm.allocate_block(0, DeviceType::CUDA));
    pc.insert(tokens, blocks);

    std::vector<TokenId> query = {1, 2, 3, 4, 5, 6, 99, 100};
    auto result = pc.lookup(query);
    assert(result.matched_tokens == 4);
    assert(result.cached_blocks.size() == 1);

    std::cout << "  PASS: test_partial_match\n";
}

static void test_no_match() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 100;
    ccfg.block_size = 4;
    ModelConfig mcfg;

    BlockManager bm(ccfg, mcfg);
    PrefixCache pc(ccfg, &bm);

    std::vector<TokenId> tokens = {1, 2, 3, 4};
    std::vector<BlockId> blocks = {
        bm.allocate_block(0, DeviceType::CUDA)};
    pc.insert(tokens, blocks);

    std::vector<TokenId> query = {99, 98, 97, 96};
    auto result = pc.lookup(query);
    assert(result.matched_tokens == 0);
    assert(result.cached_blocks.empty());

    std::cout << "  PASS: test_no_match\n";
}

static void test_hit_rate() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 100;
    ccfg.block_size = 4;
    ModelConfig mcfg;

    BlockManager bm(ccfg, mcfg);
    PrefixCache pc(ccfg, &bm);

    std::vector<TokenId> tokens = {1, 2, 3, 4};
    std::vector<BlockId> blocks = {
        bm.allocate_block(0, DeviceType::CUDA)};
    pc.insert(tokens, blocks);

    pc.lookup(tokens);
    pc.lookup(tokens);
    pc.lookup({99, 98, 97, 96});

    auto stats = pc.stats();
    assert(stats.lookups == 3);
    assert(stats.hits == 2);
    assert(stats.misses == 1);
    assert(pc.hit_rate() > 0.6f);

    std::cout << "  PASS: test_hit_rate\n";
}

static void test_stats_reset() {
    CacheConfig ccfg;
    ccfg.num_gpu_blocks = 100;
    ccfg.block_size = 4;
    ModelConfig mcfg;

    BlockManager bm(ccfg, mcfg);
    PrefixCache pc(ccfg, &bm);

    std::vector<TokenId> t = {1, 2, 3, 4};
    std::vector<BlockId> b = {bm.allocate_block(0, DeviceType::CUDA)};
    pc.insert(t, b);
    pc.lookup(t);

    pc.reset_stats();
    auto stats = pc.stats();
    assert(stats.lookups == 0);
    assert(stats.hits == 0);

    std::cout << "  PASS: test_stats_reset\n";
}

int main() {
    std::cout << "=== Prefix Cache Tests ===\n";
    test_insert_and_lookup();
    test_partial_match();
    test_no_match();
    test_hit_rate();
    test_stats_reset();
    std::cout << "All prefix cache tests passed.\n";
    return 0;
}
