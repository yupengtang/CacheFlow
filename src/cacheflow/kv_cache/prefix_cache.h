#pragma once

#include "cacheflow/common.h"
#include <unordered_map>
#include <list>
#include <vector>
#include <mutex>
#include <functional>

namespace cacheflow {

class BlockManager;

// ── Trie node for prefix matching ───────────────────────────────────────────

struct PrefixTrieNode {
    std::unordered_map<TokenId, std::unique_ptr<PrefixTrieNode>> children;
    BlockId           block_id       = INVALID_BLOCK;
    size_t            tokens_in_block = 0;
    uint64_t          last_access    = 0;
    int32_t           ref_count      = 0;
};

// ── Prefix cache: reuses KV blocks for shared prompt prefixes ───────────────

class PrefixCache {
public:
    explicit PrefixCache(const CacheConfig& config,
                         BlockManager* block_manager);
    ~PrefixCache();

    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;

    struct LookupResult {
        std::vector<BlockId> cached_blocks;
        size_t               matched_tokens = 0;
    };

    LookupResult lookup(const std::vector<TokenId>& prefix) const;

    void insert(const std::vector<TokenId>& tokens,
                const std::vector<BlockId>& blocks);

    void release(const std::vector<TokenId>& prefix);

    void evict_lru(size_t num_blocks_to_free);

    size_t num_cached_blocks() const;
    size_t num_cached_prefixes() const;
    float  hit_rate() const;

    void reset_stats();

    struct Stats {
        uint64_t lookups      = 0;
        uint64_t hits         = 0;
        uint64_t misses       = 0;
        uint64_t insertions   = 0;
        uint64_t evictions    = 0;
        size_t   cached_blocks = 0;
    };
    Stats stats() const;

private:
    PrefixTrieNode* walk_trie(const std::vector<TokenId>& tokens,
                              size_t block_boundary,
                              bool create_nodes);
    void collect_eviction_candidates(
        PrefixTrieNode* node,
        std::vector<std::pair<uint64_t, BlockId>>& candidates);

    CacheConfig    config_;
    BlockManager*  block_manager_;
    std::unique_ptr<PrefixTrieNode> root_;

    mutable std::mutex mu_;
    mutable Stats stats_;
    uint64_t access_clock_ = 0;
};

}  // namespace cacheflow
