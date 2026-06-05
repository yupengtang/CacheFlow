#include "cacheflow/kv_cache/prefix_cache.h"
#include "cacheflow/kv_cache/block_manager.h"
#include <algorithm>
#include <cassert>

namespace cacheflow {

PrefixCache::PrefixCache(const CacheConfig& config,
                         BlockManager* block_manager)
    : config_(config)
    , block_manager_(block_manager)
    , root_(std::make_unique<PrefixTrieNode>())
{}

PrefixCache::~PrefixCache() = default;

// ── Lookup: find cached KV blocks for a token prefix ────────────────────────

PrefixCache::LookupResult
PrefixCache::lookup(const std::vector<TokenId>& prefix) const {
    std::lock_guard<std::mutex> lock(mu_);
    ++stats_.lookups;

    LookupResult result;
    size_t block_size = config_.block_size;
    PrefixTrieNode* node = root_.get();

    for (size_t i = 0; i < prefix.size(); ++i) {
        auto it = node->children.find(prefix[i]);
        if (it == node->children.end()) break;

        node = it->second.get();

        if ((i + 1) % block_size == 0 && node->block_id != INVALID_BLOCK) {
            result.cached_blocks.push_back(node->block_id);
            result.matched_tokens = i + 1;
            node->last_access = ++const_cast<PrefixCache*>(this)->access_clock_;
        }
    }

    if (!result.cached_blocks.empty()) {
        ++stats_.hits;
    } else {
        ++stats_.misses;
    }

    return result;
}

// ── Insert: store KV blocks for a token sequence ────────────────────────────

void PrefixCache::insert(const std::vector<TokenId>& tokens,
                         const std::vector<BlockId>& blocks) {
    std::lock_guard<std::mutex> lock(mu_);
    size_t block_size = config_.block_size;
    PrefixTrieNode* node = root_.get();

    size_t block_idx = 0;
    for (size_t i = 0; i < tokens.size() && block_idx < blocks.size(); ++i) {
        auto& child = node->children[tokens[i]];
        if (!child) {
            child = std::make_unique<PrefixTrieNode>();
        }
        node = child.get();

        if ((i + 1) % block_size == 0) {
            if (node->block_id == INVALID_BLOCK) {
                node->block_id = blocks[block_idx];
                node->tokens_in_block = block_size;
                node->ref_count = 1;
                block_manager_->increment_ref(blocks[block_idx],
                                              DeviceType::CUDA);
                ++stats_.insertions;
                ++stats_.cached_blocks;
            }
            node->last_access = ++access_clock_;
            ++block_idx;
        }
    }
}

// ── Release: decrement reference counts ─────────────────────────────────────

void PrefixCache::release(const std::vector<TokenId>& prefix) {
    std::lock_guard<std::mutex> lock(mu_);
    size_t block_size = config_.block_size;
    PrefixTrieNode* node = root_.get();

    for (size_t i = 0; i < prefix.size(); ++i) {
        auto it = node->children.find(prefix[i]);
        if (it == node->children.end()) return;
        node = it->second.get();

        if ((i + 1) % block_size == 0 && node->block_id != INVALID_BLOCK) {
            --node->ref_count;
        }
    }
}

// ── Eviction: LRU-based eviction of unreferenced prefix blocks ──────────────

void PrefixCache::evict_lru(size_t num_blocks_to_free) {
    std::lock_guard<std::mutex> lock(mu_);

    std::vector<std::pair<uint64_t, BlockId>> candidates;
    collect_eviction_candidates(root_.get(), candidates);

    std::sort(candidates.begin(), candidates.end());

    size_t freed = 0;
    for (auto& [ts, bid] : candidates) {
        if (freed >= num_blocks_to_free) break;
        block_manager_->decrement_ref(bid, DeviceType::CUDA);
        ++stats_.evictions;
        --stats_.cached_blocks;
        ++freed;
    }
}

void PrefixCache::collect_eviction_candidates(
    PrefixTrieNode* node,
    std::vector<std::pair<uint64_t, BlockId>>& candidates) {
    if (node->block_id != INVALID_BLOCK && node->ref_count <= 0) {
        candidates.emplace_back(node->last_access, node->block_id);
    }
    for (auto& [tok, child] : node->children) {
        collect_eviction_candidates(child.get(), candidates);
    }
}

// ── Stats ───────────────────────────────────────────────────────────────────

size_t PrefixCache::num_cached_blocks() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_.cached_blocks;
}

size_t PrefixCache::num_cached_prefixes() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_.insertions;
}

float PrefixCache::hit_rate() const {
    std::lock_guard<std::mutex> lock(mu_);
    if (stats_.lookups == 0) return 0.0f;
    return static_cast<float>(stats_.hits) /
           static_cast<float>(stats_.lookups);
}

void PrefixCache::reset_stats() {
    std::lock_guard<std::mutex> lock(mu_);
    stats_ = {};
}

PrefixCache::Stats PrefixCache::stats() const {
    std::lock_guard<std::mutex> lock(mu_);
    return stats_;
}

PrefixTrieNode* PrefixCache::walk_trie(
    const std::vector<TokenId>& tokens,
    size_t block_boundary, bool create_nodes) {
    PrefixTrieNode* node = root_.get();
    for (size_t i = 0; i < std::min(tokens.size(), block_boundary); ++i) {
        auto it = node->children.find(tokens[i]);
        if (it == node->children.end()) {
            if (!create_nodes) return nullptr;
            node->children[tokens[i]] = std::make_unique<PrefixTrieNode>();
            it = node->children.find(tokens[i]);
        }
        node = it->second.get();
    }
    return node;
}

}  // namespace cacheflow
