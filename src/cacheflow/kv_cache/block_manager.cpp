#include "cacheflow/kv_cache/block_manager.h"
#include <algorithm>
#include <cassert>
#include <numeric>

namespace cacheflow {

BlockManager::BlockManager(const CacheConfig& config,
                           const ModelConfig& model_config)
    : config_(config), model_config_(model_config)
{
    init_blocks(gpu_blocks_, gpu_free_list_,
                config.num_gpu_blocks, DeviceType::CUDA);
    init_blocks(cpu_blocks_, cpu_free_list_,
                config.num_cpu_blocks, DeviceType::CPU);
}

BlockManager::~BlockManager() = default;

void BlockManager::init_blocks(std::vector<PhysicalBlock>& blocks,
                               std::vector<BlockId>& free_list,
                               size_t count, DeviceType device) {
    blocks.resize(count);
    free_list.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        blocks[i].block_id  = static_cast<BlockId>(i);
        blocks[i].device    = device;
        blocks[i].ref_count = 0;
        free_list.push_back(static_cast<BlockId>(count - 1 - i));
    }
}

// ── Allocation ──────────────────────────────────────────────────────────────

BlockId BlockManager::allocate_block(SeqId seq_id, DeviceType device) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& fl = free_list_for(device);
    BlockId bid = pop_free(fl);
    if (bid == INVALID_BLOCK) return INVALID_BLOCK;

    auto& blk = blocks_for(device)[static_cast<size_t>(bid)];
    blk.ref_count  = 1;
    blk.num_filled = 0;
    blk.hash       = 0;
    blk.is_prefix  = false;

    seq_block_map_[seq_id].emplace_back(device, bid);
    return bid;
}

void BlockManager::free_block(BlockId block_id, DeviceType device) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& blk = blocks_for(device)[static_cast<size_t>(block_id)];
    if (--blk.ref_count <= 0) {
        blk.ref_count  = 0;
        blk.num_filled = 0;
        blk.hash       = 0;
        blk.is_prefix  = false;
        push_free(free_list_for(device), block_id);
    }
}

void BlockManager::free_blocks(SeqId seq_id) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = seq_block_map_.find(seq_id);
    if (it == seq_block_map_.end()) return;

    for (auto& [dev, bid] : it->second) {
        auto& blk = blocks_for(dev)[static_cast<size_t>(bid)];
        if (--blk.ref_count <= 0) {
            blk.ref_count  = 0;
            blk.num_filled = 0;
            blk.hash       = 0;
            blk.is_prefix  = false;
            push_free(free_list_for(dev), bid);
        }
    }
    seq_block_map_.erase(it);
}

bool BlockManager::can_allocate(size_t num_blocks) const {
    std::lock_guard<std::mutex> lock(mu_);
    return gpu_free_list_.size() >= num_blocks;
}

// ── Queries ─────────────────────────────────────────────────────────────────

size_t BlockManager::num_free_gpu_blocks() const {
    std::lock_guard<std::mutex> lock(mu_);
    return gpu_free_list_.size();
}

size_t BlockManager::num_free_cpu_blocks() const {
    std::lock_guard<std::mutex> lock(mu_);
    return cpu_free_list_.size();
}

size_t BlockManager::num_total_gpu_blocks() const {
    return gpu_blocks_.size();
}

size_t BlockManager::num_total_cpu_blocks() const {
    return cpu_blocks_.size();
}

float BlockManager::gpu_cache_usage() const {
    size_t total = gpu_blocks_.size();
    if (total == 0) return 0.0f;
    return 1.0f - static_cast<float>(num_free_gpu_blocks()) /
                   static_cast<float>(total);
}

float BlockManager::cpu_cache_usage() const {
    size_t total = cpu_blocks_.size();
    if (total == 0) return 0.0f;
    return 1.0f - static_cast<float>(num_free_cpu_blocks()) /
                   static_cast<float>(total);
}

PhysicalBlock* BlockManager::get_block(BlockId id, DeviceType device) {
    auto& blks = blocks_for(device);
    if (id < 0 || static_cast<size_t>(id) >= blks.size()) return nullptr;
    return &blks[static_cast<size_t>(id)];
}

const PhysicalBlock* BlockManager::get_block(BlockId id,
                                             DeviceType device) const {
    auto& blks = blocks_for(device);
    if (id < 0 || static_cast<size_t>(id) >= blks.size()) return nullptr;
    return &blks[static_cast<size_t>(id)];
}

// ── Copy-on-write ───────────────────────────────────────────────────────────

BlockId BlockManager::copy_on_write(BlockId src_block, SeqId new_seq,
                                    DeviceType device) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& src = blocks_for(device)[static_cast<size_t>(src_block)];

    if (src.ref_count == 1) {
        seq_block_map_[new_seq].emplace_back(device, src_block);
        return src_block;
    }

    auto& fl = free_list_for(device);
    BlockId new_bid = pop_free(fl);
    if (new_bid == INVALID_BLOCK) return INVALID_BLOCK;

    auto& dst = blocks_for(device)[static_cast<size_t>(new_bid)];
    dst.ref_count  = 1;
    dst.num_filled = src.num_filled;
    dst.hash       = src.hash;
    dst.is_prefix  = false;

    src.ref_count--;
    seq_block_map_[new_seq].emplace_back(device, new_bid);
    return new_bid;
}

void BlockManager::increment_ref(BlockId block_id, DeviceType device) {
    std::lock_guard<std::mutex> lock(mu_);
    blocks_for(device)[static_cast<size_t>(block_id)].ref_count++;
}

void BlockManager::decrement_ref(BlockId block_id, DeviceType device) {
    std::lock_guard<std::mutex> lock(mu_);
    auto& blk = blocks_for(device)[static_cast<size_t>(block_id)];
    if (--blk.ref_count <= 0) {
        blk.ref_count = 0;
        push_free(free_list_for(device), block_id);
    }
}

// ── Fragmentation analysis ──────────────────────────────────────────────────

BlockManager::FragmentationStats
BlockManager::compute_fragmentation(DeviceType device) const {
    std::lock_guard<std::mutex> lock(mu_);
    const auto& blks = blocks_for(device);
    const auto& fl   = free_list_for(device);

    FragmentationStats stats{};
    stats.total_blocks = blks.size();
    stats.free_blocks  = fl.size();
    stats.used_blocks  = stats.total_blocks - stats.free_blocks;

    size_t partially_filled = 0;
    size_t total_filled     = 0;
    for (auto& b : blks) {
        if (b.ref_count > 0) {
            total_filled += b.num_filled;
            if (b.num_filled < config_.block_size)
                ++partially_filled;
        }
    }

    size_t max_capacity = stats.used_blocks * config_.block_size;
    stats.internal_fragmentation =
        (max_capacity > 0)
            ? 1.0f - static_cast<float>(total_filled) /
                      static_cast<float>(max_capacity)
            : 0.0f;

    std::unordered_set<BlockId> free_set(fl.begin(), fl.end());
    size_t max_run = 0, cur_run = 0;
    for (size_t i = 0; i < blks.size(); ++i) {
        if (free_set.count(static_cast<BlockId>(i))) {
            ++cur_run;
            max_run = std::max(max_run, cur_run);
        } else {
            cur_run = 0;
        }
    }
    stats.largest_free_run = max_run;
    stats.external_fragmentation =
        (stats.free_blocks > 0)
            ? 1.0f - static_cast<float>(max_run) /
                      static_cast<float>(stats.free_blocks)
            : 0.0f;

    return stats;
}

// ── Internal helpers ────────────────────────────────────────────────────────

BlockId BlockManager::pop_free(std::vector<BlockId>& free_list) {
    if (free_list.empty()) return INVALID_BLOCK;
    BlockId id = free_list.back();
    free_list.pop_back();
    return id;
}

void BlockManager::push_free(std::vector<BlockId>& free_list, BlockId id) {
    free_list.push_back(id);
}

std::vector<PhysicalBlock>& BlockManager::blocks_for(DeviceType d) {
    return d == DeviceType::CUDA ? gpu_blocks_ : cpu_blocks_;
}

const std::vector<PhysicalBlock>& BlockManager::blocks_for(
    DeviceType d) const {
    return d == DeviceType::CUDA ? gpu_blocks_ : cpu_blocks_;
}

std::vector<BlockId>& BlockManager::free_list_for(DeviceType d) {
    return d == DeviceType::CUDA ? gpu_free_list_ : cpu_free_list_;
}

const std::vector<BlockId>& BlockManager::free_list_for(
    DeviceType d) const {
    return d == DeviceType::CUDA ? gpu_free_list_ : cpu_free_list_;
}

}  // namespace cacheflow
