#pragma once

#include "cacheflow/common.h"
#include "cacheflow/kv_cache/memory_pool.h"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <mutex>

namespace cacheflow {

// ── Physical block descriptor ───────────────────────────────────────────────

struct PhysicalBlock {
    BlockId    block_id    = INVALID_BLOCK;
    DeviceType device      = DeviceType::CUDA;
    int32_t    ref_count   = 0;
    size_t     num_filled  = 0;
    uint64_t   hash        = 0;
    bool       is_prefix   = false;
};

// ── Block table: per-sequence mapping of logical → physical blocks ──────────

struct BlockTable {
    SeqId                  seq_id;
    std::vector<BlockId>   physical_blocks;

    size_t num_blocks() const { return physical_blocks.size(); }
};

// ── Block manager: handles allocation, deallocation, COW, and swap ──────────

class BlockManager {
public:
    BlockManager(const CacheConfig& config, const ModelConfig& model_config);
    ~BlockManager();

    BlockManager(const BlockManager&) = delete;
    BlockManager& operator=(const BlockManager&) = delete;

    BlockId allocate_block(SeqId seq_id, DeviceType device);
    void    free_block(BlockId block_id, DeviceType device);
    void    free_blocks(SeqId seq_id);
    bool    can_allocate(size_t num_blocks) const;

    size_t num_free_gpu_blocks()  const;
    size_t num_free_cpu_blocks()  const;
    size_t num_total_gpu_blocks() const;
    size_t num_total_cpu_blocks() const;
    float  gpu_cache_usage()      const;
    float  cpu_cache_usage()      const;

    PhysicalBlock* get_block(BlockId id, DeviceType device);
    const PhysicalBlock* get_block(BlockId id, DeviceType device) const;

    BlockId copy_on_write(BlockId src_block, SeqId new_seq, DeviceType device);

    void increment_ref(BlockId block_id, DeviceType device);
    void decrement_ref(BlockId block_id, DeviceType device);

    size_t block_size() const { return config_.block_size; }
    size_t block_bytes() const { return model_config_.block_bytes(); }

    struct FragmentationStats {
        size_t total_blocks;
        size_t used_blocks;
        size_t free_blocks;
        float  internal_fragmentation;
        float  external_fragmentation;
        size_t largest_free_run;
    };
    FragmentationStats compute_fragmentation(DeviceType device) const;

private:
    CacheConfig  config_;
    ModelConfig  model_config_;

    mutable std::mutex mu_;

    std::vector<PhysicalBlock> gpu_blocks_;
    std::vector<PhysicalBlock> cpu_blocks_;
    std::vector<BlockId>       gpu_free_list_;
    std::vector<BlockId>       cpu_free_list_;

    std::unordered_map<SeqId, std::vector<std::pair<DeviceType, BlockId>>>
        seq_block_map_;

    void init_blocks(std::vector<PhysicalBlock>& blocks,
                     std::vector<BlockId>& free_list,
                     size_t count, DeviceType device);
    BlockId pop_free(std::vector<BlockId>& free_list);
    void    push_free(std::vector<BlockId>& free_list, BlockId id);
    std::vector<PhysicalBlock>& blocks_for(DeviceType device);
    const std::vector<PhysicalBlock>& blocks_for(DeviceType device) const;
    std::vector<BlockId>& free_list_for(DeviceType device);
    const std::vector<BlockId>& free_list_for(DeviceType device) const;
};

}  // namespace cacheflow
