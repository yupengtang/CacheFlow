#include "cacheflow/kv_cache/memory_pool.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <set>

using namespace cacheflow;

static void test_basic_pool() {
    MemoryPool pool(4096, 10, DeviceType::CPU);
    assert(pool.num_free() == 10);
    assert(pool.num_used() == 0);

    void* p1 = pool.allocate();
    assert(p1 != nullptr);
    assert(pool.num_free() == 9);
    assert(pool.num_used() == 1);

    pool.deallocate(p1);
    assert(pool.num_free() == 10);

    std::cout << "  PASS: test_basic_pool\n";
}

static void test_exhaust_pool() {
    MemoryPool pool(1024, 5, DeviceType::CPU);

    std::vector<void*> ptrs;
    for (int i = 0; i < 5; ++i) {
        void* p = pool.allocate();
        assert(p != nullptr);
        ptrs.push_back(p);
    }

    assert(pool.num_free() == 0);
    void* p = pool.allocate();
    assert(p == nullptr);

    for (auto* ptr : ptrs)
        pool.deallocate(ptr);
    assert(pool.num_free() == 5);

    std::cout << "  PASS: test_exhaust_pool\n";
}

static void test_unique_pointers() {
    MemoryPool pool(256, 20, DeviceType::CPU);

    std::set<void*> seen;
    for (int i = 0; i < 20; ++i) {
        void* p = pool.allocate();
        assert(p != nullptr);
        assert(seen.find(p) == seen.end());
        seen.insert(p);
    }

    std::cout << "  PASS: test_unique_pointers\n";
}

static void test_block_id_conversion() {
    MemoryPool pool(512, 10, DeviceType::CPU);

    void* p0 = pool.allocate();
    BlockId id = pool.ptr_to_block_id(p0);
    void* p_back = pool.block_id_to_ptr(id);
    assert(p0 == p_back);

    pool.deallocate(p0);

    std::cout << "  PASS: test_block_id_conversion\n";
}

static void test_utilization() {
    MemoryPool pool(256, 10, DeviceType::CPU);
    assert(pool.utilization() == 0.0f);

    for (int i = 0; i < 5; ++i) pool.allocate();
    float util = pool.utilization();
    assert(util > 0.49f && util < 0.51f);

    std::cout << "  PASS: test_utilization\n";
}

static void test_reset() {
    MemoryPool pool(256, 10, DeviceType::CPU);
    for (int i = 0; i < 10; ++i) pool.allocate();
    assert(pool.num_free() == 0);

    pool.reset();
    assert(pool.num_free() == 10);

    void* p = pool.allocate();
    assert(p != nullptr);

    std::cout << "  PASS: test_reset\n";
}

static void test_pool_manager() {
    PoolManager pm(1024, 10, 20);
    assert(pm.gpu_pool().num_free() == 10);
    assert(pm.cpu_pool().num_free() == 20);

    auto& gpu = pm.pool_for(DeviceType::CUDA);
    auto& cpu = pm.pool_for(DeviceType::CPU);
    assert(gpu.num_free() == 10);
    assert(cpu.num_free() == 20);

    std::cout << "  PASS: test_pool_manager\n";
}

int main() {
    std::cout << "=== Memory Pool Tests ===\n";
    test_basic_pool();
    test_exhaust_pool();
    test_unique_pointers();
    test_block_id_conversion();
    test_utilization();
    test_reset();
    test_pool_manager();
    std::cout << "All memory pool tests passed.\n";
    return 0;
}
