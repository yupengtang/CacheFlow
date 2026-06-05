set(TEST_TARGETS
    test_scheduler
    test_block_manager
    test_memory_pool
    test_prefix_cache
    test_batch_manager
    test_profiler
)

foreach(target ${TEST_TARGETS})
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} PRIVATE cacheflow)
    target_include_directories(${target} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    add_test(NAME ${target} COMMAND ${target})
endforeach()
