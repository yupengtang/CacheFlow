set(BENCH_TARGETS
    bench_throughput
    bench_latency
    bench_scaling
    bench_kv_cache
)

foreach(target ${BENCH_TARGETS})
    add_executable(${target} ${target}.cpp)
    target_link_libraries(${target} PRIVATE cacheflow)
    target_include_directories(${target} PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/src
    )
endforeach()
