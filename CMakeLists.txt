cmake_minimum_required(VERSION 3.18)
project(CacheFlow LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(CACHEFLOW_CUDA    "Enable CUDA support"         ON)
option(CACHEFLOW_TESTS   "Build unit tests"             ON)
option(CACHEFLOW_BENCH   "Build benchmarks"             ON)
option(CACHEFLOW_PROFILE "Enable built-in profiler"     ON)
option(CACHEFLOW_ASAN    "Enable AddressSanitizer"      OFF)

# ---------------------------------------------------------------------------
# Detect llama.cpp — either as a submodule or an installed library
# ---------------------------------------------------------------------------
set(LLAMA_CPP_DIR "${CMAKE_SOURCE_DIR}/third_party/llama.cpp"
    CACHE PATH "Path to llama.cpp source tree")

if(EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
    message(STATUS "Using llama.cpp from: ${LLAMA_CPP_DIR}")
    set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
    add_subdirectory(${LLAMA_CPP_DIR} llama.cpp EXCLUDE_FROM_ALL)
    set(LLAMA_INCLUDE_DIRS
        ${LLAMA_CPP_DIR}/include
        ${LLAMA_CPP_DIR}/ggml/include
        ${LLAMA_CPP_DIR}/common)
    set(LLAMA_LIBRARIES llama ggml)
    set(LLAMA_FOUND TRUE)
else()
    message(STATUS "llama.cpp not found at ${LLAMA_CPP_DIR} — "
                   "building CacheFlow in standalone mode")
    set(LLAMA_FOUND FALSE)
endif()

# ---------------------------------------------------------------------------
# CUDA
# ---------------------------------------------------------------------------
if(CACHEFLOW_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        find_package(CUDAToolkit REQUIRED)
        message(STATUS "CUDA ${CUDAToolkit_VERSION} enabled")
        add_compile_definitions(CACHEFLOW_USE_CUDA=1)
    else()
        message(WARNING "CUDA compiler not found — disabling GPU support")
        set(CACHEFLOW_CUDA OFF)
    endif()
endif()

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
add_compile_options(-Wall -Wextra -Wpedantic -O2)
if(CACHEFLOW_ASAN)
    add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address)
endif()

# ---------------------------------------------------------------------------
# CacheFlow core library
# ---------------------------------------------------------------------------
set(CACHEFLOW_SOURCES
    src/cacheflow/scheduler/scheduler.cpp
    src/cacheflow/scheduler/batch_manager.cpp
    src/cacheflow/kv_cache/block_manager.cpp
    src/cacheflow/kv_cache/memory_pool.cpp
    src/cacheflow/kv_cache/prefix_cache.cpp
    src/cacheflow/decode/batch_decode.cpp
    src/cacheflow/profiler/profiler.cpp
    src/cacheflow/profiler/trace.cpp
)

if(CACHEFLOW_CUDA)
    list(APPEND CACHEFLOW_SOURCES
        src/cacheflow/cuda/paged_attention.cu
        src/cacheflow/cuda/kv_cache_kernels.cu
        src/cacheflow/cuda/memory_utils.cu
    )
endif()

add_library(cacheflow STATIC ${CACHEFLOW_SOURCES})
target_include_directories(cacheflow PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/src/cacheflow
)

if(LLAMA_FOUND)
    target_include_directories(cacheflow PUBLIC ${LLAMA_INCLUDE_DIRS})
    target_link_libraries(cacheflow PUBLIC ${LLAMA_LIBRARIES})
    target_compile_definitions(cacheflow PUBLIC CACHEFLOW_HAS_LLAMA=1)
endif()

if(CACHEFLOW_CUDA)
    target_link_libraries(cacheflow PUBLIC CUDA::cudart CUDA::cuda_driver)
    set_target_properties(cacheflow PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "70;75;80;86;89;90"
    )
endif()

if(CACHEFLOW_PROFILE)
    target_compile_definitions(cacheflow PUBLIC CACHEFLOW_PROFILING=1)
endif()

target_link_libraries(cacheflow PUBLIC pthread)

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
if(CACHEFLOW_BENCH)
    add_subdirectory(benchmarks)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(CACHEFLOW_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS cacheflow ARCHIVE DESTINATION lib)
install(DIRECTORY src/cacheflow/
        DESTINATION include/cacheflow
        FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh")

message(STATUS "")
message(STATUS "CacheFlow configuration:")
message(STATUS "  CUDA support:   ${CACHEFLOW_CUDA}")
message(STATUS "  Tests:          ${CACHEFLOW_TESTS}")
message(STATUS "  Benchmarks:     ${CACHEFLOW_BENCH}")
message(STATUS "  Profiler:       ${CACHEFLOW_PROFILE}")
message(STATUS "  llama.cpp:      ${LLAMA_FOUND}")
message(STATUS "")
