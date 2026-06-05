#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${PROJECT_DIR}/build"
BUILD_TYPE="${1:-Release}"
CUDA="${2:-auto}"

echo "CacheFlow Build"
echo "  Source: ${PROJECT_DIR}"
echo "  Build:  ${BUILD_DIR}"
echo "  Type:   ${BUILD_TYPE}"
echo ""

# Load CUDA module on PACE if available
if command -v module &>/dev/null; then
    module load cuda/12.6.1 2>/dev/null || true
fi

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DCACHEFLOW_TESTS=ON
    -DCACHEFLOW_BENCH=ON
    -DCACHEFLOW_PROFILE=ON
)

if [ "${CUDA}" = "off" ]; then
    CMAKE_ARGS+=(-DCACHEFLOW_CUDA=OFF)
    echo "  CUDA: disabled"
elif [ "${CUDA}" = "on" ] || ([ "${CUDA}" = "auto" ] && command -v nvcc &>/dev/null); then
    CMAKE_ARGS+=(-DCACHEFLOW_CUDA=ON)
    echo "  CUDA: enabled ($(nvcc --version 2>/dev/null | grep release | awk '{print $6}' || echo 'unknown'))"
else
    CMAKE_ARGS+=(-DCACHEFLOW_CUDA=OFF)
    echo "  CUDA: not found, building CPU-only"
fi

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo ""
echo "── Configuring ──"
cmake "${PROJECT_DIR}" "${CMAKE_ARGS[@]}"

echo ""
echo "── Building ──"
NPROC=$(nproc 2>/dev/null || echo 4)
cmake --build . --parallel "${NPROC}"

echo ""
echo "── Running Tests ──"
ctest --output-on-failure --parallel "${NPROC}" || true

echo ""
echo "Build complete. Binaries in: ${BUILD_DIR}"
echo ""
echo "Available executables:"
find "${BUILD_DIR}" -maxdepth 3 -type f -executable -name "bench_*" -o -name "test_*" 2>/dev/null | sort
