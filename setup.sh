#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
THIRD_PARTY="${PROJECT_DIR}/third_party"
LLAMA_DIR="${THIRD_PARTY}/llama.cpp"

echo "CacheFlow Setup"
echo "  Project: ${PROJECT_DIR}"
echo ""

# Clone llama.cpp if not present
if [ ! -d "${LLAMA_DIR}" ]; then
    echo "── Cloning llama.cpp ──"
    mkdir -p "${THIRD_PARTY}"
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "${LLAMA_DIR}"
    echo "  Done: $(cd "${LLAMA_DIR}" && git log --oneline -1)"
else
    echo "  llama.cpp already present at ${LLAMA_DIR}"
fi

# Create results directory
mkdir -p "${PROJECT_DIR}/results"

# Check Python dependencies for plotting
echo ""
echo "── Checking Python dependencies ──"
MISSING=()
for pkg in matplotlib numpy; do
    if python3 -c "import ${pkg}" 2>/dev/null; then
        echo "  ${pkg}: OK"
    else
        echo "  ${pkg}: MISSING"
        MISSING+=("${pkg}")
    fi
done

if [ ${#MISSING[@]} -gt 0 ]; then
    echo ""
    echo "Install missing packages:"
    echo "  pip install ${MISSING[*]}"
fi

# Build
echo ""
echo "── Building CacheFlow ──"
bash "${PROJECT_DIR}/scripts/build.sh" Release auto

echo ""
echo "Setup complete."
echo ""
echo "Quick start:"
echo "  cd ${PROJECT_DIR}/build"
echo "  ./benchmarks/bench_throughput -n 50 -p 128 -o 256 -c 4"
echo ""
echo "Run full benchmarks on PACE GPU node:"
echo "  sbatch scripts/slurm/benchmark.sbatch"
