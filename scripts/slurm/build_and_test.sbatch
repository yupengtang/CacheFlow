#!/bin/bash
#SBATCH --job-name=cacheflow-build
#SBATCH --account=bios
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

echo "========================================="
echo "CacheFlow CUDA Build & Test"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "========================================="

module load cuda/12.6.1

echo ""
nvidia-smi
echo ""
nvcc --version
echo ""

PROJECT_DIR="${HOME}/projects/Engineering/CacheFlow"
BUILD_DIR="${PROJECT_DIR}/build_cuda"
RESULTS_DIR="${PROJECT_DIR}/results/gpu_test_${SLURM_JOB_ID}"

mkdir -p "${BUILD_DIR}" "${RESULTS_DIR}"

# ── Build ──
echo "═══ Configuring with CUDA ═══"
cd "${BUILD_DIR}"
cmake "${PROJECT_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCACHEFLOW_CUDA=ON \
    -DCACHEFLOW_TESTS=ON \
    -DCACHEFLOW_BENCH=ON \
    -DCACHEFLOW_PROFILE=ON

echo ""
echo "═══ Building ═══"
cmake --build . --parallel 8

# ── Tests ──
echo ""
echo "═══ Running Tests ═══"
ctest --output-on-failure --parallel 4

# ── Quick benchmarks ──
echo ""
echo "═══ Quick Throughput Benchmark ═══"
./benchmarks/bench_throughput \
    -n 50 -p 128 -o 256 -c 4 -i 3 \
    -d "${RESULTS_DIR}" -v

echo ""
echo "═══ Scaling Benchmark (1-16) ═══"
./benchmarks/bench_scaling \
    -n 30 -p 128 -o 128 -i 2 \
    -d "${RESULTS_DIR}"

echo ""
echo "═══ KV-Cache Benchmark ═══"
./benchmarks/bench_kv_cache \
    -n 200 -p 128 -d "${RESULTS_DIR}"

echo ""
echo "═══ Latency Benchmark ═══"
./benchmarks/bench_latency \
    -n 30 -p 128 -d "${RESULTS_DIR}"

# ── Report ──
echo ""
echo "═══ Generating Report ═══"
cd "${PROJECT_DIR}"
python3 benchmarks/scripts/generate_report.py \
    -d "${RESULTS_DIR}" -o "${RESULTS_DIR}/report.json"

echo ""
echo "Build + test + benchmarks complete."
echo "Results: ${RESULTS_DIR}"
