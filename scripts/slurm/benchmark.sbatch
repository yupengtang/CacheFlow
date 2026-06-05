#!/bin/bash
#SBATCH --job-name=cacheflow-bench
#SBATCH --account=mgt6655
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:V100:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=cacheflow-bench-%j.out
#SBATCH --error=cacheflow-bench-%j.err

set -euo pipefail

module load cuda/12.6.1
module load gcc/12.3.0 2>/dev/null || true

PROJECT_DIR="${HOME}/projects/Engineering/CacheFlow"
BUILD_DIR="${PROJECT_DIR}/build"
RESULTS_DIR="${PROJECT_DIR}/results/${SLURM_JOB_ID}"

mkdir -p "${RESULTS_DIR}"

echo "========================================="
echo "CacheFlow Benchmark Suite"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "CUDA:   $(nvcc --version | tail -1)"
echo "========================================="
echo ""

cd "${BUILD_DIR}"

echo "── Throughput Benchmark ──"
./benchmarks/bench_throughput \
    --num-requests 200 \
    --prompt-len 128 \
    --output-len 256 \
    --concurrent 8 \
    --iterations 5 \
    --output-dir "${RESULTS_DIR}" \
    --verbose

echo ""
echo "── Latency Benchmark ──"
./benchmarks/bench_latency \
    --num-requests 100 \
    --prompt-len 128 \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "── Scaling Benchmark ──"
./benchmarks/bench_scaling \
    --num-requests 50 \
    --prompt-len 128 \
    --output-len 256 \
    --iterations 3 \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "── KV-Cache Benchmark ──"
./benchmarks/bench_kv_cache \
    --num-requests 1000 \
    --prompt-len 256 \
    --output-dir "${RESULTS_DIR}"

echo ""
echo "── Generating Plots ──"
cd "${PROJECT_DIR}"
python3 benchmarks/scripts/plot_throughput.py \
    "${RESULTS_DIR}/throughput_results.csv" -o "${RESULTS_DIR}"
python3 benchmarks/scripts/plot_latency.py \
    "${RESULTS_DIR}/latency_results.csv" -o "${RESULTS_DIR}"
python3 benchmarks/scripts/plot_scaling.py \
    "${RESULTS_DIR}/scaling_results.csv" -o "${RESULTS_DIR}"

echo ""
echo "── Generating Report ──"
python3 benchmarks/scripts/generate_report.py \
    -d "${RESULTS_DIR}" -o "${RESULTS_DIR}/report.json"

echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "Done."
