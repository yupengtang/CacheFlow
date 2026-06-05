#!/bin/bash
#SBATCH --job-name=cacheflow-sweep
#SBATCH --account=bios
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=cacheflow-sweep-%j.out
#SBATCH --error=cacheflow-sweep-%j.err

set -euo pipefail

module load cuda/12.6.1

PROJECT_DIR="${HOME}/projects/Engineering/CacheFlow"
BUILD_DIR="${PROJECT_DIR}/build"
RESULTS_DIR="${PROJECT_DIR}/results/sweep_${SLURM_JOB_ID}"

mkdir -p "${RESULTS_DIR}"

echo "CacheFlow Parameter Sweep"
echo "Job ID: ${SLURM_JOB_ID}"
echo ""

cd "${BUILD_DIR}"

# Sweep over prompt lengths
for PLEN in 64 128 256 512 1024 2048; do
    echo "── Prompt length: ${PLEN} ──"
    PDIR="${RESULTS_DIR}/plen_${PLEN}"
    mkdir -p "${PDIR}"

    ./benchmarks/bench_scaling \
        --num-requests 50 \
        --prompt-len "${PLEN}" \
        --output-len 256 \
        --iterations 3 \
        --output-dir "${PDIR}" 2>&1 | tail -20
done

# Sweep over generation lengths for long-horizon profiling
for OLEN in 256 512 1024 2048 4096 8192; do
    echo "── Output length: ${OLEN} (long-horizon) ──"
    ODIR="${RESULTS_DIR}/olen_${OLEN}"
    mkdir -p "${ODIR}"

    ./benchmarks/bench_latency \
        --num-requests 20 \
        --prompt-len 128 \
        --output-dir "${ODIR}" 2>&1 | tail -10
done

# Sweep over block sizes for KV-cache optimization
for BS in 8 16 32 64; do
    echo "── Block size: ${BS} ──"
    BDIR="${RESULTS_DIR}/bsize_${BS}"
    mkdir -p "${BDIR}"

    ./benchmarks/bench_kv_cache \
        --num-requests 500 \
        --prompt-len 256 \
        --block-size "${BS}" \
        --output-dir "${BDIR}" 2>&1 | tail -10
done

# Sweep over prefix sharing ratios
for SR in 0.0 0.25 0.5 0.75; do
    echo "── Shared prefix: ${SR} ──"
    SDIR="${RESULTS_DIR}/shared_${SR}"
    mkdir -p "${SDIR}"

    ./benchmarks/bench_throughput \
        --num-requests 100 \
        --prompt-len 256 \
        --output-len 256 \
        --concurrent 8 \
        --iterations 3 \
        --shared-prefix "${SR}" \
        --prefix-cache \
        --output-dir "${SDIR}" 2>&1 | tail -10
done

echo ""
echo "Sweep complete. Results: ${RESULTS_DIR}"
