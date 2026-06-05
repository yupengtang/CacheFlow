#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "${SCRIPT_DIR}")")"
BUILD_DIR="${PROJECT_DIR}/build"
RESULTS_DIR="${PROJECT_DIR}/results/run_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${RESULTS_DIR}"

echo "CacheFlow — Full Benchmark Suite"
echo "  Results: ${RESULTS_DIR}"
echo ""

cd "${BUILD_DIR}"

echo "═══ Throughput ═══"
./benchmarks/bench_throughput \
    -n 100 -p 128 -o 256 -c 8 -i 3 \
    -d "${RESULTS_DIR}" -v

echo ""
echo "═══ Latency ═══"
./benchmarks/bench_latency \
    -n 50 -p 128 -d "${RESULTS_DIR}"

echo ""
echo "═══ Scaling ═══"
./benchmarks/bench_scaling \
    -n 50 -p 128 -o 256 -i 3 \
    -d "${RESULTS_DIR}"

echo ""
echo "═══ KV-Cache ═══"
./benchmarks/bench_kv_cache \
    -n 500 -p 256 -d "${RESULTS_DIR}"

echo ""
echo "═══ Generating Visualizations ═══"
cd "${PROJECT_DIR}"

for script in plot_throughput plot_latency plot_scaling; do
    csv_file=$(ls "${RESULTS_DIR}"/*results*.csv 2>/dev/null | grep "${script#plot_}" | head -1 || true)
    if [ -n "${csv_file}" ]; then
        python3 "benchmarks/scripts/${script}.py" "${csv_file}" -o "${RESULTS_DIR}"
    fi
done

echo ""
echo "═══ Report ═══"
python3 benchmarks/scripts/generate_report.py \
    -d "${RESULTS_DIR}" -o "${RESULTS_DIR}/report.json"

echo ""
echo "All benchmarks complete."
echo "Results: ${RESULTS_DIR}"
