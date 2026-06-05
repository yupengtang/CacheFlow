#!/usr/bin/env python3
"""Plot throughput benchmark results."""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def load_csv(path):
    data = {}
    with open(path) as f:
        header = f.readline().strip().split(",")
        for col in header:
            data[col] = []
        for line in f:
            vals = line.strip().split(",")
            for col, val in zip(header, vals):
                try:
                    data[col].append(float(val))
                except ValueError:
                    data[col].append(val)
    return data


def plot_throughput_iterations(data, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    iters = range(1, len(data["throughput_tok_s"]) + 1)

    ax = axes[0]
    ax.bar(iters, data["throughput_tok_s"], color="#2196F3", alpha=0.8)
    avg = np.mean(data["throughput_tok_s"])
    ax.axhline(y=avg, color="#F44336", linestyle="--", linewidth=2,
               label=f"Mean: {avg:.1f} tok/s")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput Across Iterations")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    metrics = ["avg_latency_ms", "p50_ms", "p90_ms", "p95_ms", "p99_ms"]
    labels = ["Avg", "P50", "P90", "P95", "P99"]
    colors = ["#4CAF50", "#2196F3", "#FF9800", "#F44336", "#9C27B0"]
    x = np.arange(len(iters))
    width = 0.15
    for i, (m, l, c) in enumerate(zip(metrics, labels, colors)):
        if m in data:
            ax.bar(x + i * width, data[m], width, label=l, color=c,
                   alpha=0.8)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Distribution per Iteration")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([str(i) for i in iters])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "throughput_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_ttft_tpot(data, output_dir):
    if "ttft_ms" not in data or "tpot_ms" not in data:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    iters = range(1, len(data["ttft_ms"]) + 1)

    ax.plot(iters, data["ttft_ms"], "o-", color="#2196F3",
            linewidth=2, markersize=8, label="TTFT")
    ax.plot(iters, data["tpot_ms"], "s-", color="#F44336",
            linewidth=2, markersize=8, label="TPOT")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Time (ms)")
    ax.set_title("Time to First Token / Time per Output Token")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "ttft_tpot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot throughput results")
    parser.add_argument("csv", help="Path to throughput_results.csv")
    parser.add_argument("-o", "--output-dir", default="results",
                        help="Output directory for plots")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_csv(args.csv)

    plot_throughput_iterations(data, args.output_dir)
    plot_ttft_tpot(data, args.output_dir)

    print("\nSummary:")
    print(f"  Avg Throughput: {np.mean(data['throughput_tok_s']):.1f} tok/s")
    print(f"  Std Throughput: {np.std(data['throughput_tok_s']):.1f} tok/s")
    if "p99_ms" in data:
        print(f"  Avg P99 Latency: {np.mean(data['p99_ms']):.2f} ms")


if __name__ == "__main__":
    main()
