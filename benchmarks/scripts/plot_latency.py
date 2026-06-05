#!/usr/bin/env python3
"""Plot latency benchmark results across output lengths."""

import argparse
import os

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


def plot_latency_vs_output_len(data, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    olens = np.array(data["output_len"])

    ax = axes[0, 0]
    ax.plot(olens, data["avg_ms"], "o-", color="#2196F3",
            linewidth=2, markersize=6, label="Avg")
    ax.fill_between(olens,
                    np.array(data["avg_ms"]) - np.array(data["stddev_ms"]),
                    np.array(data["avg_ms"]) + np.array(data["stddev_ms"]),
                    alpha=0.2, color="#2196F3")
    ax.set_xlabel("Output Length (tokens)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Average Latency vs Output Length")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for pct, col, label in [
        ("p50_ms", "#4CAF50", "P50"),
        ("p90_ms", "#FF9800", "P90"),
        ("p95_ms", "#F44336", "P95"),
        ("p99_ms", "#9C27B0", "P99"),
    ]:
        if pct in data:
            ax.plot(olens, data[pct], "o-", color=col,
                    linewidth=2, markersize=6, label=label)
    ax.set_xlabel("Output Length (tokens)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency Percentiles vs Output Length")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    if "tpot_ms" in data and "ttft_ms" in data:
        ax.plot(olens, data["ttft_ms"], "o-", color="#2196F3",
                linewidth=2, markersize=6, label="TTFT")
        ax.plot(olens, data["tpot_ms"], "s-", color="#F44336",
                linewidth=2, markersize=6, label="TPOT")
        ax.set_xlabel("Output Length (tokens)")
        ax.set_ylabel("Time (ms)")
        ax.set_title("TTFT & TPOT vs Output Length")
        ax.legend()
        ax.grid(alpha=0.3)

    ax = axes[1, 1]
    if "throughput_tok_s" in data:
        ax.bar(range(len(olens)), data["throughput_tok_s"],
               tick_label=[str(int(o)) for o in olens],
               color="#2196F3", alpha=0.8)
        ax.set_xlabel("Output Length (tokens)")
        ax.set_ylabel("Throughput (tokens/sec)")
        ax.set_title("Throughput vs Output Length")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "latency_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_latency_variance(data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    olens = data["output_len"]
    cv = [s / m if m > 0 else 0
          for s, m in zip(data["stddev_ms"], data["avg_ms"])]

    ax.bar(range(len(olens)), cv,
           tick_label=[str(int(o)) for o in olens],
           color="#FF9800", alpha=0.8)
    ax.set_xlabel("Output Length (tokens)")
    ax.set_ylabel("Coefficient of Variation")
    ax.set_title("Latency Variance Stability (lower = more stable)")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "latency_variance.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot latency results")
    parser.add_argument("csv", help="Path to latency_results.csv")
    parser.add_argument("-o", "--output-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_csv(args.csv)

    plot_latency_vs_output_len(data, args.output_dir)
    plot_latency_variance(data, args.output_dir)


if __name__ == "__main__":
    main()
