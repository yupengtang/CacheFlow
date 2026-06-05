#!/usr/bin/env python3
"""Plot scaling benchmark results (1-16 concurrent requests)."""

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


def plot_scaling(data, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    concurrency = np.array(data["concurrency"])
    throughput = np.array(data["throughput_tok_s"])

    # Throughput vs concurrency
    ax = axes[0, 0]
    ax.plot(concurrency, throughput, "o-", color="#2196F3",
            linewidth=2.5, markersize=8, label="Measured")
    ideal = throughput[0] * concurrency / concurrency[0]
    ax.plot(concurrency, ideal, "--", color="#9E9E9E",
            linewidth=1.5, label="Ideal linear")
    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Throughput (tokens/sec)")
    ax.set_title("Throughput Scaling")
    ax.legend()
    ax.grid(alpha=0.3)

    # Scaling efficiency
    ax = axes[0, 1]
    efficiency = throughput / ideal * 100
    ax.bar(range(len(concurrency)), efficiency,
           tick_label=[str(int(c)) for c in concurrency],
           color="#4CAF50", alpha=0.8)
    ax.axhline(y=100, color="#F44336", linestyle="--", linewidth=1)
    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Scaling Efficiency (%)")
    ax.set_title("Scaling Efficiency (vs Linear)")
    ax.set_ylim(0, 120)
    ax.grid(axis="y", alpha=0.3)

    # Latency vs concurrency
    ax = axes[1, 0]
    for pct, col, label in [
        ("avg_latency_ms", "#4CAF50", "Avg"),
        ("p50_ms", "#2196F3", "P50"),
        ("p90_ms", "#FF9800", "P90"),
        ("p99_ms", "#F44336", "P99"),
    ]:
        if pct in data:
            ax.plot(concurrency, data[pct], "o-", color=col,
                    linewidth=2, markersize=6, label=label)
    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Latency vs Concurrency")
    ax.legend()
    ax.grid(alpha=0.3)

    # Cache utilization and fragmentation
    ax = axes[1, 1]
    if "gpu_util" in data:
        ax.plot(concurrency, np.array(data["gpu_util"]) * 100, "o-",
                color="#2196F3", linewidth=2, markersize=6,
                label="GPU Cache %")
    if "fragmentation" in data:
        ax.plot(concurrency, np.array(data["fragmentation"]) * 100, "s-",
                color="#F44336", linewidth=2, markersize=6,
                label="Fragmentation %")
    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Percentage")
    ax.set_title("Cache Utilization & Fragmentation")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "scaling_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_speedup_curve(data, output_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    concurrency = np.array(data["concurrency"])
    throughput = np.array(data["throughput_tok_s"])
    speedup = throughput / throughput[0]

    ax.plot(concurrency, speedup, "o-", color="#2196F3",
            linewidth=2.5, markersize=8, label="Actual speedup")
    ax.plot(concurrency, concurrency / concurrency[0], "--",
            color="#9E9E9E", linewidth=1.5, label="Ideal (linear)")

    amdahl_p = 0.85
    amdahl = 1.0 / ((1 - amdahl_p) + amdahl_p / concurrency)
    ax.plot(concurrency, amdahl, ":", color="#FF9800",
            linewidth=1.5, label=f"Amdahl (p={amdahl_p})")

    ax.set_xlabel("Concurrent Requests")
    ax.set_ylabel("Speedup (×)")
    ax.set_title("Throughput Speedup Curve")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "speedup_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Plot scaling results")
    parser.add_argument("csv", help="Path to scaling_results.csv")
    parser.add_argument("-o", "--output-dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    data = load_csv(args.csv)

    plot_scaling(data, args.output_dir)
    plot_speedup_curve(data, args.output_dir)

    throughput = np.array(data["throughput_tok_s"])
    concurrency = np.array(data["concurrency"])
    print(f"\nPeak throughput: {throughput.max():.1f} tok/s "
          f"at {int(concurrency[throughput.argmax()])} concurrent requests")
    print(f"Max speedup: {throughput.max() / throughput[0]:.2f}×")


if __name__ == "__main__":
    main()
