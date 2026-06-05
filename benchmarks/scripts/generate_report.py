#!/usr/bin/env python3
"""Generate a comprehensive benchmark report from all result CSVs."""

import argparse
import os
import glob
import json
from datetime import datetime


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


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def generate_report(results_dir, output_path):
    report = {
        "timestamp": datetime.now().isoformat(),
        "system": "CacheFlow Benchmark Suite",
        "results": {},
    }

    tp_path = os.path.join(results_dir, "throughput_results.csv")
    if os.path.exists(tp_path):
        data = load_csv(tp_path)
        report["results"]["throughput"] = {
            "avg_throughput_tok_s": mean(data.get("throughput_tok_s", [])),
            "avg_latency_ms": mean(data.get("avg_latency_ms", [])),
            "avg_p99_ms": mean(data.get("p99_ms", [])),
            "avg_ttft_ms": mean(data.get("ttft_ms", [])),
            "avg_tpot_ms": mean(data.get("tpot_ms", [])),
            "iterations": len(data.get("throughput_tok_s", [])),
        }

    lat_path = os.path.join(results_dir, "latency_results.csv")
    if os.path.exists(lat_path):
        data = load_csv(lat_path)
        rows = []
        olens = data.get("output_len", [])
        for i in range(len(olens)):
            rows.append({
                "output_len": int(olens[i]),
                "avg_ms": data["avg_ms"][i],
                "p50_ms": data["p50_ms"][i],
                "p90_ms": data["p90_ms"][i],
                "p99_ms": data["p99_ms"][i],
                "stddev_ms": data["stddev_ms"][i],
            })
        report["results"]["latency"] = {"by_output_len": rows}

    sc_path = os.path.join(results_dir, "scaling_results.csv")
    if os.path.exists(sc_path):
        data = load_csv(sc_path)
        rows = []
        conc = data.get("concurrency", [])
        tput = data.get("throughput_tok_s", [])
        for i in range(len(conc)):
            speedup = tput[i] / tput[0] if tput[0] > 0 else 0
            rows.append({
                "concurrency": int(conc[i]),
                "throughput_tok_s": tput[i],
                "avg_latency_ms": data["avg_latency_ms"][i],
                "p99_ms": data["p99_ms"][i],
                "speedup": round(speedup, 2),
            })
        report["results"]["scaling"] = {
            "data": rows,
            "peak_throughput": max(tput),
            "peak_concurrency": int(conc[tput.index(max(tput))]),
            "max_speedup": round(max(tput) / tput[0], 2) if tput[0] else 0,
        }

    kv_path = os.path.join(results_dir, "kv_cache_results.csv")
    if os.path.exists(kv_path):
        data = load_csv(kv_path)
        rows = []
        for i in range(len(data.get("block_size", []))):
            rows.append({
                "block_size": int(data["block_size"][i]),
                "alloc_ops_s": data["alloc_ops_s"][i],
                "free_ops_s": data["free_ops_s"][i],
                "int_frag": data["int_frag"][i],
                "ext_frag": data["ext_frag"][i],
            })
        report["results"]["kv_cache"] = {"by_block_size": rows}

    summary = []
    tp = report["results"].get("throughput", {})
    if tp:
        summary.append(
            f"Throughput: {tp['avg_throughput_tok_s']:.1f} tok/s "
            f"(P99 latency: {tp['avg_p99_ms']:.2f} ms)")

    sc = report["results"].get("scaling", {})
    if sc:
        summary.append(
            f"Scaling: {sc['max_speedup']:.2f}x speedup at "
            f"{sc['peak_concurrency']} concurrent requests "
            f"({sc['peak_throughput']:.1f} tok/s peak)")

    report["summary"] = summary

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {output_path}")

    print("\n" + "=" * 60)
    print("  CacheFlow Benchmark Report")
    print("=" * 60)
    for line in summary:
        print(f"  {line}")
    print("=" * 60)

    return report


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("-d", "--results-dir", default="results",
                        help="Directory containing CSV results")
    parser.add_argument("-o", "--output", default="results/report.json",
                        help="Output report path")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    generate_report(args.results_dir, args.output)


if __name__ == "__main__":
    main()
