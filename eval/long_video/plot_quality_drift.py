#!/usr/bin/env python3
"""Plot quality drift metrics over time from CSV output.

Usage:
  python eval/long_video/plot_quality_drift.py \
    --csv results/quality_drift/method1/quality_drift_summary.csv \
         results/quality_drift/method2/quality_drift_summary.csv \
    --output results/quality_drift/comparison.png
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True, nargs="+",
                        help="Summary CSVs (label from parent dir name)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", nargs="+", default=None,
                        help="Which metrics to plot (default: all in CSV)")
    parser.add_argument("--title", default="Quality Drift Over Time")
    args = parser.parse_args()

    data = {}
    for csv_path in args.csv:
        label = csv_path.parent.name
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue

        cols = [c for c in rows[0].keys() if c != "time_s"]
        if args.metrics:
            cols = [c for c in cols
                    if any(m.lower() in c.lower() for m in args.metrics)]

        data[label] = {}
        for col in cols:
            times, vals = [], []
            for row in rows:
                v = row.get(col, "")
                if v:
                    times.append(float(row["time_s"]))
                    vals.append(float(v))
            if vals:
                data[label][col] = (np.array(times), np.array(vals))

    all_metrics = set()
    for d in data.values():
        all_metrics.update(d.keys())
    all_metrics = sorted(all_metrics)
    n = len(all_metrics)
    if n == 0:
        print("No data to plot")
        return

    fig, axes = plt.subplots(n, 1, figsize=(10, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(data), 10)))

    for ax, metric in zip(axes, all_metrics):
        for i, (label, d) in enumerate(data.items()):
            if metric in d:
                t, v = d[metric]
                ax.plot(t, v, marker="o", markersize=4, label=label,
                        color=colors[i])
        ax.set_ylabel(metric)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle(args.title, fontsize=14)
    plt.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved: {args.output}")


if __name__ == "__main__":
    main()
