"""Generate report-ready benchmark data for the TSPTW CSP solver."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path
from typing import Iterable

import numpy as np

from csp.tsptw_solver import TSPTWInstance, TSPTWSolver


def generate_feasible_tsptw_instance(n_cities: int, seed: int) -> TSPTWInstance:
    """Create a deterministic TSPTW instance with at least one feasible route."""

    rng = np.random.default_rng(seed)
    coords = rng.random((n_cities, 2)) * 10.0
    service_times = rng.uniform(0.05, 0.5, n_cities).tolist()
    service_times[0] = 0.0

    route = [0] + rng.permutation(np.arange(1, n_cities)).tolist()
    time_windows: list[tuple[float, float] | None] = [None] * n_cities
    time_windows[0] = (0.0, 1_000.0)

    current_time = 0.0
    for previous, city in zip(route, route[1:]):
        travel = float(np.linalg.norm(coords[previous] - coords[city]))
        arrival = current_time + travel
        earliest = max(0.0, arrival - rng.uniform(0.0, 2.0))
        latest = arrival + rng.uniform(10.0, 22.0)
        time_windows[city] = (earliest, latest)
        current_time = max(arrival, earliest) + service_times[city]

    return TSPTWInstance(coords, [tw for tw in time_windows if tw is not None], service_times)


def benchmark(
    sizes: Iterable[int],
    samples_per_size: int,
    seed_base: int = 2026,
) -> tuple[list[dict], dict]:
    """Run the benchmark and return row-level results plus summary metrics."""

    rows: list[dict] = []
    for n_cities in sizes:
        for sample_index in range(samples_per_size):
            seed = seed_base + n_cities * 1_000 + sample_index
            instance = generate_feasible_tsptw_instance(n_cities, seed)
            solver = TSPTWSolver(instance)

            started = time.perf_counter()
            result = solver.solve()
            elapsed = time.perf_counter() - started
            feasible = False
            message = "not solved"
            if result["success"]:
                feasible, message = solver.verify_solution(result["path"])

            rows.append({
                "sample_id": f"tsptw_{n_cities}_{sample_index:02d}",
                "n_cities": n_cities,
                "seed": seed,
                "success": bool(result["success"] and feasible),
                "cost": None if not result["success"] else float(result["cost"]),
                "solve_time_seconds": elapsed,
                "solve_time_ms": elapsed * 1000.0,
                "path_length": len(result["path"]),
                "verification": message,
            })

    successful = [row for row in rows if row["success"]]
    solve_times = [row["solve_time_seconds"] for row in rows]
    costs = [row["cost"] for row in successful if row["cost"] is not None]

    by_size: dict[str, dict] = {}
    for n_cities in sorted({row["n_cities"] for row in rows}):
        subset = [row for row in rows if row["n_cities"] == n_cities]
        subset_times = [row["solve_time_seconds"] for row in subset]
        subset_success = [row for row in subset if row["success"]]
        by_size[str(n_cities)] = {
            "samples": len(subset),
            "success_rate": len(subset_success) / len(subset),
            "avg_solve_time_seconds": statistics.mean(subset_times),
            "avg_solve_time_ms": statistics.mean(subset_times) * 1000.0,
            "max_solve_time_ms": max(subset_times) * 1000.0,
        }

    summary = {
        "total_samples": len(rows),
        "sizes": sorted({row["n_cities"] for row in rows}),
        "samples_per_size": samples_per_size,
        "success_count": len(successful),
        "success_rate": len(successful) / len(rows) if rows else 0.0,
        "avg_solve_time_seconds": statistics.mean(solve_times) if solve_times else 0.0,
        "avg_solve_time_ms": statistics.mean(solve_times) * 1000.0 if solve_times else 0.0,
        "median_solve_time_ms": statistics.median(solve_times) * 1000.0 if solve_times else 0.0,
        "max_solve_time_ms": max(solve_times) * 1000.0 if solve_times else 0.0,
        "avg_success_cost": statistics.mean(costs) if costs else None,
        "by_size": by_size,
    }
    return rows, summary


def write_outputs(rows: list[dict], summary: dict, output_dir: Path) -> None:
    """Write CSV, JSON, and text summary artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "csp_benchmark.csv"
    json_path = output_dir / "csp_benchmark.json"
    txt_path = output_dir / "csp_benchmark_summary.txt"
    plot_path = output_dir / "csp_benchmark_time.png"

    fieldnames = [
        "sample_id",
        "n_cities",
        "seed",
        "success",
        "cost",
        "solve_time_seconds",
        "solve_time_ms",
        "path_length",
        "verification",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=2)

    lines = [
        "TSPTW CSP Benchmark Summary",
        "=" * 60,
        f"Total samples: {summary['total_samples']}",
        f"City sizes: {summary['sizes']}",
        f"Samples per size: {summary['samples_per_size']}",
        f"Success rate: {summary['success_rate']:.1%}",
        f"Average solve time: {summary['avg_solve_time_ms']:.3f} ms",
        f"Median solve time: {summary['median_solve_time_ms']:.3f} ms",
        f"Max solve time: {summary['max_solve_time_ms']:.3f} ms",
        "",
        "By city count",
        "-" * 60,
    ]
    for n_cities, stats in summary["by_size"].items():
        lines.append(
            f"N={n_cities}: samples={stats['samples']}, "
            f"success={stats['success_rate']:.1%}, "
            f"avg={stats['avg_solve_time_ms']:.3f} ms, "
            f"max={stats['max_solve_time_ms']:.3f} ms"
        )
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib.pyplot as plt

        sizes = [int(size) for size in summary["by_size"]]
        avg_times = [summary["by_size"][str(size)]["avg_solve_time_ms"] for size in sizes]
        max_times = [summary["by_size"][str(size)]["max_solve_time_ms"] for size in sizes]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(sizes, avg_times, marker="o", linewidth=2.5, label="Average")
        ax.plot(sizes, max_times, marker="s", linewidth=2, linestyle="--", label="Max")
        ax.set_title("TSPTW CSP Solver Time by City Count")
        ax.set_xlabel("Number of cities")
        ax.set_ylabel("Solve time (ms)")
        ax.set_xticks(sizes)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
    except ImportError:
        plot_path = None

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(f"Wrote {txt_path}")
    if plot_path is not None:
        print(f"Wrote {plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-per-size", type=int, default=10)
    parser.add_argument("--sizes", default="4,5,6,7,8")
    parser.add_argument("--output-dir", default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sizes = [int(value.strip()) for value in args.sizes.split(",") if value.strip()]
    rows, summary = benchmark(sizes, args.samples_per_size)
    write_outputs(rows, summary, Path(args.output_dir))


if __name__ == "__main__":
    main()
