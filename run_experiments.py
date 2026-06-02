"""Generate reproducible comparison tables and figures for the project report."""

from argparse import ArgumentParser
from pathlib import Path

from data.data_processor import load_tsp_instances
from experiments.runner import ExperimentRunner
from experiments.statistics import StatisticalAnalyzer


CORE_ALGORITHMS = ["bfs", "ucs", "astar"]
ML_ALGORITHMS = ["astar", "learning_astar_rf", "learning_astar_mlp"]


def save_artifacts(results, output_dir: Path, stem: str):
    """Save JSON, CSV, report text, and two comparison figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    runner = ExperimentRunner()
    runner.results = results
    runner.save_results(str(output_dir / f"{stem}.json"))
    runner.results_to_dataframe().to_csv(output_dir / f"{stem}.csv", index=False)

    analyzer = StatisticalAnalyzer(results)
    analyzer.generate_report(str(output_dir / f"{stem}_analysis.txt"))
    analyzer.plot_performance_comparison(
        "nodes_expanded", str(output_dir / f"{stem}_nodes.png")
    )
    analyzer.plot_performance_comparison(
        "time_seconds", str(output_dir / f"{stem}_time.png")
    )


def run_core_experiments(instances, output_dir: Path):
    """Compare the three required search strategies on identical instances."""
    runner = ExperimentRunner(instances=instances)
    results = runner.run_batch(CORE_ALGORITHMS, instances, verbose=True)
    save_artifacts(results, output_dir, "core_search")
    return results


def run_ml_experiments(training_instances, test_instances, output_dir: Path):
    """Compare standard and learning-enhanced A* on held-out instances."""
    runner = ExperimentRunner(instances=test_instances)
    runner.train_ml_models(training_instances)
    results = runner.run_batch(ML_ALGORITHMS, test_instances, verbose=True)
    save_artifacts(results, output_dir, "learning_astar")
    return results


def print_summary(title: str, results):
    """Print compact metrics after an experiment run."""
    print(f"\n{title}")
    print("-" * len(title))
    runner = ExperimentRunner()
    runner.results = results
    for algorithm, metrics in runner.summarize().items():
        print(
            f"{algorithm}: nodes={metrics['avg_nodes']:.1f}, "
            f"time={metrics['avg_time']:.4f}s, "
            f"success={metrics['success_rate']:.0%}, "
            f"optimal={metrics['optimality_rate']:.0%}"
        )


def main():
    """Run report-oriented experiments from the command line."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="tsp_small_instances.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--core-instances", type=int, default=20)
    parser.add_argument("--include-ml", action="store_true")
    parser.add_argument("--ml-train-instances", type=int, default=8)
    parser.add_argument("--ml-test-instances", type=int, default=6)
    args = parser.parse_args()

    required = args.core_instances
    if args.include_ml:
        required = max(required, args.ml_train_instances + args.ml_test_instances)

    instances = load_tsp_instances(num_instances=required, data_path=args.data)
    if len(instances) < required:
        raise ValueError(f"Requested {required} instances but loaded {len(instances)}")

    output_dir = Path(args.output_dir)
    core_results = run_core_experiments(instances[:args.core_instances], output_dir)
    print_summary("Core search comparison", core_results)

    if args.include_ml:
        split = args.ml_train_instances
        ml_results = run_ml_experiments(
            instances[:split],
            instances[split:split + args.ml_test_instances],
            output_dir
        )
        print_summary("Learning A* comparison", ml_results)

    print(f"\nArtifacts written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
