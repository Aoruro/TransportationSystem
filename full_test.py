"""Comprehensive smoke test for the TSP project."""

from pathlib import Path
import sys

import numpy as np


failures = []


def run_check(name, check):
    """Run one smoke check while allowing later modules to be inspected."""
    print(f"\nTesting {name}...")
    try:
        check()
        print(f"  [PASS] {name}")
    except Exception as exc:
        failures.append((name, exc))
        print(f"  [FAIL] {name}: {exc}")


def make_small_instance():
    from data.data_processor import TSPInstance

    coords = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [0.5, 1.5],
        [1.5, 0.5],
    ])
    return TSPInstance("smoke_test", coords)


def check_data():
    from data.data_processor import DataProcessor

    processor = DataProcessor("tsp_instances_dataset.csv")
    loaded = processor.load_from_csv()
    filtered = processor.preprocess(max_cities=25)
    if not loaded or not filtered:
        raise RuntimeError("Dataset did not produce any valid instances")
    train, val, test = processor.split_dataset(train_size=10, val_size=5, test_size=5)
    print(f"  Loaded {len(loaded)} instances; split {len(train)}/{len(val)}/{len(test)}")


def check_search():
    from search.bfs import BFSSolver
    from search.ucs import UCSSolver
    from search.astar import AStarSolver

    instance = make_small_instance()
    results = [
        BFSSolver(instance.dist_matrix).solve(),
        UCSSolver(instance.dist_matrix).solve(),
        AStarSolver(instance.dist_matrix).solve(),
    ]
    costs = [result['cost'] for result in results]
    if not all(result['success'] for result in results):
        raise RuntimeError("At least one search algorithm failed")
    if max(costs) - min(costs) > 1e-9:
        raise RuntimeError(f"Search algorithms disagree: {costs}")

    callback_count = [0]

    def callback(_node):
        callback_count[0] += 1

    result = AStarSolver(instance.dist_matrix).solve_with_callback(callback)
    if callback_count[0] != len(result['search_tree']):
        raise RuntimeError("A* callback count does not match search tree size")
    print(f"  Shared optimal cost: {costs[0]:.4f}")


def check_ml():
    from ml.model import TSPMLModel
    from ml.pseudo_labels import generate_pseudo_labels_for_instance
    from ml.learning_astar import LearningAStar

    instance = make_small_instance()
    X, y = generate_pseudo_labels_for_instance(instance)
    model = TSPMLModel("rf")
    model.train(X, y, cv_folds=3)
    result = LearningAStar(instance.dist_matrix, model=model).solve()
    if not result['success']:
        raise RuntimeError("Learning A* failed")
    print(f"  Generated {len(y)} pseudo-label samples")


def check_csp():
    from csp.tsptw_solver import TSPTWInstance, TSPTWSolver

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    instance = TSPTWInstance(coords, [(0, 10), (1, 8), (2, 9), (1, 10)])
    solver = TSPTWSolver(instance)
    result = solver.solve()
    if not result['success']:
        raise RuntimeError("TSPTW solver failed")
    valid, message = solver.verify_solution(result['path'])
    if not valid:
        raise RuntimeError(message)


def check_experiments():
    from experiments import ExperimentRunner

    instance = make_small_instance()
    result = ExperimentRunner(instances=[instance]).run_single_experiment(instance, "astar")
    if not result.optimality_maintained or result.relative_error != 0:
        raise RuntimeError("Experiment baseline metrics are inconsistent")


def check_visualization():
    from visualization import SimpleVisualizer

    instance = make_small_instance()
    output = Path("smoke_test_plot.png")
    try:
        SimpleVisualizer(instance).plot_instance(path=[0, 1, 2, 3, 4, 5, 0], save_path=str(output))
        if not output.exists():
            raise RuntimeError("Visualization did not create an output image")
    finally:
        if output.exists():
            output.unlink()


def main():
    print("=" * 60)
    print("TSP Project - Comprehensive Smoke Test")
    print("=" * 60)
    run_check("data module", check_data)
    run_check("search algorithms", check_search)
    run_check("machine learning module", check_ml)
    run_check("CSP module", check_csp)
    run_check("experiments module", check_experiments)
    run_check("visualization module", check_visualization)

    print("\n" + "=" * 60)
    if failures:
        print(f"{len(failures)} smoke check(s) failed")
        return 1
    print("All smoke checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
