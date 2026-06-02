"""TSP project command-line entry point."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_menu():
    """Print the interactive menu."""
    print("=" * 60)
    print("TSP Project - Main Menu")
    print("=" * 60)
    print("1. Run Quick Test")
    print("2. Run Visualization UI")
    print("3. Test Search Algorithms (BFS/UCS/A*)")
    print("4. Test Machine Learning Module")
    print("5. Test CSP Module (TSPTW)")
    print("6. Run Unit Tests")
    print("0. Exit")
    print("=" * 60)


def _generate_small_tsp_instance(n=8):
    """Generate a deterministic small TSP instance."""
    import numpy as np
    from data.data_processor import TSPInstance

    rng = np.random.default_rng(42)
    coords = rng.random((n, 2))
    dist_matrix = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    return TSPInstance(f"random_{n}city", coords, dist_matrix)


def test_search():
    """Run a small comparison of the three search algorithms."""
    from data.data_processor import DataProcessor
    from search.bfs import BFSSolver
    from search.ucs import UCSSolver
    from search.astar import AStarSolver

    processor = DataProcessor("tsp_instances_dataset.csv")
    processor.load_from_csv()
    astar_instances = processor.preprocess(max_cities=25)
    small_instance = _generate_small_tsp_instance()

    print("\n=== Testing BFS and UCS ===")
    for name, solver_class in (("BFS", BFSSolver), ("UCS", UCSSolver)):
        result = solver_class(small_instance.dist_matrix).solve()
        print(
            f"[OK] {name}: Success={result['success']}, "
            f"Cost={result['cost']:.2f}, Nodes={result['nodes_expanded']}"
        )

    print("\n=== Testing A* ===")
    astar_instance = astar_instances[0] if astar_instances else small_instance
    result = AStarSolver(astar_instance.dist_matrix).solve()
    print(
        f"[OK] A*: Success={result['success']}, "
        f"Cost={result['cost']:.2f}, Nodes={result['nodes_expanded']}"
    )


def test_ml():
    """Run a small machine-learning smoke test."""
    from ml.model import TSPMLModel
    from ml.pseudo_labels import generate_pseudo_labels_for_instance
    from ml.learning_astar import LearningAStar

    instance = _generate_small_tsp_instance()
    X, y = generate_pseudo_labels_for_instance(instance)
    print(f"\nGenerated {len(X)} pseudo-label samples")

    model = TSPMLModel("rf")
    metrics = model.train(X, y)
    print(f"[OK] Random Forest accuracy: {metrics['accuracy']:.4f}")

    result = LearningAStar(instance.dist_matrix, model=model, lambda_param=0.5).solve()
    print(f"[OK] Learning A*: Cost={result['cost']:.2f}, Nodes={result['nodes_expanded']}")


def test_csp():
    """Run a small TSPTW smoke test."""
    import numpy as np
    from csp.tsptw_solver import TSPTWInstance, TSPTWSolver

    coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    instance = TSPTWInstance(coords, [(0, 10), (1, 8), (2, 9), (1, 10)])
    solver = TSPTWSolver(instance)
    result = solver.solve()
    print(f"\n[OK] TSPTW success: {result['success']}")
    if result['success']:
        print(f"[OK] Path: {result['path']}")
        print(f"[OK] Cost: {result['cost']:.2f}")


def run_ui():
    """Run the visualization UI."""
    from visualization.ui import TSPVisualizer

    print("\nStarting TSP Visualizer...")
    app = TSPVisualizer()
    app.run()


def run_unit_tests():
    """Run the standard-library unittest suite."""
    import subprocess

    result = subprocess.run([
        sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"
    ])
    if result.returncode != 0:
        print("Unit tests failed.")


def run_quick_test():
    """Run the quick smoke test."""
    from quick_test import main as quick_test_main

    quick_test_main()


def main():
    """Run the interactive menu."""
    actions = {
        '1': run_quick_test,
        '2': run_ui,
        '3': test_search,
        '4': test_ml,
        '5': test_csp,
        '6': run_unit_tests,
    }

    while True:
        print_menu()
        choice = input("Enter your choice (0-6): ")
        if choice == '0':
            print("\nGoodbye!")
            return

        action = actions.get(choice)
        if action is None:
            print("\nInvalid choice. Please enter 0-6.")
        else:
            try:
                action()
            except Exception as exc:
                print(f"\nAction failed: {exc}")

        if choice != '2':
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
