"""Experiment runner regression tests."""

import unittest
import json
import os
import tempfile

import numpy as np

from data.data_processor import TSPInstance
from experiments import ExperimentRunner
from experiments.runner import ExperimentResult


class TestExperimentRunner(unittest.TestCase):
    """Experiment metric and algorithm selection tests."""

    def setUp(self):
        coords = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
        ])
        self.instance = TSPInstance("square", coords)

    def test_astar_uses_independent_optimal_baseline(self):
        """Test reference metrics for a successful exact search."""
        result = ExperimentRunner(instances=[self.instance]).run_single_experiment(
            self.instance, "astar"
        )

        self.assertAlmostEqual(result.cost, result.optimal_cost)
        self.assertTrue(result.optimality_maintained)
        self.assertEqual(result.relative_error, 0.0)

    def test_unsupported_bfs_is_not_relabelled_astar(self):
        """Test that over-limit BFS is reported as unsupported."""
        coords = np.column_stack((np.arange(11, dtype=float), np.zeros(11)))
        instance = TSPInstance("eleven", coords)

        result = ExperimentRunner(instances=[instance]).run_single_experiment(instance, "bfs")

        self.assertFalse(result.optimality_maintained)
        self.assertEqual(result.path, [])
        self.assertEqual(result.algorithm, "bfs")

    def test_learning_astar_requires_trained_model(self):
        """Test that ML experiments cannot silently become standard A*."""
        result = ExperimentRunner(instances=[self.instance]).run_single_experiment(
            self.instance, "learning_astar_rf"
        )

        self.assertFalse(result.optimality_maintained)
        self.assertEqual(result.path, [])

    def test_save_results_uses_standard_json(self):
        """Test that failed experiments serialize without Infinity values."""
        runner = ExperimentRunner(instances=[self.instance])
        runner.results = [ExperimentResult(
            algorithm="bfs",
            instance_name="failed",
            n_cities=11,
            path=[],
            cost=float('inf'),
            optimal_cost=1.0,
            nodes_expanded=0,
            time_seconds=0.0,
            optimality_maintained=False,
            relative_error=float('inf'),
        )]

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
            path = handle.name
        try:
            runner.save_results(path)
            with open(path) as handle:
                data = json.load(handle)
            self.assertIsNone(data[0]['cost'])
            self.assertIsNone(data[0]['relative_error'])

            runner.load_results(path)
            self.assertTrue(np.isinf(runner.results[0].cost))
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
