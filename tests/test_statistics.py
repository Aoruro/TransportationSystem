"""Statistical analysis regression tests."""

import unittest

from experiments.runner import ExperimentResult
from experiments.statistics import StatisticalAnalyzer


def result(algorithm, instance, nodes, cost=1.0):
    """Create a compact experiment result fixture."""
    return ExperimentResult(
        algorithm=algorithm,
        instance_name=instance,
        n_cities=4,
        path=[],
        cost=cost,
        optimal_cost=1.0,
        nodes_expanded=nodes,
        time_seconds=0.1,
        optimality_maintained=True,
        relative_error=0.0,
    )


class TestStatisticalAnalyzer(unittest.TestCase):
    """Statistics boundary behavior."""

    def test_empty_results_have_empty_summary(self):
        """Test that an empty result list is supported."""
        self.assertEqual(StatisticalAnalyzer([]).get_summary_stats(), {})

    def test_paired_test_ignores_non_finite_rows(self):
        """Test that failed experiment values do not poison t-tests."""
        analyzer = StatisticalAnalyzer([
            result("a", "one", 1),
            result("b", "one", 2),
            result("a", "two", 2),
            result("b", "two", 4),
            result("a", "failed", float('inf')),
            result("b", "failed", float('inf')),
        ])

        comparison = analyzer.paired_t_test("a", "b")

        self.assertEqual(comparison.mean1, 1.5)
        self.assertEqual(comparison.mean2, 3.0)

    def test_invalid_metric_is_rejected(self):
        """Test explicit validation for unsupported columns."""
        analyzer = StatisticalAnalyzer([result("a", "one", 1)])
        with self.assertRaises(ValueError):
            analyzer.paired_t_test("a", "b", metric="missing")


if __name__ == '__main__':
    unittest.main()
