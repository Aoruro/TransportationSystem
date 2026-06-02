"""
CSP Module Tests

Test coverage:
- TSPTW instance creation
- Time window constraints
- CSP solver functionality
- Solution verification
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from csp.tsptw_solver import (
    TimeWindow, TSPTWInstance, CSPSolver, TSPTWSolver,
    generate_random_twpt_instance
)


class TestTimeWindow(unittest.TestCase):
    """Time Window Tests"""

    def test_contains(self):
        """Test time window contains method"""
        tw = TimeWindow(5.0, 10.0)

        self.assertTrue(tw.contains(5.0))
        self.assertTrue(tw.contains(7.5))
        self.assertTrue(tw.contains(10.0))
        self.assertFalse(tw.contains(4.9))
        self.assertFalse(tw.contains(10.1))

    def test_wait_time(self):
        """Test wait time calculation"""
        tw = TimeWindow(5.0, 10.0)

        self.assertAlmostEqual(tw.wait_time(3.0), 2.0)
        self.assertAlmostEqual(tw.wait_time(6.0), 0.0)
        self.assertAlmostEqual(tw.wait_time(12.0), 0.0)


class TestTSPTWInstance(unittest.TestCase):
    """TSPTW Instance Tests"""

    def setUp(self):
        np.random.seed(42)
        self.n = 5
        self.coords = np.random.rand(self.n, 2) * 100
        self.time_windows = [(0, 100), (10, 60), (20, 80), (15, 70), (5, 60)]
        self.instance = TSPTWInstance(self.coords, self.time_windows)

    def test_creation(self):
        """Test instance creation"""
        self.assertEqual(self.instance.n, self.n)
        self.assertEqual(len(self.instance.time_windows), self.n)

    def test_distance_matrix(self):
        """Test distance matrix computation"""
        dist = self.instance.dist_matrix
        self.assertEqual(dist.shape, (self.n, self.n))
        np.testing.assert_array_equal(np.diag(dist), 0)

    def test_travel_time(self):
        """Test travel time calculation"""
        time = self.instance.travel_time(0, 1)
        expected = np.linalg.norm(self.coords[0] - self.coords[1])
        self.assertAlmostEqual(time, expected)

    def test_is_feasible_true(self):
        """Test feasible move"""
        feasible = self.instance.is_feasible(0.0, 0, 1)
        self.assertTrue(feasible)

    def test_is_feasible_false(self):
        """Test infeasible move"""
        feasible = self.instance.is_feasible(100.0, 0, 1)
        self.assertFalse(feasible)

    def test_rejects_invalid_time_window(self):
        """Test ordered finite time-window validation"""
        with self.assertRaises(ValueError):
            TSPTWInstance(self.coords, [(0, 100)] * 4 + [(10, 5)])

    def test_rejects_negative_service_time(self):
        """Test non-negative service-time validation"""
        with self.assertRaises(ValueError):
            TSPTWInstance(self.coords, self.time_windows, [0, 0, -1, 0, 0])


class TestCSPSolver(unittest.TestCase):
    """CSP Solver Tests"""

    def setUp(self):
        np.random.seed(42)
        self.n = 5
        coords = np.random.rand(self.n, 2) * 50
        time_windows = [(0, 100)] * self.n
        self.instance = TSPTWInstance(coords, time_windows)

    def test_solve(self):
        """Test solving capability"""
        solver = CSPSolver(self.instance)
        result = solver.solve()

        self.assertIn('success', result)
        self.assertIn('cost', result)
        self.assertIn('time', result)

    def test_verify_solution_valid(self):
        """Test valid solution verification"""
        solver = CSPSolver(self.instance)
        result = solver.solve()

        if result['success']:
            is_valid, msg = solver.verify_solution(result['path'])
            self.assertTrue(is_valid)

    def test_verify_solution_invalid(self):
        """Test invalid solution verification"""
        solver = CSPSolver(self.instance)
        is_valid, msg = solver.verify_solution([0, 1, 2])

        self.assertFalse(is_valid)

    def test_service_times_are_enforced_during_search(self):
        """Test that the solver does not return a route invalidated by service time"""
        coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
        instance = TSPTWInstance(coords, [(0, 100), (0, 2), (0, 3)], [0, 5, 0])

        result = CSPSolver(instance).solve()

        self.assertFalse(result['success'])

    def test_city_limit_is_enforced(self):
        """Test documented CSP scale limit"""
        coords = np.column_stack((np.arange(11, dtype=float), np.zeros(11)))
        instance = TSPTWInstance(coords, [(0, 100)] * 11)

        with self.assertRaises(ValueError):
            CSPSolver(instance)


class TestTSPTWSolver(unittest.TestCase):
    """TSPTW Solver Tests"""

    def setUp(self):
        np.random.seed(42)
        self.n = 6
        coords = np.random.rand(self.n, 2) * 50
        time_windows = [(0, 80), (10, 60), (15, 70), (5, 50), (20, 65), (8, 55)]
        self.instance = TSPTWInstance(coords, time_windows)

    def test_solve(self):
        """Test TSPTW solving"""
        solver = TSPTWSolver(self.instance)
        result = solver.solve()

        self.assertIn('success', result)
        self.assertIn('path', result)
        self.assertIn('cost', result)

    def test_solution_feasibility(self):
        """Test solution feasibility"""
        solver = TSPTWSolver(self.instance)
        result = solver.solve()

        if result['success']:
            is_valid, msg = solver.verify_solution(result['path'])
            self.assertTrue(is_valid, msg)


class TestRandomInstanceGeneration(unittest.TestCase):
    """Random Instance Generation Tests"""

    def test_generate_instance(self):
        """Test single instance generation"""
        instance = generate_random_twpt_instance(n_cities=8, seed=42)

        self.assertEqual(instance.n, 8)
        self.assertEqual(len(instance.time_windows), 8)

    def test_generate_multiple(self):
        """Test multiple instance generation"""
        instances = [generate_random_twpt_instance(n_cities=6, seed=i)
                    for i in range(5)]

        self.assertEqual(len(instances), 5)
        for inst in instances:
            self.assertEqual(inst.n, 6)


if __name__ == '__main__':
    unittest.main()
