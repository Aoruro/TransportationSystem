"""
Search Algorithm Module Tests

Test coverage:
- BFS correctness and optimality
- UCS correctness and optimality
- A* correctness and optimality
- MST heuristic
- Edge cases
"""

import unittest
from unittest.mock import patch
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.bfs import BFSSolver
from search.ucs import UCSSolver
from search.astar import AStarSolver
from search.utils import (
    TSPState, verify_solution, held_karp, verify_optimality,
    prim_mst, compute_complexity
)


WEIGHTED_COUNTEREXAMPLE = np.array([
    [0, 1, 1, 100],
    [1, 0, 1, 4],
    [1, 1, 0, 1],
    [100, 4, 1, 0]
], dtype=float)


class TestTSPState(unittest.TestCase):
    """TSP State Tests"""

    def test_state_creation(self):
        """Test state creation"""
        state = TSPState(visited_mask=1, current_city=0)
        self.assertEqual(state.visited_mask, 1)
        self.assertEqual(state.current_city, 0)
        self.assertEqual(state.g_cost, 0.0)

    def test_is_complete(self):
        """Test completeness check"""
        state = TSPState(visited_mask=0b1111, current_city=3)
        self.assertTrue(state.is_complete(4))

        state = TSPState(visited_mask=0b0111, current_city=2)
        self.assertFalse(state.is_complete(4))

    def test_get_unvisited(self):
        """Test unvisited cities retrieval"""
        state = TSPState(visited_mask=0b0101, current_city=0)
        unvisited = state.get_unvisited(4)
        self.assertEqual(set(unvisited), {1, 3})

    def test_comparison_with_unrelated_type(self):
        """Test equality follows the Python protocol for unrelated objects."""
        state = TSPState(visited_mask=1, current_city=0)
        self.assertNotEqual(state, object())


class TestBFSSolver(unittest.TestCase):
    """BFS Solver Tests"""

    def setUp(self):
        self.dist = np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [1, 2, 1, 0]
        ])

    def test_small_instance(self):
        """Test small instance"""
        solver = BFSSolver(self.dist)
        result = solver.solve()

        self.assertTrue(result['success'])
        self.assertEqual(len(result['path']), 5)
        self.assertEqual(result['path'][0], result['path'][-1])

    def test_optimality(self):
        """Test optimality guarantee"""
        solver = BFSSolver(self.dist)
        result = solver.solve()

        is_valid, cost, _ = verify_solution(result['path'], self.dist)
        self.assertTrue(is_valid)

        opt_path, opt_cost = held_karp(self.dist)
        self.assertAlmostEqual(cost, opt_cost, places=5)

    def test_limit_n(self):
        """Test size limit enforcement"""
        large_dist = np.random.rand(15, 15)
        large_dist = (large_dist + large_dist.T) / 2
        np.fill_diagonal(large_dist, 0)

        with self.assertRaises(ValueError):
            solver = BFSSolver(large_dist)

    def test_weighted_counterexample(self):
        """Test that BFS keeps the cheapest path for duplicate states"""
        result = BFSSolver(WEIGHTED_COUNTEREXAMPLE).solve()
        _, optimal_cost = held_karp(WEIGHTED_COUNTEREXAMPLE)
        self.assertAlmostEqual(result['cost'], optimal_cost)

    def test_iteration_mode_optimality(self):
        """Test that UI iteration mode uses the optimized BFS logic"""
        solver = BFSSolver(WEIGHTED_COUNTEREXAMPLE)
        solver.prepare_for_iteration()
        list(solver.search_generator())
        self.assertAlmostEqual(solver.get_result()['cost'], 7.0)


class TestUCSSolver(unittest.TestCase):
    """UCS Solver Tests"""

    def setUp(self):
        self.dist = np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [1, 2, 1, 0]
        ])

    def test_small_instance(self):
        """Test small instance"""
        solver = UCSSolver(self.dist)
        result = solver.solve()

        self.assertTrue(result['success'])
        self.assertEqual(result['path'][0], result['path'][-1])

    def test_optimality(self):
        """Test optimality guarantee"""
        solver = UCSSolver(self.dist)
        result = solver.solve()

        is_valid, cost, _ = verify_solution(result['path'], self.dist)
        self.assertTrue(is_valid)

        opt_path, opt_cost = held_karp(self.dist)
        self.assertAlmostEqual(cost, opt_cost, places=5)

    def test_limit_n(self):
        """Test size limit enforcement"""
        large_dist = np.random.rand(15, 15)
        large_dist = (large_dist + large_dist.T) / 2
        np.fill_diagonal(large_dist, 0)

        with self.assertRaises(ValueError):
            solver = UCSSolver(large_dist)

    def test_weighted_counterexample(self):
        """Test that UCS includes the closing edge in goal priority"""
        result = UCSSolver(WEIGHTED_COUNTEREXAMPLE).solve()
        _, optimal_cost = held_karp(WEIGHTED_COUNTEREXAMPLE)
        self.assertAlmostEqual(result['cost'], optimal_cost)

    def test_iteration_mode_optimality(self):
        """Test that UI iteration mode uses the optimized UCS logic"""
        solver = UCSSolver(WEIGHTED_COUNTEREXAMPLE)
        solver.prepare_for_iteration()
        list(solver.search_generator())
        self.assertAlmostEqual(solver.get_result()['cost'], 7.0)


class TestAStarSolver(unittest.TestCase):
    """A* Solver Tests"""

    def setUp(self):
        self.dist = np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [1, 2, 1, 0]
        ])

    def test_small_instance(self):
        """Test small instance"""
        solver = AStarSolver(self.dist)
        result = solver.solve()

        self.assertTrue(result['success'])
        self.assertEqual(result['path'][0], result['path'][-1])

    def test_optimality(self):
        """Test optimality guarantee"""
        solver = AStarSolver(self.dist)
        result = solver.solve()

        is_valid, cost, _ = verify_solution(result['path'], self.dist)
        self.assertTrue(is_valid)

        opt_path, opt_cost = held_karp(self.dist)
        self.assertAlmostEqual(cost, opt_cost, places=5)

    def test_larger_instance(self):
        """Test larger instance (N=10)"""
        np.random.seed(42)
        n = 10
        coords = np.random.rand(n, 2) * 100
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                dist[i][j] = np.linalg.norm(coords[i] - coords[j])

        solver = AStarSolver(dist)
        result = solver.solve()

        self.assertTrue(result['success'])

    def test_heuristic_admissibility(self):
        """Test heuristic admissibility"""
        solver = AStarSolver(self.dist)
        n = len(self.dist)
        full_mask = (1 << n) - 1

        state = TSPState(visited_mask=1, current_city=0, g_cost=0)
        h = solver._heuristic(state.visited_mask, state.current_city)

        _, optimal = held_karp(self.dist)
        self.assertLessEqual(h, optimal)

    def test_mst_heuristic_cache(self):
        """Test that repeated MST subsets reuse cached lower bounds"""
        solver = AStarSolver(self.dist)
        solver._heuristic(0b0011, 0)
        cache_size = len(solver._mst_cache)
        solver._heuristic(0b0011, 1)
        self.assertEqual(len(solver._mst_cache), cache_size)

    def test_rejects_negative_distances(self):
        """Test validation required by cost-based search algorithms"""
        invalid = self.dist.copy()
        invalid[0, 1] = -1

        for solver_class in (BFSSolver, UCSSolver, AStarSolver):
            with self.subTest(solver=solver_class.__name__):
                with self.assertRaises(ValueError):
                    solver_class(invalid)

    def test_callback_mode_optimality(self):
        """Test the callback mode used by visualizations"""
        _, optimal_cost = held_karp(WEIGHTED_COUNTEREXAMPLE)

        for solver_class in (BFSSolver, UCSSolver, AStarSolver):
            with self.subTest(solver=solver_class.__name__):
                seen = []
                result = solver_class(WEIGHTED_COUNTEREXAMPLE).solve_with_callback(seen.append)
                self.assertAlmostEqual(result['cost'], optimal_cost)
                self.assertEqual(len(seen), len(result['search_tree']))

    def test_iteration_mode_optimality(self):
        """Test the A* generator used by the UI"""
        solver = AStarSolver(WEIGHTED_COUNTEREXAMPLE)
        solver.prepare_for_iteration()
        list(solver.search_generator())
        self.assertAlmostEqual(solver.get_result()['cost'], 7.0)

    def test_iteration_mode_reports_elapsed_time(self):
        """Test that UI-style A* iteration reports real elapsed time."""
        solver = AStarSolver(self.dist)

        with patch("search.astar.time.time", side_effect=[10.0, 10.25]):
            solver.prepare_for_iteration()
            list(solver.search_generator())

        self.assertTrue(solver.get_result()['success'])
        self.assertEqual(solver.get_result()['time'], 0.25)


class TestMST(unittest.TestCase):
    """MST Heuristic Tests"""

    def test_prim_mst(self):
        """Test Prim's MST algorithm"""
        dist = np.array([
            [0, 2, 0, 6, 0],
            [2, 0, 3, 8, 5],
            [0, 3, 0, 0, 7],
            [6, 8, 0, 0, 9],
            [0, 5, 7, 9, 0]
        ])

        mst_cost = prim_mst(5, dist)
        self.assertGreater(mst_cost, 0)
        self.assertLess(mst_cost, 100)

    def test_mst_empty(self):
        """Test empty graph"""
        self.assertEqual(prim_mst(0, np.array([])), 0.0)

    def test_mst_single(self):
        """Test single node"""
        self.assertEqual(prim_mst(1, np.array([[0]])), 0.0)


class TestHeldKarp(unittest.TestCase):
    """Held-Karp Algorithm Tests"""

    def setUp(self):
        self.dist = np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [1, 2, 1, 0]
        ])

    def test_basic(self):
        """Test basic functionality"""
        path, cost = held_karp(self.dist)
        self.assertEqual(len(path), 5)
        self.assertEqual(path[0], path[-1])
        self.assertEqual(set(path[:-1]), {0, 1, 2, 3})

    def test_cost(self):
        """Test cost calculation"""
        path, cost = held_karp(self.dist)
        expected_cost = sum(self.dist[path[i]][path[i+1]] for i in range(len(path)-1))
        self.assertAlmostEqual(cost, expected_cost, places=5)

    def test_single_city(self):
        """Test the degenerate one-city tour"""
        path, cost = held_karp(np.array([[0.0]]))
        self.assertEqual(path, [0, 0])
        self.assertEqual(cost, 0.0)

    def test_rejects_non_integer_start(self):
        """Test that bitmask searches reject ambiguous start values."""
        with self.assertRaises(ValueError):
            held_karp(self.dist, start=0.5)


class TestVerifySolution(unittest.TestCase):
    """Solution Verification Tests"""

    def test_valid_solution(self):
        """Test valid solution"""
        path = [0, 1, 2, 3, 0]
        dist = np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [1, 2, 1, 0]
        ])

        is_valid, cost, msg = verify_solution(path, dist)
        self.assertTrue(is_valid)

    def test_invalid_not_cycle(self):
        """Test non-cyclic path"""
        path = [0, 1, 2, 3]
        dist = np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [1, 2, 1, 0]
        ])

        is_valid, cost, msg = verify_solution(path, dist)
        self.assertFalse(is_valid)

    def test_duplicate_city(self):
        """Test path with duplicate cities"""
        path = [0, 1, 2, 1, 0]
        dist = np.array([
            [0, 1, 2, 1],
            [1, 0, 1, 2],
            [2, 1, 0, 1],
            [1, 2, 1, 0]
        ])

        is_valid, cost, msg = verify_solution(path, dist)
        self.assertFalse(is_valid)

    def test_out_of_range_city(self):
        """Test that invalid city identifiers are reported without indexing."""
        dist = np.eye(4)
        is_valid, cost, msg = verify_solution([0, 1, 2, 4, 0], dist)
        self.assertFalse(is_valid)
        self.assertIn("indices", msg)

    def test_non_integer_city(self):
        """Test that float city identifiers are rejected before indexing."""
        dist = np.eye(4)
        is_valid, cost, msg = verify_solution([0, 1, 2.5, 3, 0], dist)
        self.assertFalse(is_valid)
        self.assertIn("integers", msg)

    def test_invalid_distance_matrix(self):
        """Test that malformed matrices produce validation failures."""
        is_valid, cost, msg = verify_solution([0, 0], np.array([[0, 1]]))
        self.assertFalse(is_valid)
        self.assertIn("square", msg)


class TestComplexity(unittest.TestCase):
    """Complexity helper tests"""

    def test_compute_complexity(self):
        """Test NumPy-independent factorial estimates"""
        result = compute_complexity(5, 'bfs')
        self.assertEqual(result['nodes_estimate'], 600)

if __name__ == '__main__':
    unittest.main()
