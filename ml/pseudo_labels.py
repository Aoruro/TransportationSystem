"""
Pseudo Label Generation Module

Features:
- N <= 15: Use Held-Karp to generate optimal solutions
- N = 16-20: Use standard A* to generate near-optimal solutions
- Negative samples: Select 2-3 successors with highest f-values per state

References:
- Held-Karp algorithm from "An efficient algorithm for the traveling-salesman problem" (Held & Karp, 1962)
- A* algorithm with MST heuristic
"""

import numpy as np
from typing import List, Tuple

from search.utils import held_karp, prim_mst
from search.astar import AStarSolver
from data.data_processor import TSPInstance


class PseudoLabelGenerator:
    """
    Pseudo Label Generator
    
    Generates positive and negative training samples for machine learning.
    
    Positive samples: Extracted from optimal/near-optimal paths
    Negative samples: 2-3 successors with highest f-values per state
    """

    def __init__(self, dist_matrix: np.ndarray):
        """
        Initialize pseudo label generator.
        
        Args:
            dist_matrix: NxN distance matrix
        """
        self.dist_matrix = dist_matrix
        self.n = len(dist_matrix)

    def generate_optimal_solution(self) -> Tuple[List[int], float]:
        """
        Generate optimal or near-optimal solution.
        
        Uses Held-Karp for N <= 15, A* for N > 15.
        
        Returns:
            (path, cost)
        """
        if self.n <= 15:
            return held_karp(self.dist_matrix)
        else:
            solver = AStarSolver(self.dist_matrix)
            result = solver.solve()
            if result['success']:
                return result['path'], result['cost']
            else:
                return self._greedy_solution()

    def _greedy_solution(self) -> Tuple[List[int], float]:
        """Generate greedy solution as fallback."""
        path = [0]
        visited = {0}
        while len(path) < self.n:
            current = path[-1]
            next_city = min(
                [c for c in range(self.n) if c not in visited],
                key=lambda x: self.dist_matrix[current][x]
            )
            path.append(next_city)
            visited.add(next_city)
        path.append(0)
        cost = sum(self.dist_matrix[path[i]][path[i+1]] for i in range(len(path)-1))
        return path, cost

    def generate_training_data(self, path: List[int], cost: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data from path.
        
        Args:
            path: Optimal/near-optimal path
            cost: Total path cost
            
        Returns:
            (Feature matrix X, Label vector y)
        """
        X = []
        y = []

        for pos in range(len(path) - 2):
            current = path[pos]
            good_next = path[pos + 1]

            visited_before = path[:pos+1]
            g_cost = sum(self.dist_matrix[path[i]][path[i+1]] for i in range(pos))

            X.append(self._extract_features(visited_before, good_next, g_cost, cost))
            y.append(1)

            negative_candidates = [
                candidate for candidate in range(self.n)
                if candidate not in visited_before and candidate != good_next
            ]
            negative_candidates.sort(
                key=lambda candidate: self.dist_matrix[current][candidate],
                reverse=True
            )
            for candidate in negative_candidates[:2]:
                X.append(self._extract_features(visited_before, candidate, g_cost, cost))
                y.append(0)

        return np.array(X), np.array(y)

    def _extract_features(self, path: List[int], next_city: int,
                         cost_so_far: float, total_cost: float) -> np.ndarray:
        """
        Extract features for a sample.
        
        Args:
            path: Current path
            next_city: Candidate next city
            cost_so_far: Accumulated cost to current state
            total_cost: Total path cost
            
        Returns:
            Feature vector (10 dimensions)
        """
        n = self.n
        current = path[-1]
        visited = set(path)

        visited_ratio = len(path) / n

        all_dists = self.dist_matrix[current]
        mean_dist = np.mean(all_dists)
        other_dists = np.delete(all_dists, current)
        min_dist = np.min(other_dists) if len(other_dists) else 0.0

        unvisited = [i for i in range(n) if i not in visited]
        unvisited_ratio = len(unvisited) / n

        if unvisited:
            candidate_dists = [self.dist_matrix[current][u] for u in unvisited]
            min_candidate = min(candidate_dists)
            mean_candidate = np.mean(candidate_dists)
            max_candidate = max(candidate_dists)
        else:
            min_candidate = mean_candidate = max_candidate = 0.0

        if len(unvisited) > 1:
            sub_matrix = np.array([[self.dist_matrix[u][v] for v in unvisited] for u in unvisited])
            mst_cost = prim_mst(len(unvisited), sub_matrix)
        else:
            mst_cost = 0.0

        mst_ratio = mst_cost / cost_so_far if cost_so_far > 0 else 0.0

        features = [
            visited_ratio,
            mean_dist,
            min_dist,
            min_candidate,
            mean_candidate,
            unvisited_ratio,
            max_candidate,
            mst_ratio,
            current / n,
            next_city / n
        ]

        return np.array(features, dtype=np.float32)

    def generate_negative_samples(self, path: List[int], g_cost: float,
                                  n_negatives: int = 2) -> List[Tuple[np.ndarray, int]]:
        """
        Generate negative samples.
        
        Args:
            path: Current path
            g_cost: Accumulated cost
            n_negatives: Number of negative samples
            
        Returns:
            List of (features, 0) tuples
        """
        n = self.n
        current = path[-1]
        visited = set(path)
        unvisited = [i for i in range(n) if i not in visited]

        f_values = []
        for city in unvisited:
            edge_cost = self.dist_matrix[current][city]
            f = g_cost + edge_cost
            f_values.append((city, f))

        f_values.sort(key=lambda x: x[1], reverse=True)
        selected = [city for city, _ in f_values[:n_negatives]]

        negatives = []
        for city in selected:
            features = self._extract_features(path, city, g_cost, 0)
            negatives.append((features, 0))

        return negatives


def generate_pseudo_labels_for_instance(inst: TSPInstance) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate pseudo labels for a single TSP instance.
    
    Args:
        inst: TSPInstance object
        
    Returns:
        (Feature matrix, Label vector)
    """
    generator = PseudoLabelGenerator(inst.dist_matrix)
    path, cost = generator.generate_optimal_solution()
    X, y = generator.generate_training_data(path, cost)
    return X, y


def generate_pseudo_labels_batch(instances: List[TSPInstance]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate pseudo labels for multiple instances.
    
    Args:
        instances: List of TSPInstance objects
        
    Returns:
        (Combined feature matrix, Combined label vector)
    """
    all_X = []
    all_y = []

    for inst in instances:
        try:
            X, y = generate_pseudo_labels_for_instance(inst)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Error processing instance {inst.name}: {e}")
            continue

    if not all_X:
        return np.array([]), np.array([])

    return np.vstack(all_X), np.concatenate(all_y)
