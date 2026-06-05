"""
Learning-enhanced A* Algorithm Module

Integrates machine learning model into A* search.

Formula: f' = f - lambda * prob

Guarantees heuristic admissibility.

Reference notes:
- This module builds on the A* implementation in search.astar.
- scikit-learn is used for the project-specific prediction model.
- Full bibliographic details and reuse rationale are in REFERENCES.md.
"""

import numpy as np
from typing import List, Dict, Optional, Callable, Tuple
import heapq
import time

from search.utils import TSPState, prim_mst, validate_distance_matrix
from .model import TSPMLModel, ModelCache, FeatureExtractor


class LearningAStar:
    """
    Learning-enhanced A* Solver
    
    Enhances A* search with ML model predictions.
    
    Formula: f' = g + h' = g + h - lambda * prob
    
    Where:
    - g: Actual accumulated cost
    - h: MST lower bound heuristic
    - prob: ML model predicted probability of "good action"
    - lambda: Tuning parameter (0 = standard A*)
    """

    def __init__(self, dist_matrix: np.ndarray, model: TSPMLModel = None,
                 lambda_param: float = 0.5, start: int = 0):
        """
        Initialize learning-enhanced A* solver.
        
        Args:
            dist_matrix: NxN distance matrix
            model: Trained ML model (optional)
            lambda_param: Weighting parameter for ML guidance
            start: Starting city index
        """
        self.dist = validate_distance_matrix(dist_matrix, start)
        self.n = len(self.dist)
        self.start = start
        self.full_mask = (1 << self.n) - 1

        self.model = model
        self.lambda_param = lambda_param
        self.feature_extractor = FeatureExtractor(self.n)

        if model is not None:
            self.model_cache = ModelCache(model)
        else:
            self.model_cache = None

    def _base_heuristic(self, mask: int, current: int) -> float:
        """
        Base MST heuristic.
        
        h = min_dist(current, unvisited) + mst(unvisited) + min_dist_to_start
        
        Args:
            mask: Bitmask of visited cities
            current: Current city index
            
        Returns:
            Heuristic value
        """
        unvisited = [i for i in range(self.n) if not (mask & (1 << i))]

        if not unvisited:
            return self.dist[current][self.start]

        min_to_unvisited = min(self.dist[current][u] for u in unvisited)

        mst_cost = self._compute_mst(unvisited)

        min_to_start = min(self.dist[u][self.start] for u in unvisited)

        return min_to_unvisited + mst_cost + min_to_start

    def _compute_mst(self, unvisited: List[int]) -> float:
        """Compute MST for subset of unvisited cities."""
        if len(unvisited) <= 1:
            return 0.0

        sub_matrix = np.array([[self.dist[u][v] for v in unvisited] for u in unvisited])
        return prim_mst(len(unvisited), sub_matrix)

    def _ml_heuristic(self, mask: int, current: int, path: List[int],
                      cost_so_far: float, next_city: int = None) -> Tuple[float, float]:
        """
        ML-enhanced heuristic.
        
        f' = f - lambda * prob
        
        Returns:
            Tuple of (adjusted heuristic value, raw probability)
        """
        base_h = self._base_heuristic(mask, current)

        if mask == self.full_mask or self.model is None or self.model_cache is None:
            return base_h, 0.0

        features = self.feature_extractor.extract_with_next(
            path, self.dist, next_city if next_city is not None else current, cost_so_far
        )
        prob = self.model_cache.predict(features)

        adjusted_h = base_h - self.lambda_param * prob

        return max(adjusted_h, 0.0), prob

    def _path_to_mask(self, path: List[int]) -> int:
        """Convert path to bitmask."""
        mask = 0
        for city in path:
            mask |= (1 << city)
        return mask

    def solve(self) -> Dict:
        """
        Execute learning-enhanced A* search.
        
        Returns:
            Result dictionary with path, cost, nodes_expanded, time, success
        """
        start_time = time.time()
        start_mask = 1 << self.start

        initial_state = TSPState(
            visited_mask=start_mask,
            current_city=self.start,
            g_cost=0.0,
            path=[self.start]
        )

        h0, _ = self._ml_heuristic(start_mask, self.start, [self.start], 0.0)

        heap = [(h0, 0.0, start_mask, self.start, [self.start])]
        g_dict = {(start_mask, self.start): 0.0}
        nodes_expanded = 0
        counter = 0

        while heap:
            f_val, g_val, mask, current, path = heapq.heappop(heap)
            counter += 1

            if mask == self.full_mask:
                final_path = path + [self.start]
                final_cost = g_val + self.dist[current][self.start]
                elapsed = time.time() - start_time

                cache_stats = {}
                if self.model_cache:
                    cache_stats = self.model_cache.get_stats()

                return {
                    'path': final_path,
                    'cost': final_cost,
                    'nodes_expanded': counter,
                    'time': elapsed,
                    'success': True,
                    'cache_stats': cache_stats
                }

            if (mask, current) in g_dict and g_dict[(mask, current)] < g_val:
                continue

            nodes_expanded += 1
            unvisited = [i for i in range(self.n) if not (mask & (1 << i))]

            for next_city in unvisited:
                new_mask = mask | (1 << next_city)
                new_g = g_val + self.dist[current][next_city]
                new_path = path + [next_city]

                if (new_mask, next_city) not in g_dict or new_g < g_dict[(new_mask, next_city)]:
                    g_dict[(new_mask, next_city)] = new_g

                    h_new, prob = self._ml_heuristic(
                        new_mask, next_city, path, g_val, next_city
                    )
                    new_f = new_g + h_new

                    heapq.heappush(heap, (new_f, new_g, new_mask, next_city, new_path))

        elapsed = time.time() - start_time
        return {
            'path': [],
            'cost': float('inf'),
            'nodes_expanded': nodes_expanded,
            'time': elapsed,
            'success': False
        }

    def solve_with_callback(self, callback: Callable = None) -> Dict:
        """
        Solve with callback for visualization.
        
        Args:
            callback: Function called after each node expansion
            
        Returns:
            Result dictionary with additional search_tree
        """
        start_time = time.time()
        start_mask = 1 << self.start

        initial_state = TSPState(
            visited_mask=start_mask,
            current_city=self.start,
            g_cost=0.0,
            path=[self.start]
        )

        h0, _ = self._ml_heuristic(start_mask, self.start, [self.start], 0.0)

        heap = [(h0, 0.0, start_mask, self.start, [self.start], -1, 0)]
        g_dict = {(start_mask, self.start): 0.0}
        nodes_expanded = 0
        counter = 0
        search_tree = []

        while heap:
            f_val, g_val, mask, current, path, parent_idx, depth = heapq.heappop(heap)
            counter += 1

            current_idx = len(search_tree)
            search_tree.append({
                'state': TSPState(mask, current, g_val, path),
                'parent': parent_idx,
                'depth': depth,
                'idx': current_idx,
                'f': f_val,
                'g': g_val
            })

            if callback:
                callback(search_tree[-1])

            if mask == self.full_mask:
                final_path = path + [self.start]
                final_cost = g_val + self.dist[current][self.start]
                elapsed = time.time() - start_time

                return {
                    'path': final_path,
                    'cost': final_cost,
                    'nodes_expanded': counter,
                    'time': elapsed,
                    'success': True,
                    'search_tree': search_tree
                }

            if (mask, current) in g_dict and g_dict[(mask, current)] < g_val:
                continue

            nodes_expanded += 1
            unvisited = [i for i in range(self.n) if not (mask & (1 << i))]

            for next_city in unvisited:
                new_mask = mask | (1 << next_city)
                new_g = g_val + self.dist[current][next_city]
                new_path = path + [next_city]

                if (new_mask, next_city) not in g_dict or new_g < g_dict[(new_mask, next_city)]:
                    g_dict[(new_mask, next_city)] = new_g

                    h_new, prob = self._ml_heuristic(
                        new_mask, next_city, path, g_val, next_city
                    )
                    new_f = new_g + h_new

                    heapq.heappush(heap, (new_f, new_g, new_mask, next_city, new_path, current_idx, depth + 1))

        elapsed = time.time() - start_time
        return {
            'path': [],
            'cost': float('inf'),
            'nodes_expanded': nodes_expanded,
            'time': elapsed,
            'success': False,
            'search_tree': search_tree
        }

    def prepare_for_iteration(self):
        """Prepare solver for iterative search"""
        self._search_heap = []
        self._search_g_dict = {}
        self._search_counter = 0
        self._search_result = None
        self._search_start_time = time.time()
        
        start_mask = 1 << self.start

        h0, _ = self._ml_heuristic(start_mask, self.start, [self.start], 0.0)
        heapq.heappush(self._search_heap, (h0, 0.0, start_mask, self.start, [self.start], -1, 0))
        self._search_g_dict[(start_mask, self.start)] = 0.0

    def search_generator(self):
        """Generator that yields nodes one by one for visualization"""
        while self._search_heap:
            f_val, g_val, mask, current, path, parent_idx, depth = heapq.heappop(self._search_heap)
            
            current_idx = self._search_counter
            self._search_counter += 1
            
            node_info = {
                'state': TSPState(mask, current, g_val, path),
                'parent': parent_idx,
                'depth': depth,
                'idx': current_idx,
                'f': f_val,
                'g': g_val
            }
            
            # Check completion before expanding successors.
            if mask == self.full_mask:
                final_path = path + [self.start]
                final_cost = g_val + self.dist[current][self.start]
                self._search_result = {
                    'path': final_path,
                    'cost': final_cost,
                    'nodes_expanded': self._search_counter,
                    'time': time.time() - self._search_start_time,
                    'success': True
                }
                node_info['is_complete'] = True
                yield node_info
                return
            
            yield node_info

            if (mask, current) in self._search_g_dict and self._search_g_dict[(mask, current)] < g_val:
                continue

            unvisited = [i for i in range(self.n) if not (mask & (1 << i))]

            for next_city in unvisited:
                new_mask = mask | (1 << next_city)
                new_g = g_val + self.dist[current][next_city]
                new_path = path + [next_city]

                if (new_mask, next_city) not in self._search_g_dict or new_g < self._search_g_dict[(new_mask, next_city)]:
                    self._search_g_dict[(new_mask, next_city)] = new_g

                    h_new, prob = self._ml_heuristic(
                        new_mask, next_city, path, g_val, next_city
                    )
                    new_f = new_g + h_new

                    heapq.heappush(self._search_heap, (new_f, new_g, new_mask, next_city, new_path, current_idx, depth + 1))

        self._search_result = {
            'path': [],
            'cost': float('inf'),
            'nodes_expanded': self._search_counter,
            'time': time.time() - self._search_start_time,
            'success': False
        }

    def get_result(self) -> Dict:
        """Get the search result after iteration completes"""
        return self._search_result if self._search_result else {
            'path': [],
            'cost': float('inf'),
            'nodes_expanded': 0,
            'time': 0,
            'success': False
        }
