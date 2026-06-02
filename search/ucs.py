"""Uniform-cost search for weighted TSP instances."""

import heapq
from itertools import count
from typing import Callable, Dict, List, Optional, Tuple
import time

import numpy as np

from .utils import TSPState, validate_distance_matrix


StateKey = Tuple[int, int]


class UCSSolver:
    """Uniform-cost search with state dominance and lazy heap deletion."""

    MAX_N = 12

    def __init__(self, dist_matrix: np.ndarray, start: int = 0):
        self.dist = validate_distance_matrix(dist_matrix, start)
        self.n = len(self.dist)
        self.start = start
        self.full_mask = (1 << self.n) - 1

        if self.n > self.MAX_N:
            raise ValueError(f"UCS only supports N <= {self.MAX_N}, but current N = {self.n}")

    def _initial_state(self) -> TSPState:
        return TSPState(1 << self.start, self.start, 0.0, [self.start])

    @staticmethod
    def _key(state: TSPState) -> StateKey:
        return state.visited_mask, state.current_city

    def _priority(self, state: TSPState) -> float:
        """Include the closing edge when a state represents a complete tour."""
        if state.visited_mask == self.full_mask:
            return state.g_cost + self.dist[state.current_city][self.start]
        return state.g_cost

    @staticmethod
    def _failure(nodes_expanded: int, start_time: float) -> Dict:
        return {
            'path': [],
            'cost': float('inf'),
            'nodes_expanded': nodes_expanded,
            'time': time.time() - start_time,
            'success': False
        }

    def _success(self, state: TSPState, nodes_expanded: int,
                 start_time: float) -> Dict:
        return {
            'path': state.path + [self.start],
            'cost': state.g_cost + self.dist[state.current_city][self.start],
            'nodes_expanded': nodes_expanded,
            'time': time.time() - start_time,
            'success': True
        }

    def _run(self, callback: Optional[Callable] = None,
             max_nodes: Optional[int] = None,
             stop_at_goal: bool = True) -> Tuple[Dict, List[Dict]]:
        start_time = time.time()
        initial = self._initial_state()
        sequence = count()
        heap = [(self._priority(initial), next(sequence), initial, -1, 0)]
        best_g = {self._key(initial): 0.0}
        search_tree: List[Dict] = []
        best_result = None

        while heap and (max_nodes is None or len(search_tree) < max_nodes):
            _, _, state, parent_idx, depth = heapq.heappop(heap)
            if state.g_cost > best_g.get(self._key(state), float('inf')):
                continue

            current_idx = len(search_tree)
            node_info = {
                'state': state,
                'parent': parent_idx,
                'depth': depth,
                'idx': current_idx
            }
            search_tree.append(node_info)

            if callback:
                callback(node_info)

            if state.is_complete(self.n):
                result = self._success(state, len(search_tree), start_time)
                if best_result is None or result['cost'] < best_result['cost']:
                    best_result = result
                if stop_at_goal:
                    return result, search_tree
                continue

            for next_city in state.get_unvisited(self.n):
                new_state = TSPState(
                    state.visited_mask | (1 << next_city),
                    next_city,
                    state.g_cost + self.dist[state.current_city][next_city],
                    state.path + [next_city]
                )
                new_key = self._key(new_state)

                if new_state.g_cost < best_g.get(new_key, float('inf')):
                    best_g[new_key] = new_state.g_cost
                    heapq.heappush(
                        heap,
                        (self._priority(new_state), next(sequence),
                         new_state, current_idx, depth + 1)
                    )

        return best_result or self._failure(len(search_tree), start_time), search_tree

    def solve(self) -> Dict:
        """Return the optimal tour using uniform-cost expansion."""
        result, _ = self._run()
        return result

    def solve_with_callback(self, callback=None) -> Dict:
        """Solve and invoke callback after each expanded state."""
        result, search_tree = self._run(callback=callback)
        result['search_tree'] = search_tree
        return result

    def get_search_tree(self, max_nodes: int = 10000) -> List[Dict]:
        """Return up to max_nodes expanded states for visualization."""
        _, search_tree = self._run(max_nodes=max_nodes, stop_at_goal=False)
        return search_tree

    def prepare_for_iteration(self):
        """Prepare the incremental UCS generator used by the UI."""
        initial = self._initial_state()
        self._search_start_time = time.time()
        self._search_heap = []
        self._search_sequence = count()
        self._search_best_g = {self._key(initial): 0.0}
        self._search_node_index = 0
        self._search_result = None
        heapq.heappush(
            self._search_heap,
            (self._priority(initial), next(self._search_sequence), initial, -1, 0)
        )

    def search_generator(self):
        """Yield UCS expansions one by one for visualization."""
        while self._search_heap:
            _, _, state, parent_idx, depth = heapq.heappop(self._search_heap)
            if state.g_cost > self._search_best_g.get(self._key(state), float('inf')):
                continue

            current_idx = self._search_node_index
            self._search_node_index += 1
            node_info = {
                'state': state,
                'parent': parent_idx,
                'depth': depth,
                'idx': current_idx
            }

            if state.is_complete(self.n):
                self._search_result = self._success(
                    state, self._search_node_index, self._search_start_time
                )
                node_info['is_complete'] = True
                yield node_info
                return

            yield node_info

            for next_city in state.get_unvisited(self.n):
                new_state = TSPState(
                    state.visited_mask | (1 << next_city),
                    next_city,
                    state.g_cost + self.dist[state.current_city][next_city],
                    state.path + [next_city]
                )
                new_key = self._key(new_state)

                if new_state.g_cost < self._search_best_g.get(new_key, float('inf')):
                    self._search_best_g[new_key] = new_state.g_cost
                    heapq.heappush(
                        self._search_heap,
                        (self._priority(new_state), next(self._search_sequence),
                         new_state, current_idx, depth + 1)
                    )

        self._search_result = self._failure(
            self._search_node_index, self._search_start_time
        )

    def get_result(self) -> Dict:
        """Get the result after incremental search completes."""
        return self._search_result or {
            'path': [],
            'cost': float('inf'),
            'nodes_expanded': 0,
            'time': 0,
            'success': False
        }
