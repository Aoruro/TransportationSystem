"""
Breadth-first search for small weighted TSP instances.

Reference notes:
- General BFS state-space search is covered by Russell and Norvig.
- Full bibliographic details and project reuse notes are in REFERENCES.md.
"""

from collections import deque
from typing import Callable, Dict, List, Optional, Tuple
import time

import numpy as np

from .utils import TSPState, validate_distance_matrix


StateKey = Tuple[int, int]


class BFSSolver:
    """
    BFS solver for TSP.

    States are expanded level by level. Because TSP edges are weighted, the
    solver keeps the cheapest path for each (visited_mask, current_city) state
    and checks every reachable complete state before returning the best tour.
    """

    MAX_N = 10

    def __init__(self, dist_matrix: np.ndarray, start: int = 0):
        self.dist = validate_distance_matrix(dist_matrix, start)
        self.n = len(self.dist)
        self.start = start

        if self.n > self.MAX_N:
            raise ValueError(f"BFS only supports N <= {self.MAX_N}, but current N = {self.n}")

    def _initial_state(self) -> TSPState:
        return TSPState(1 << self.start, self.start, 0.0, [self.start])

    @staticmethod
    def _key(state: TSPState) -> StateKey:
        return state.visited_mask, state.current_city

    def _result(self, path: Optional[List[int]], cost: float,
                nodes_expanded: int, start_time: float) -> Dict:
        return {
            'path': path or [],
            'cost': cost,
            'nodes_expanded': nodes_expanded,
            'time': time.time() - start_time,
            'success': path is not None
        }

    def _run(self, callback: Optional[Callable] = None,
             max_nodes: Optional[int] = None) -> Tuple[Dict, List[Dict]]:
        start_time = time.time()
        initial = self._initial_state()
        initial_key = self._key(initial)

        queue = deque([initial_key])
        best_states = {initial_key: initial}
        parents: Dict[StateKey, Tuple[Optional[StateKey], int]] = {
            initial_key: (None, 0)
        }
        expanded_indices: Dict[StateKey, int] = {}
        search_tree: List[Dict] = []
        best_path = None
        best_cost = float('inf')

        while queue and (max_nodes is None or len(search_tree) < max_nodes):
            key = queue.popleft()
            state = best_states[key]
            parent_key, depth = parents[key]
            current_idx = len(search_tree)
            expanded_indices[key] = current_idx
            node_info = {
                'state': state,
                'parent': expanded_indices.get(parent_key, -1),
                'depth': depth,
                'idx': current_idx
            }
            search_tree.append(node_info)

            if callback:
                callback(node_info)

            if state.is_complete(self.n):
                total_cost = state.g_cost + self.dist[state.current_city][self.start]
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_path = state.path + [self.start]
                continue

            if state.g_cost >= best_cost:
                continue

            for next_city in state.get_unvisited(self.n):
                new_state = TSPState(
                    state.visited_mask | (1 << next_city),
                    next_city,
                    state.g_cost + self.dist[state.current_city][next_city],
                    state.path + [next_city]
                )
                new_key = self._key(new_state)
                previous = best_states.get(new_key)

                if previous is None:
                    best_states[new_key] = new_state
                    parents[new_key] = (key, depth + 1)
                    queue.append(new_key)
                elif new_state.g_cost < previous.g_cost:
                    best_states[new_key] = new_state
                    parents[new_key] = (key, depth + 1)

        return self._result(best_path, best_cost, len(search_tree), start_time), search_tree

    def solve(self) -> Dict:
        """Return the lowest-cost tour found by breadth-first expansion."""
        result, _ = self._run()
        return result

    def solve_with_callback(self, callback=None) -> Dict:
        """Solve and invoke callback after each expanded state."""
        result, search_tree = self._run(callback=callback)
        result['search_tree'] = search_tree
        return result

    def get_search_tree(self, max_nodes: int = 10000) -> List[Dict]:
        """Return up to max_nodes expanded states for visualization."""
        _, search_tree = self._run(max_nodes=max_nodes)
        return search_tree

    def prepare_for_iteration(self):
        """Prepare the incremental BFS generator used by the UI."""
        initial = self._initial_state()
        initial_key = self._key(initial)
        self._search_start_time = time.time()
        self._search_queue = deque([initial_key])
        self._search_best_states = {initial_key: initial}
        self._search_parents = {initial_key: (None, 0)}
        self._search_expanded_indices = {}
        self._search_node_index = 0
        self._search_best_path = None
        self._search_best_cost = float('inf')
        self._search_result = None

    def search_generator(self):
        """Yield BFS expansions one by one for visualization."""
        while self._search_queue:
            key = self._search_queue.popleft()
            state = self._search_best_states[key]
            parent_key, depth = self._search_parents[key]
            current_idx = self._search_node_index
            self._search_node_index += 1
            self._search_expanded_indices[key] = current_idx

            node_info = {
                'state': state,
                'parent': self._search_expanded_indices.get(parent_key, -1),
                'depth': depth,
                'idx': current_idx
            }
            yield node_info

            if state.is_complete(self.n):
                total_cost = state.g_cost + self.dist[state.current_city][self.start]
                if total_cost < self._search_best_cost:
                    self._search_best_cost = total_cost
                    self._search_best_path = state.path + [self.start]
                continue

            if state.g_cost >= self._search_best_cost:
                continue

            for next_city in state.get_unvisited(self.n):
                new_state = TSPState(
                    state.visited_mask | (1 << next_city),
                    next_city,
                    state.g_cost + self.dist[state.current_city][next_city],
                    state.path + [next_city]
                )
                new_key = self._key(new_state)
                previous = self._search_best_states.get(new_key)

                if previous is None:
                    self._search_best_states[new_key] = new_state
                    self._search_parents[new_key] = (key, depth + 1)
                    self._search_queue.append(new_key)
                elif new_state.g_cost < previous.g_cost:
                    self._search_best_states[new_key] = new_state
                    self._search_parents[new_key] = (key, depth + 1)

        self._search_result = self._result(
            self._search_best_path,
            self._search_best_cost,
            self._search_node_index,
            self._search_start_time
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
