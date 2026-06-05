"""
A* Algorithm for TSP

Limit: N <= 25

Features:
- Uses heuristic function to guide search
- f = g + h, where h is a consistent MST lower bound
- Heuristic: min_dist(current, unvisited) + mst(unvisited) + min_dist_to_start
- Guaranteed optimal solution

Reference notes:
- A* and admissible heuristic search are covered by Russell and Norvig.
- The MST lower-bound term is implemented locally with Prim's algorithm.
- Full bibliographic details and reuse rationale are in REFERENCES.md.
"""

import heapq
from typing import List, Tuple, Dict, Optional
import numpy as np
import time

from .utils import TSPState, validate_distance_matrix


class AStarSolver:
    """
    A* Solver for TSP
    
    Limit: N <= 25
    
    Heuristic function:
    h(state) = min_dist(current, unvisited) + mst(unvisited) + min_dist_to_start
    
    MST computed using Prim's algorithm, ensuring admissibility
    """

    MAX_N = 25

    def __init__(self, dist_matrix: np.ndarray, start: int = 0):
        """
        Initialize A* solver.
        
        Args:
            dist_matrix: NxN distance matrix
            start: Starting city index (default 0)
            
        Raises:
            ValueError: If N > MAX_N
        """
        self.dist = validate_distance_matrix(dist_matrix, start)
        self.n = len(self.dist)
        self.start = start
        self.full_mask = (1 << self.n) - 1
        self._heuristic_cache = {}
        self._mst_cache = {0: 0.0}

        if self.n > self.MAX_N:
            raise ValueError(f"A* only supports N <= {self.MAX_N}, but current N = {self.n}")

    def _heuristic(self, mask: int, current: int) -> float:
        """
        Compute heuristic function value.
        
        h = min_edge_to_unvisited + mst(unvisited) + min_edge_to_start
        
        Args:
            mask: Bitmask of visited cities
            current: Current city index
            
        Returns:
            Heuristic estimate value
        """
        cache_key = (mask, current)
        if cache_key in self._heuristic_cache:
            return self._heuristic_cache[cache_key]

        unvisited_mask = self.full_mask & ~mask
        unvisited = [i for i in range(self.n) if unvisited_mask & (1 << i)]

        if not unvisited:
            result = self.dist[current][self.start]
            self._heuristic_cache[cache_key] = result
            return result

        min_to_unvisited = min(self.dist[current][u] for u in unvisited)

        mst_cost = self._mst_cost(unvisited_mask)

        min_to_start = min(self.dist[u][self.start] for u in unvisited)

        result = min_to_unvisited + mst_cost + min_to_start
        self._heuristic_cache[cache_key] = result
        return result

    def _mst_heuristic(self, unvisited: List[int]) -> float:
        """
        Compute MST lower bound for unvisited cities subset.
        
        Uses Prim's algorithm on the subset of unvisited cities.
        
        Args:
            unvisited: List of unvisited city indices
            
        Returns:
            MST total weight
        """
        mask = 0
        for city in unvisited:
            mask |= 1 << city
        return self._mst_cost(mask)

    def _mst_cost(self, mask: int) -> float:
        """Compute and cache an MST lower bound directly on a city bitmask."""
        if mask in self._mst_cache:
            return self._mst_cache[mask]
        if mask & (mask - 1) == 0:
            self._mst_cache[mask] = 0.0
            return 0.0

        cities = [i for i in range(self.n) if mask & (1 << i)]
        remaining = set(cities[1:])
        min_edges = {
            city: min(self.dist[cities[0]][city], self.dist[city][cities[0]])
            for city in remaining
        }
        total = 0.0

        while remaining:
            city = min(remaining, key=min_edges.get)
            total += min_edges[city]
            remaining.remove(city)

            for other in remaining:
                edge_cost = min(self.dist[city][other], self.dist[other][city])
                if edge_cost < min_edges[other]:
                    min_edges[other] = edge_cost

        self._mst_cache[mask] = total
        return total

    def solve(self) -> Dict:
        """
        Execute A* search for TSP.
        
        Returns:
            Dictionary containing:
            - path: Optimal path (list of city indices)
            - cost: Total path cost
            - nodes_expanded: Number of expanded nodes
            - time: Running time in seconds
            - success: Boolean indicating success
        """
        start_time = time.time()
        start_mask = 1 << self.start

        initial_state = TSPState(
            visited_mask=start_mask,
            current_city=self.start,
            g_cost=0.0,
            path=[self.start]
        )

        h0 = self._heuristic(start_mask, self.start)
        heap = [(h0, 0.0, start_mask, self.start, initial_state.path)]
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

                return {
                    'path': final_path,
                    'cost': final_cost,
                    'nodes_expanded': counter,
                    'time': elapsed,
                    'success': True
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
                    h = self._heuristic(new_mask, next_city)
                    new_f = new_g + h
                    heapq.heappush(heap, (new_f, new_g, new_mask, next_city, new_path))

        elapsed = time.time() - start_time
        return {
            'path': [],
            'cost': float('inf'),
            'nodes_expanded': nodes_expanded,
            'time': elapsed,
            'success': False
        }

    def solve_with_callback(self, callback=None) -> Dict:
        """
        Execute A* search with callback for visualization.

        Args:
            callback: Function called after each node expansion

        Returns:
            Dictionary containing search results and search_tree
        """
        start_time = time.time()
        start_mask = 1 << self.start

        initial_state = TSPState(
            visited_mask=start_mask,
            current_city=self.start,
            g_cost=0.0,
            path=[self.start]
        )

        h0 = self._heuristic(start_mask, self.start)
        heap = [(h0, 0.0, start_mask, self.start, initial_state.path, -1, 0)]
        g_dict = {(start_mask, self.start): 0.0}
        nodes_expanded = 0
        counter = 0
        search_tree = []

        while heap:
            f_val, g_val, mask, current, path, parent_idx, depth = heapq.heappop(heap)
            counter += 1

            current_idx = len(search_tree)
            node_info = {
                'state': TSPState(mask, current, g_val, path),
                'parent': parent_idx,
                'depth': depth,
                'idx': current_idx,
                'f': f_val,
                'g': g_val
            }
            search_tree.append(node_info)

            if callback:
                callback(node_info)

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
                    h = self._heuristic(new_mask, next_city)
                    new_f = new_g + h
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

    def get_search_tree(self, max_nodes: int = 10000) -> List[Dict]:
        """
        Get search tree structure for visualization.
        
        Args:
            max_nodes: Maximum number of nodes to include
            
        Returns:
            List of nodes with state, parent, depth, f-value, and g-value
        """
        start_time = time.time()
        start_mask = 1 << self.start

        initial_state = TSPState(
            visited_mask=start_mask,
            current_city=self.start,
            g_cost=0.0,
            path=[self.start]
        )

        h0 = self._heuristic(start_mask, self.start)
        heap = [(h0, 0.0, start_mask, self.start, initial_state.path, -1, 0)]
        g_dict = {(start_mask, self.start): 0.0}
        nodes_expanded = 0
        search_tree = []

        while heap and nodes_expanded < max_nodes:
            f_val, g_val, mask, current, path, parent_idx, depth = heapq.heappop(heap)

            current_idx = len(search_tree)
            search_tree.append({
                'state': TSPState(mask, current, g_val, path),
                'parent': parent_idx,
                'depth': depth,
                'idx': current_idx,
                'f': f_val,
                'g': g_val
            })
            nodes_expanded += 1

            if mask == self.full_mask:
                continue

            if (mask, current) in g_dict and g_dict[(mask, current)] < g_val:
                continue

            unvisited = [i for i in range(self.n) if not (mask & (1 << i))]

            for next_city in unvisited:
                new_mask = mask | (1 << next_city)
                new_g = g_val + self.dist[current][next_city]
                new_path = path + [next_city]

                if (new_mask, next_city) not in g_dict or new_g < g_dict[(new_mask, next_city)]:
                    g_dict[(new_mask, next_city)] = new_g
                    h = self._heuristic(new_mask, next_city)
                    new_f = new_g + h
                    heapq.heappush(heap, (new_f, new_g, new_mask, next_city, new_path, current_idx, depth + 1))

        return search_tree

    def prepare_for_iteration(self):
        """Prepare solver for iterative search"""
        self._search_heap = []
        self._search_g_dict = {}
        self._search_counter = 0
        self._search_result = None
        self._search_start_time = time.time()
        
        start_mask = 1 << self.start

        initial_state = TSPState(
            visited_mask=start_mask,
            current_city=self.start,
            g_cost=0.0,
            path=[self.start]
        )

        h0 = self._heuristic(start_mask, self.start)
        heapq.heappush(self._search_heap, (h0, 0.0, start_mask, self.start, initial_state.path, -1, 0))
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
                    h = self._heuristic(new_mask, next_city)
                    new_f = new_g + h
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
