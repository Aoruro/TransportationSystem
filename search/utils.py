"""
search module utilities

Contains:
- TSP state representation: (visited_mask, current_city)
- General validation functions
- Optimality verification
- Complexity analysis

Reference notes:
- Held-Karp and Prim's algorithm are implemented locally for TSP utilities.
- Full bibliographic details and reuse rationale are in REFERENCES.md.
"""

import math
import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass


def validate_distance_matrix(dist_matrix: np.ndarray, start: int = 0) -> np.ndarray:
    """Validate search input and return a numeric distance matrix."""
    dist = np.asarray(dist_matrix, dtype=float)

    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("Distance matrix must be a non-empty square matrix")
    if dist.shape[0] == 0:
        raise ValueError("Distance matrix must contain at least one city")
    if isinstance(start, bool) or not isinstance(start, (int, np.integer)):
        raise ValueError("Start city must be an integer")
    if not 0 <= start < dist.shape[0]:
        raise ValueError(f"Start city must be in [0, {dist.shape[0] - 1}]")
    if not np.all(np.isfinite(dist)):
        raise ValueError("Distance matrix must contain only finite values")
    if np.any(dist < 0):
        raise ValueError("Distance matrix must not contain negative distances")

    return dist


@dataclass
class TSPState:
    """
    TSP State Representation Class
    
    Attributes:
        visited_mask: Bitmask representing visited cities 
            (e.g., for N=8, mask=0b00001101 means cities 0, 2, 3 visited)
        current_city: Current city index
        g_cost: Accumulated cost from start to current state
        path: List of visited cities in order
    """
    visited_mask: int
    current_city: int
    g_cost: float = 0.0
    path: List[int] = None

    def __post_init__(self):
        """Initialize path if not provided"""
        if self.path is None:
            self.path = []

    def is_complete(self, n_cities: int) -> bool:
        """Check if all cities have been visited"""
        full_mask = (1 << n_cities) - 1
        return self.visited_mask == full_mask

    def get_unvisited(self, n_cities: int) -> List[int]:
        """Get list of unvisited cities"""
        return [i for i in range(n_cities) if not (self.visited_mask & (1 << i))]

    def __hash__(self):
        """Hash function for state uniqueness"""
        return hash((self.visited_mask, self.current_city))

    def __eq__(self, other):
        """Equality comparison for states"""
        if not isinstance(other, TSPState):
            return NotImplemented
        return (self.visited_mask == other.visited_mask and
                self.current_city == other.current_city)

    def __lt__(self, other):
        """Less than comparison for heapq compatibility"""
        return self.g_cost < other.g_cost


def verify_solution(path: List[int], dist_matrix: np.ndarray,
                   start: int = 0) -> Tuple[bool, float, str]:
    """
    Verify correctness of TSP solution.
    
    Args:
        path: Path list (should start and end with start city)
        dist_matrix: NxN distance matrix
        start: Starting city index (default 0)
        
    Returns:
        Tuple of (is_valid, total_cost, error_message)
    """
    try:
        dist_matrix = validate_distance_matrix(dist_matrix, start)
    except (TypeError, ValueError) as exc:
        return False, 0.0, str(exc)

    if not isinstance(path, (list, tuple, np.ndarray)):
        return False, 0.0, "Path must be a sequence of city indices"

    n = len(dist_matrix)

    if len(path) < 2:
        return False, 0.0, "Path is too short"

    if any(
        isinstance(city, bool)
        or not isinstance(city, (int, np.integer))
        or city < 0
        or city >= n
        for city in path
    ):
        return False, 0.0, f"Path city indices must be integers in [0, {n - 1}]"

    if path[0] != start or path[-1] != start:
        return False, 0.0, f"Path must start and end at city {start}"

    if len(set(path[:-1])) != len(path[:-1]):
        return False, 0.0, "Path contains duplicate cities"

    if len(path) - 1 != n:
        return False, 0.0, f"Path must visit all {n} cities"

    total_cost = 0.0
    for i in range(len(path) - 1):
        total_cost += dist_matrix[path[i]][path[i+1]]

    return True, total_cost, "Valid"


def held_karp(dist_matrix: np.ndarray, start: int = 0) -> Tuple[List[int], float]:
    """
    Held-Karp algorithm: Dynamic programming for TSP optimal solution.
    Suitable for N <= 15.
    
    Args:
        dist_matrix: NxN distance matrix
        start: Starting city index
        
    Returns:
        Tuple of (optimal_path, minimal_cost)
        
    Raises:
        ValueError: If N > 15
    """
    dist_matrix = validate_distance_matrix(dist_matrix, start)
    n = len(dist_matrix)
    if n > 15:
        raise ValueError("Held-Karp only supports N <= 15")
    if n == 1:
        return [start, start], float(dist_matrix[start][start])

    full_mask = (1 << n) - 1

    dp = np.full((1 << n, n), float('inf'))
    parent = np.full((1 << n, n), -1)

    dp[1 << start, start] = 0

    for mask in range(1 << n):
        for last in range(n):
            if not (mask & (1 << last)):
                continue
            if dp[mask, last] == float('inf'):
                continue

            for next_city in range(n):
                if mask & (1 << next_city):
                    continue
                new_mask = mask | (1 << next_city)
                new_cost = dp[mask, last] + dist_matrix[last][next_city]

                if new_cost < dp[new_mask, next_city]:
                    dp[new_mask, next_city] = new_cost
                    parent[new_mask, next_city] = last

    min_cost = float('inf')
    last_city = start

    for city in range(n):
        if city != start:
            cost = dp[full_mask, city] + dist_matrix[city][start]
            if cost < min_cost:
                min_cost = cost
                last_city = city

    path = []
    mask = full_mask
    while last_city != -1:
        path.append(last_city)
        prev = parent[mask, last_city]
        mask = mask ^ (1 << last_city)
        last_city = prev

    path.reverse()
    path.append(start)

    return path, min_cost


def verify_optimality(path: List[int], dist_matrix: np.ndarray,
                     optimal_cost: float = None) -> Tuple[bool, float]:
    """
    Verify optimality of TSP solution.
    
    Args:
        path: Path list to verify
        dist_matrix: NxN distance matrix
        optimal_cost: Known optimal cost (optional)
        
    Returns:
        Tuple of (is_optimal, computed_cost)
    """
    is_valid, cost, _ = verify_solution(path, dist_matrix)
    if not is_valid:
        return False, cost

    if optimal_cost is not None:
        return abs(cost - optimal_cost) < 1e-6, cost

    try:
        opt_path, opt_cost = held_karp(dist_matrix)
        return abs(cost - opt_cost) < 1e-6, cost
    except ValueError:
        return False, cost


def compute_complexity(n_cities: int, algorithm: str) -> Dict[str, float]:
    """
    Compute algorithm complexity analysis.
    
    Args:
        n_cities: Number of cities
        algorithm: Algorithm name ('bfs', 'ucs', 'astar', 'held_karp')
        
    Returns:
        Dictionary with complexity information
    """
    complexities = {
        'bfs': {
            'time': "O(N! * N)", 'space': "O(N! * N)",
            'nodes_estimate': math.factorial(n_cities) * n_cities
        },
        'ucs': {
            'time': "O(N! * N)", 'space': "O(N! * N)",
            'nodes_estimate': math.factorial(n_cities) * n_cities
        },
        'astar': {
            'time': "O(b^d)", 'space': "O(b^d)",
            'nodes_estimate': min(math.factorial(n_cities), 1000000)
        },
        'held_karp': {
            'time': "O(N^2 * 2^N)", 'space': "O(N * 2^N)",
            'nodes_estimate': n_cities * n_cities * (2 ** n_cities)
        }
    }

    result = complexities.get(algorithm, {})
    if 'nodes_estimate' in result:
        result['nodes_estimate'] = min(result['nodes_estimate'], 1e9)
    return result


def prim_mst(n: int, dist_matrix: np.ndarray) -> float:
    """
    Prim's algorithm for minimum spanning tree weight.
    Used in A* heuristic function.
    
    Args:
        n: Number of cities in subset
        dist_matrix: Distance matrix for the subset
        
    Returns:
        MST total weight
    """
    if n <= 0:
        return 0.0

    key = [float('inf')] * n
    in_mst = [False] * n
    key[0] = 0.0
    total = 0.0

    for _ in range(n):
        u = -1
        m = float('inf')
        for i in range(n):
            if not in_mst[i] and key[i] < m:
                m = key[i]
                u = i

        if u == -1:
            break

        in_mst[u] = True
        total += m

        for v in range(n):
            if not in_mst[v] and dist_matrix[u][v] < key[v]:
                key[v] = dist_matrix[u][v]

    return total
