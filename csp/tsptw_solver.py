"""
TSPTW (Traveling Salesman Problem with Time Windows) CSP Solver

Limit: N <= 10

Features:
- Implements TSPTW: Each city has time window [earliest, latest]
- CSP solving: Backtracking search + Forward checking + MRV heuristic
- Outputs constraint-satisfying optimal path

References:
- Constraint Satisfaction Problem techniques from "Artificial Intelligence: A Modern Approach"
- MRV (Minimum Remaining Values) heuristic
- Forward checking algorithm
- Full bibliographic details and project reuse notes are in REFERENCES.md.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import time


@dataclass
class TimeWindow:
    """Time Window class"""
    earliest: float
    latest: float

    def contains(self, time: float) -> bool:
        """Check if time is within window"""
        return self.earliest <= time <= self.latest

    def wait_time(self, current_time: float) -> float:
        """Time to wait after arrival before service can start"""
        if current_time < self.earliest:
            return self.earliest - current_time
        return 0.0


class TSPTWInstance:
    """
    TSPTW Problem Instance
    
    Each city has:
    - Coordinates
    - Time window [earliest, latest]
    - Service time (optional, defaults to 0)
    """

    def __init__(self, coords: np.ndarray,
                 time_windows: List[Tuple[float, float]],
                 service_times: List[float] = None):
        """
        Initialize TSPTW instance.
        
        Args:
            coords: Nx2 array of city coordinates
            time_windows: List of (earliest, latest) tuples
            service_times: Optional list of service times per city
        """
        self.coords = np.asarray(coords, dtype=float)
        self.n = len(coords)
        if self.coords.ndim != 2 or self.coords.shape[1] != 2:
            raise ValueError("coords must be an Nx2 array")
        if self.n == 0 or not np.all(np.isfinite(self.coords)):
            raise ValueError("coords must contain finite city coordinates")
        if len(time_windows) != self.n:
            raise ValueError("time_windows must contain one entry per city")
        if service_times is not None and len(service_times) != self.n:
            raise ValueError("service_times must contain one entry per city")

        self.time_windows = [TimeWindow(tw[0], tw[1]) for tw in time_windows]
        if any(
            not np.isfinite(tw.earliest)
            or not np.isfinite(tw.latest)
            or tw.earliest > tw.latest
            for tw in self.time_windows
        ):
            raise ValueError("time windows must be finite and ordered")

        self.service_times = service_times if service_times is not None else [0.0] * self.n
        if any(not np.isfinite(value) or value < 0 for value in self.service_times):
            raise ValueError("service times must be finite and non-negative")

        self.dist_matrix = self._compute_dist_matrix()

    def _compute_dist_matrix(self) -> np.ndarray:
        """Compute Euclidean distance matrix"""
        n = self.n
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(self.coords[i] - self.coords[j])
                dist[i][j] = d
                dist[j][i] = d
        return dist

    def travel_time(self, from_city: int, to_city: int) -> float:
        """Travel time from from_city to to_city"""
        return self.dist_matrix[from_city][to_city]

    def arrival_time(self, current_time: float, from_city: int, to_city: int) -> float:
        """Compute arrival time at to_city (including waiting time)"""
        travel = self.travel_time(from_city, to_city)
        arrival = current_time + travel
        wait = self.time_windows[to_city].wait_time(arrival)
        return arrival + wait

    def is_feasible(self, current_time: float, from_city: int, to_city: int) -> bool:
        """Check if traveling from from_city to to_city is feasible"""
        travel = self.travel_time(from_city, to_city)
        arrival = current_time + travel
        return arrival <= self.time_windows[to_city].latest


class CSPSolver:
    """
    CSP Solver Base Class
    
    Implements:
    - Backtracking search
    - Forward checking
    - MRV (Minimum Remaining Values) heuristic
    """

    MAX_N = 10

    def __init__(self, instance: TSPTWInstance, start_city: int = 0):
        """
        Initialize CSP solver.
        
        Args:
            instance: TSPTWInstance object
            start_city: Starting city index (default 0)
        """
        self.instance = instance
        self.start = start_city
        self.n = instance.n
        if self.n > self.MAX_N:
            raise ValueError(f"CSP only supports N <= {self.MAX_N}, but current N = {self.n}")
        if not 0 <= self.start < self.n:
            raise ValueError(f"Start city must be in [0, {self.n - 1}]")
        self.best_cost = float('inf')
        self.best_path = None

    def _select_unassigned(self, path: List[int], current_time: float) -> List[int]:
        """
        MRV heuristic: Select most constrained next city.
        
        Prioritizes cities with tightest time windows that can still be reached.
        
        Args:
            path: Current path
            current_time: Current time
            
        Returns:
            Sorted list of candidate cities
        """
        visited = set(path)
        unvisited = [i for i in range(self.n) if i not in visited and i != self.start]

        if not unvisited:
            return []

        scored = []
        for city in unvisited:
            tw = self.instance.time_windows[city]
            current_arrival = current_time + self.instance.travel_time(path[-1], city)

            if current_arrival > tw.latest:
                continue

            remaining = tw.latest - current_arrival
            scored.append((remaining, city))

        scored.sort()
        return [city for _, city in scored]

    def _forward_checking(self, path: List[int], current_time: float,
                         next_city: int) -> bool:
        """
        Forward checking: Check if selecting next_city could lead to dead end.
        
        Args:
            path: Current path
            current_time: Current time
            next_city: Candidate next city
            
        Returns:
            True if search can continue safely
        """
        new_time = self.instance.arrival_time(current_time, path[-1], next_city)
        new_time += self.instance.service_times[next_city]

        visited = set(path)
        visited.add(next_city)
        unvisited = [i for i in range(self.n) if i not in visited and i != self.start]

        for city in unvisited:
            min_possible_time = new_time + self.instance.travel_time(next_city, city)
            if min_possible_time > self.instance.time_windows[city].latest:
                return False

        return True

    def _compute_cost(self, path: List[int]) -> float:
        """Compute total path cost (including return to start)"""
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(len(path) - 1):
            total += self.instance.travel_time(path[i], path[i + 1])
        total += self.instance.travel_time(path[-1], self.start)
        return total

    def _backtrack(self, path: List[int], current_time: float,
                  visited: Set[int], depth: int):
        """
        Backtracking search.
        
        Args:
            path: Current path
            current_time: Current time
            visited: Set of visited cities
            depth: Recursion depth
        """
        if len(path) == self.n:
            if not self.instance.is_feasible(current_time, path[-1], self.start):
                return
            cost = self._compute_cost(path)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_path = path.copy()
            return

        candidates = self._select_unassigned(path, current_time)

        for next_city in candidates:
            if not self._forward_checking(path, current_time, next_city):
                continue

            new_time = self.instance.arrival_time(current_time, path[-1], next_city)
            new_time += self.instance.service_times[next_city]

            path.append(next_city)
            visited.add(next_city)

            self._backtrack(path, new_time, visited, depth + 1)

            path.pop()
            visited.remove(next_city)

    def solve(self) -> Dict:
        """
        Execute CSP solving.
        
        Returns:
            Result dictionary with path, cost, time, success
        """
        start_time = time.time()

        path = [self.start]
        visited = {self.start}
        current_time = max(0.0, self.instance.time_windows[self.start].earliest)
        if current_time > self.instance.time_windows[self.start].latest:
            return {
                'path': [],
                'cost': float('inf'),
                'time': time.time() - start_time,
                'success': False
            }

        self.best_cost = float('inf')
        self.best_path = None

        self._backtrack(path, current_time, visited, 0)

        elapsed = time.time() - start_time

        if self.best_path:
            final_path = self.best_path + [self.start]
            return {
                'path': final_path,
                'cost': self.best_cost,
                'time': elapsed,
                'success': True
            }
        else:
            return {
                'path': [],
                'cost': float('inf'),
                'time': elapsed,
                'success': False
            }

    def verify_solution(self, path: List[int]) -> Tuple[bool, str]:
        """
        Verify solution feasibility.
        
        Args:
            path: Path to verify
            
        Returns:
            (is_feasible, error_message)
        """
        if len(path) < 2:
            return False, "Path is too short"

        if path[0] != self.start or path[-1] != self.start:
            return False, "Path must start and end at start city"

        if len(set(path[:-1])) != len(path[:-1]):
            return False, "Path contains duplicate cities"

        if len(path) - 1 != self.n:
            return False, f"Path must visit all {self.n} cities"

        current_time = max(0.0, self.instance.time_windows[self.start].earliest)
        for i in range(len(path) - 1):
            from_city = path[i]
            to_city = path[i + 1]

            if not self.instance.is_feasible(current_time, from_city, to_city):
                arrival = current_time + self.instance.travel_time(from_city, to_city)
                tw = self.instance.time_windows[to_city]
                return False, f"City {to_city} time window [{tw.earliest}, {tw.latest}] violated, arrival time {arrival}"

            current_time = self.instance.arrival_time(current_time, from_city, to_city)
            if i < len(path) - 2:
                current_time += self.instance.service_times[to_city]

        return True, "Feasible"


class TSPTWSolver:
    """
    TSPTW-Specific Solver
    
    CSP-based solver optimized for TSPTW problem
    """

    def __init__(self, instance: TSPTWInstance, start_city: int = 0):
        """
        Initialize TSPTW solver.
        
        Args:
            instance: TSPTWInstance object
            start_city: Starting city index
        """
        self.instance = instance
        self.start = start_city
        self.n = instance.n
        self.csp_solver = CSPSolver(instance, start_city)

    def solve(self) -> Dict:
        """Solve TSPTW problem"""
        return self.csp_solver.solve()

    def verify_solution(self, path: List[int]) -> Tuple[bool, str]:
        """
        Verify solution feasibility.
        
        Args:
            path: Path to verify
            
        Returns:
            (is_feasible, error_message)
        """
        return self.csp_solver.verify_solution(path)


def generate_random_twpt_instance(n_cities: int, seed: int = 42) -> TSPTWInstance:
    """
    Generate random TSPTW instance (for testing).
    
    Args:
        n_cities: Number of cities
        seed: Random seed
        
    Returns:
        TSPTWInstance
    """
    np.random.seed(seed)

    coords = np.random.rand(n_cities, 2) * 100

    time_windows = []
    for i in range(n_cities):
        earliest = np.random.uniform(0, 50)
        window_size = np.random.uniform(20, 80)
        latest = earliest + window_size
        time_windows.append((earliest, latest))

    service_times = np.random.uniform(0, 5, n_cities).tolist()

    return TSPTWInstance(coords, time_windows, service_times)
