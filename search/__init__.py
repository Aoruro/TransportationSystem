"""
search module initialization file

Contains search algorithms for TSP:
- BFSSolver: Breadth-First Search (N <= 10)
- UCSSolver: Uniform Cost Search (N <= 12)
- AStarSolver: A* algorithm with MST heuristic (N <= 25)
- Utilities: TSPState, verification, optimality, complexity analysis

Reference notes:
- Search algorithm theory follows standard AI state-space search references.
- Full bibliographic details and reuse rationale are in REFERENCES.md.
"""
from .bfs import BFSSolver
from .ucs import UCSSolver
from .astar import AStarSolver
from .utils import TSPState, verify_solution, verify_optimality, compute_complexity

__all__ = [
    'BFSSolver', 'UCSSolver', 'AStarSolver',
    'TSPState', 'verify_solution', 'verify_optimality', 'compute_complexity'
]
