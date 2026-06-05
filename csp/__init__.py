"""
csp module initialization file

TSPTW (Traveling Salesman Problem with Time Windows) CSP solving:
- TSPTWInstance: TSPTW problem instance
- TSPTWSolver: TSPTW-specific solver
- CSPSolver: Generic CSP solver with backtracking + forward checking + MRV

Reference notes:
- CSP, forward checking, and MRV follow standard AI references.
- Full bibliographic details and reuse rationale are in REFERENCES.md.
"""
from .tsptw_solver import TSPTWInstance, TSPTWSolver, CSPSolver

__all__ = ['TSPTWInstance', 'TSPTWSolver', 'CSPSolver']
