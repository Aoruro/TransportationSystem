# References and Reuse Notes

## Algorithms

- Russell, S. J. and Norvig, P. *Artificial Intelligence: A Modern Approach*.
  Used as the general reference for breadth-first search, uniform-cost search,
  A*, constraint satisfaction, forward checking, and MRV.
- Held, M. and Karp, R. M. (1962). "A Dynamic Programming Approach to
  Sequencing Problems." *Journal of the Society for Industrial and Applied
  Mathematics*, 10(1), 196-210. Used for the exact small-instance TSP baseline.
- The A* heuristic combines the cheapest edge from the current city, the
  minimum spanning tree cost over unvisited cities, and the cheapest return edge
  to the start city. Prim's algorithm is implemented locally for the MST term.

## Libraries

- NumPy: numerical arrays and distance calculations.
- pandas: CSV loading and result tables.
- scikit-learn: Random Forest, MLP, train/test splitting, cross-validation, and
  classification metrics.
- SciPy: paired t-tests for experiment comparisons.
- Matplotlib: static figures and Tkinter-embedded visualizations.
- Tkinter: desktop user interface.

## Original Work

The project-specific implementation includes:

- TSP state encoding and validation.
- BFS, UCS, and A* solvers with scale guards and visualization generators.
- Learning-enhanced A* integration, feature extraction, pseudo-label
  generation, and prediction caching.
- TSP with time windows as a CSP extension.
- Dataset processing, experiment automation, statistical analysis, tests, and
  the interactive UI.

## AI Assistance

AI assistance was used during development for code review, debugging, test
coverage, documentation improvements, and UI refinement. The submitted team
should review the code, understand each module, and describe this assistance
according to the course policy.
