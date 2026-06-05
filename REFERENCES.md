# References and Reuse Notes

This file records the main external sources, libraries, and reused materials in
the project. Module docstrings point here for full details so that code comments
stay concise.

## Dataset

- Ziya. (n.d.). *Traveling Salesman Problem (TSPLIB Dataset)* [Dataset].
  Kaggle. CC0 Public Domain license.
  https://www.kaggle.com/datasets/ziya07/traveling-salesman-problem-tsplib-dataset
  Accessed 5 June 2026.

## Algorithm and Theory References

- Russell, S. J., & Norvig, P. (2021). *Artificial Intelligence: A Modern
  Approach* (4th ed.). Pearson. Used as the general AI reference for
  state-space search, breadth-first search, uniform-cost search, A*, constraint
  satisfaction, forward checking, and MRV.
- Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). A formal basis for the
  heuristic determination of minimum cost paths. *IEEE Transactions on Systems
  Science and Cybernetics, 4*(2), 100-107.
  https://doi.org/10.1109/TSSC.1968.300136. Used as the original A* reference.
- Held, M., & Karp, R. M. (1962). A dynamic programming approach to sequencing
  problems. *Journal of the Society for Industrial and Applied Mathematics,
  10*(1), 196-210. https://doi.org/10.1137/0110015. Used for the exact
  small-instance TSP baseline.
- Prim, R. C. (1957). Shortest connection networks and some generalizations.
  *Bell System Technical Journal, 36*(6), 1389-1401. Used for the minimum
  spanning tree term in the A* lower-bound heuristic.

## Software Libraries

- Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array
  programming with NumPy. *Nature, 585*, 357-362.
  https://doi.org/10.1038/s41586-020-2649-2.
- pandas development team. (2026). *pandas documentation*.
  https://pandas.pydata.org/docs/. Accessed 5 June 2026.
- Pedregosa, F., Varoquaux, G., Gramfort, A., et al. (2011). Scikit-learn:
  Machine learning in Python. *Journal of Machine Learning Research, 12*,
  2825-2830.
- Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0:
  Fundamental algorithms for scientific computing in Python. *Nature Methods,
  17*, 261-272. https://doi.org/10.1038/s41592-019-0686-2.
- Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in
  Science & Engineering, 9*(3), 90-95.
  https://doi.org/10.1109/MCSE.2007.55.
- Python Software Foundation. (2026). *tkinter - Python interface to Tcl/Tk*.
  https://docs.python.org/3/library/tkinter.html. Accessed 5 June 2026.

## Reuse Rationale

- The Kaggle dataset was reused because it provides coordinate-based TSP
  instances suitable for a transportation routing problem. Reusing a public
  dataset gives the experiments an external data source instead of hand-made
  examples. The project performs its own cleaning, validation, small-instance
  derivation, distance-matrix computation, filtering, and testing.
- Standard AI algorithms and theory references were reused because BFS, UCS,
  A*, CSP, and Held-Karp are established methods. The project does not copy an
  external TSP solver; the state representation, scale guards, dominance
  checks, visualization generators, MST heuristic cache, and experiment
  integration are implemented locally.
- Scientific Python libraries were reused for infrastructure: array
  operations, CSV loading, statistical tests, machine-learning models, and
  plotting. These libraries support reproducibility and reliability while the
  project-specific routing logic remains in the repository code.
- Tkinter and Matplotlib were reused to build the desktop visualizer because
  they are standard Python-compatible UI and plotting tools. The interface
  layout, dynamic controls, animation behavior, and algorithm integration are
  project-specific.

## Original Project Work

The project-specific implementation includes:

- TSP state encoding, input validation, and solution verification.
- BFS, UCS, and A* solvers with algorithm-specific scale guards and
  visualization generators.
- Learning-enhanced A* integration, feature extraction, pseudo-label
  generation, and prediction caching.
- TSP with time windows as a CSP extension.
- Dataset preprocessing, deterministic small-dataset generation, experiment
  automation, statistical analysis, tests, documentation, and the interactive
  UI.

## AI Assistance

AI assistance was used during development for code review, debugging, test
coverage, documentation improvements, and UI refinement. The submitted team
should review the code, understand each module, and describe this assistance
according to the course policy.
