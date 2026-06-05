# References and Reuse Notes

This file records the main external sources, libraries, and reused materials in
the project. References use IEEE-style numbering. Module docstrings point here
for full details so that code comments stay concise.

## References

[1] Ziya, "Traveling Salesman Problem (TSPLIB Dataset)," Kaggle, n.d. [Online].
Available: https://www.kaggle.com/datasets/ziya07/traveling-salesman-problem-tsplib-dataset.
Accessed: Jun. 5, 2026.

[2] S. J. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*,
4th ed. Boston, MA, USA: Pearson, 2021.

[3] P. E. Hart, N. J. Nilsson, and B. Raphael, "A formal basis for the
heuristic determination of minimum cost paths," *IEEE Transactions on Systems
Science and Cybernetics*, vol. 4, no. 2, pp. 100-107, 1968, doi:
10.1109/TSSC.1968.300136.

[4] M. Held and R. M. Karp, "A dynamic programming approach to sequencing
problems," *Journal of the Society for Industrial and Applied Mathematics*,
vol. 10, no. 1, pp. 196-210, 1962, doi: 10.1137/0110015.

[5] R. C. Prim, "Shortest connection networks and some generalizations," *Bell
System Technical Journal*, vol. 36, no. 6, pp. 1389-1401, 1957.

[6] C. R. Harris, K. J. Millman, S. J. van der Walt, *et al.*, "Array
programming with NumPy," *Nature*, vol. 585, pp. 357-362, 2020, doi:
10.1038/s41586-020-2649-2.

[7] pandas development team, "pandas documentation," 2026. [Online].
Available: https://pandas.pydata.org/docs/. Accessed: Jun. 5, 2026.

[8] F. Pedregosa, G. Varoquaux, A. Gramfort, *et al.*, "Scikit-learn: Machine
learning in Python," *Journal of Machine Learning Research*, vol. 12, pp.
2825-2830, 2011.

[9] P. Virtanen, R. Gommers, T. E. Oliphant, *et al.*, "SciPy 1.0:
Fundamental algorithms for scientific computing in Python," *Nature Methods*,
vol. 17, pp. 261-272, 2020, doi: 10.1038/s41592-019-0686-2.

[10] J. D. Hunter, "Matplotlib: A 2D graphics environment," *Computing in
Science & Engineering*, vol. 9, no. 3, pp. 90-95, 2007, doi:
10.1109/MCSE.2007.55.

[11] Python Software Foundation, "tkinter - Python interface to Tcl/Tk," 2026.
[Online]. Available: https://docs.python.org/3/library/tkinter.html. Accessed:
Jun. 5, 2026.

## Reuse Rationale

- The Kaggle dataset [1] was reused because it provides coordinate-based TSP
  instances suitable for a transportation routing problem. Reusing a public
  dataset gives the experiments an external data source instead of hand-made
  examples. The project performs its own cleaning, validation, small-instance
  derivation, distance-matrix computation, filtering, and testing.
- Standard AI algorithms and theory references [2]-[5] were reused because
  BFS, UCS, A*, CSP, Held-Karp, and Prim's algorithm are established methods.
  The project does not copy an external TSP solver; the state representation,
  scale guards, dominance checks, visualization generators, MST heuristic
  cache, and experiment integration are implemented locally.
- Scientific Python libraries [6]-[10] were reused for infrastructure: array
  operations, CSV loading, statistical tests, machine-learning models, and
  plotting. These libraries support reproducibility and reliability while the
  project-specific routing logic remains in the repository code.
- Tkinter [11] and Matplotlib [10] were reused to build the desktop visualizer
  because they are standard Python-compatible UI and plotting tools. The
  interface layout, dynamic controls, animation behavior, and algorithm
  integration are project-specific.

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
