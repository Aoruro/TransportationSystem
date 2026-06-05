# Data and Infrastructure Section Draft

## Data and Pre-processing

The project reuses a Kaggle-hosted Traveling Salesman Problem dataset [1] because
it provides public coordinate-based routing instances rather than hand-made toy
examples. The repository stores it as `tsp_instances_dataset.csv`, with 113
instances ranging from 20 to 149 cities. Each row contains metadata and city
coordinate pairs such as `City_1_X` and `City_1_Y`.

The preprocessing pipeline in `data/data_processor.py` parses the CSV with
pandas and converts valid rows into `TSPInstance` objects. It rejects incomplete
rows, invalid numeric values, non-finite coordinates, and duplicate city
locations. Each valid instance is converted into a Euclidean distance matrix:

```text
d(i, j) = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
```

The symmetric matrix has a zero diagonal and is shared by BFS, UCS, A*, and
Learning A*. Coordinates are also standardised for analysis and ML features. To
keep search feasible, the main filter uses `N <= 25`, leaving eight original
instances for A*. For fair BFS, UCS, and A* comparison,
`scripts/build_small_dataset.py` creates 113 deterministic ten-city instances.

The infrastructure is modular: loading, validation, distance computation,
filtering, and splitting are isolated in the data module and reused by the UI
and experiments. `requirements.txt` defines the environment, and unit tests
cover loading, normalisation, distance matrices, validation, splitting, and
small-dataset generation.

## Suggested Reference Entry

[1] Ziya, "Traveling Salesman Problem (TSPLIB Dataset)," Kaggle, n.d. [Online].
Available: https://www.kaggle.com/datasets/ziya07/traveling-salesman-problem-tsplib-dataset.
Accessed: Jun. 5, 2026.
