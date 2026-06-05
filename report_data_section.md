# Data and Infrastructure Section Draft

## Data and Pre-processing

The project reuses a Kaggle-hosted Traveling Salesman Problem dataset as the
main source of routing instances. This dataset was selected because it provides
public coordinate-based TSP examples, which are more realistic and reproducible
than hand-made toy data. In the repository, the source file is stored as
`tsp_instances_dataset.csv`. It contains 113 TSP instances, with each row
representing one route-planning problem. The main fields are the instance name,
the number of cities, optional metadata such as total distance and route
category, and a sequence of two-dimensional city coordinates in the form
`City_1_X`, `City_1_Y`, `City_2_X`, `City_2_Y`, and so on. The source data
covers instances from 20 to 149 cities, which is useful for demonstrating the
scalability challenge of TSP, but too large for exhaustive search methods in an
interactive application.

The data pipeline was implemented in `data/data_processor.py`. During loading,
the CSV file is parsed with pandas and each row is converted into a
`TSPInstance` object. The loader checks that the declared number of cities has
matching coordinate columns and that all required values can be converted to
floating-point numbers. Incomplete rows, invalid numeric values, and instances
outside the supported raw loading range are skipped. A separate validator then
checks each coordinate array for the expected `N x 2` shape, non-finite values
such as `NaN` or infinity, and duplicate city coordinates. These checks reduce
the risk of invalid routes, zero-distance duplicate nodes, or matrix errors
during search.

For each valid instance, the project computes a Euclidean distance matrix:

```text
d(i, j) = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
```

The matrix is symmetric, has a zero diagonal, and is used by BFS, UCS, A*, and
Learning A* as the common cost representation. Coordinates are also
standardised to zero mean and unit variance and stored as normalised
coordinates for analysis and machine-learning-related processing. The search
algorithms themselves use the distance matrix derived from the validated
coordinate data, ensuring that all algorithms are compared on the same cost
model.

Because the assessment requires comparison between several search strategies,
the project applies instance filtering based on algorithmic feasibility. The
standard application limit is `N <= 25`, which leaves eight source instances
from the original CSV for A* demonstrations. However, BFS and UCS grow too
quickly to run smoothly on 20-city instances. To enable fair comparison of BFS,
UCS, and A* on the same data distribution, a deterministic derived dataset,
`tsp_small_instances.csv`, was created. The script
`scripts/build_small_dataset.py` takes every usable source row and keeps the
first 10 cities, producing 113 ten-city instances with the `_small` suffix. This
keeps the experiment reproducible while making exhaustive and cost-based search
methods practical for the UI and automated tests.

The data and infrastructure work was designed to keep the project modular.
Dataset loading, validation, distance computation, filtering, and splitting are
isolated in the data module; experiment scripts and the UI call this module
rather than re-implementing parsing logic. The environment is defined in
`requirements.txt`, and the project includes unit tests for data loading,
normalisation, distance matrix correctness, validation of invalid coordinates,
dataset splitting, and deterministic generation of the small dataset. These
tests help ensure that later algorithmic results are caused by search behaviour
rather than data inconsistencies.

The reused dataset is suitable for the project because it contains many
coordinate-based TSP instances under a public license. The project-specific
contribution is the preprocessing layer around that dataset: validation,
cleaning, distance-matrix construction, scale filtering, deterministic
small-instance generation, and tests. These steps are necessary to keep the
problem size aligned with the computational limits of state-space search.

## Suggested Reference Entry

Ziya. (n.d.). *Traveling Salesman Problem (TSPLIB Dataset)* [Dataset]. Kaggle.
CC0 Public Domain license.
https://www.kaggle.com/datasets/ziya07/traveling-salesman-problem-tsplib-dataset.
Accessed 5 June 2026.
