# Dataset Notes

## Purpose

The project models transportation routing as a Traveling Salesman Problem
(TSP). Each row describes one routing instance as a set of two-dimensional city
coordinates. Search algorithms calculate Euclidean distances between cities and
seek the shortest closed tour.

## Repository Files

### `tsp_instances_dataset.csv`

- 113 source rows.
- Columns: `TSP_Instance`, `Num_Cities`, optional source metadata, followed by
  coordinate pairs such as `City_1_X`, `City_1_Y`, `City_2_X`, `City_2_Y`.
- City counts range from 20 to 149.
- The application filters this file to `N <= 25` before running standard A* in
  the UI. This leaves eight large demonstration instances.

### `tsp_small_instances.csv`

- 113 derived rows.
- Each row contains the first 10 cities from the corresponding source row.
- Names use the `_small` suffix.
- This file enables fair comparisons of BFS, UCS, and A*. Exhaustive strategies
  become expensive quickly, so all three core algorithms are compared on the
  same 10-city instances.
- Learning A* also uses these instances in the UI because interactive model
  training is intentionally limited to `N <= 15`.

## Rebuilding the Small Dataset

Run:

```powershell
py -3.12 scripts/build_small_dataset.py
```

The builder is deterministic. It selects every usable source row, keeps the
first 10 coordinate pairs, appends `_small` to the instance name, and writes
`tsp_small_instances.csv`.

## Data Preparation

`data/data_processor.py`:

1. Reads coordinate columns from CSV.
2. Rejects incomplete rows, invalid numeric coordinates, non-finite values, and
   duplicate cities.
3. Computes a Euclidean distance matrix for every accepted instance.
4. Filters instances by algorithm scale.
5. Stores normalized coordinates for optional analysis.

## Source Record

The exact upstream download URL was not recorded in the repository. Before
submission, add the verified dataset URL, author or organization, download date,
and license here. Do not claim a Kaggle source in the report unless the team has
verified the original page.
