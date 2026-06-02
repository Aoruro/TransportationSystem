# TSP Search Algorithms and Machine Learning Project

## Overview

This project models transportation routing as a Traveling Salesman Problem
(TSP). It compares three required search strategies and adds optional AI
extensions:

- Breadth-first search (BFS)
- Uniform-cost search (UCS)
- A* with an MST lower-bound heuristic
- Learning-enhanced A* with Random Forest or MLP guidance
- TSP with time windows (TSPTW) as a constraint satisfaction problem
- An interactive Tkinter and Matplotlib desktop UI
- Automated experiments, plots, paired t-tests, and regression tests

## Project Structure

```text
TransportationSystem/
|-- data/                    # CSV loading, validation, distance matrices
|-- search/                  # BFS, UCS, A*, Held-Karp, MST utilities
|-- ml/                      # Features, models, pseudo-labels, Learning A*
|-- csp/                     # TSPTW constraint satisfaction solver
|-- experiments/             # Batch experiments and statistical analysis
|-- visualization/           # Interactive and static visualizations
|-- scripts/                 # Reproducible dataset preparation
|-- tests/                   # Unit and regression tests
|-- run_ui.py                # Desktop UI entry point
|-- run_experiments.py       # Report artifact generator
|-- DATASET.md               # Dataset structure and provenance notes
|-- REFERENCES.md            # Algorithm, library, and reuse notes
`-- requirements.txt
```

## Installation

Python 3.12 is recommended:

```powershell
py -3.12 -m pip install -r requirements.txt
```

See `INSTALLATION.md` for platform notes.

## Quick Start

Run smoke checks:

```powershell
py -3.12 quick_test.py
py -3.12 full_test.py
```

Run the UI:

```powershell
py -3.12 run_ui.py
```

Run the complete test suite:

```powershell
py -3.12 -m unittest discover -s tests -v
```

## Core Search Comparison

Generate report-ready core search tables and figures:

```powershell
py -3.12 run_experiments.py
```

This compares BFS, UCS, and A* on identical 10-city instances and writes
artifacts under `results/`.

Include the slower ML comparison when needed:

```powershell
py -3.12 run_experiments.py --include-ml
```

Learning A* is an above-and-beyond experiment. It is not assumed to outperform
standard A*: model prediction overhead and training quality must be measured and
discussed honestly.

## Dataset

The repository contains:

- `tsp_instances_dataset.csv`: 113 source rows with 20-149 cities.
- `tsp_small_instances.csv`: 113 deterministic 10-city derivatives for fair
  BFS, UCS, A*, and interactive Learning A* comparisons.

Rebuild the small dataset:

```powershell
py -3.12 scripts/build_small_dataset.py
```

See `DATASET.md` for fields, filtering, derivation details, and the provenance
record that must be completed before submission.

## Algorithm Limits

Scale guards prevent accidental exponential blow-ups:

| Algorithm | Limit | Notes |
| --- | ---: | --- |
| BFS | `N <= 10` | Weighted BFS with per-state dominance |
| UCS | `N <= 12` | Includes the closing edge in goal priority |
| A* | `N <= 25` | MST lower-bound heuristic |
| Held-Karp | `N <= 15` | Exact dynamic-programming baseline |
| UI Learning A* training | `N <= 15` | Keeps interactive training responsive |
| TSPTW CSP | `N <= 10` | Backtracking, forward checking, and MRV |

## Machine Learning Extension

The ML extension uses pseudo-labels from exact or near-optimal routes and ten
features:

1. Visited ratio
2. Mean distance from the current city
3. Minimum non-self distance from the current city
4. Minimum candidate distance
5. Mean candidate distance
6. Unvisited ratio
7. Maximum candidate distance
8. MST lower-bound ratio
9. Normalized current city index
10. Normalized candidate next-city index

Models:

- Random Forest
- Two-layer MLP

Learning A* applies:

```text
adjusted heuristic = base heuristic - lambda * predicted probability
```

Completed states retain the exact return cost so ML guidance cannot make a tour
appear cheaper after completion.

## UI Features

- Select BFS, UCS, A*, or Learning A*
- Select algorithm-compatible examples
- Load custom CSV instances
- Train a Random Forest model for Learning A*
- Adjust Lambda and animation speed
- Visualize batched search expansion without freezing the UI
- Draw the first 1000 search nodes and animate the final route

## Experiment Metrics

The experiment runner records:

- Path cost
- Optimal baseline cost
- Nodes expanded
- Runtime
- Solver success rate
- Optimality rate
- Relative error

The statistics module generates comparison plots and paired t-tests.

## References and Originality

See `REFERENCES.md` for algorithm references, library use, original work, and
the AI-assistance disclosure. Add the verified upstream dataset URL to
`DATASET.md` before submission.
