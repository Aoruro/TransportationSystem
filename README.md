# TSP Search Algorithms and Machine Learning Project

## Project Overview

This project implements a comprehensive Traveling Salesman Problem (TSP) solving system, integrating multiple classic search algorithms, machine learning-enhanced algorithms, and constraint satisfaction problem solvers.

## Project Structure

```
TransportationSystem/
├── data/                       # Data processing module
│   ├── __init__.py
│   └── data_processor.py      # TSP dataset loading, cleaning, splitting
├── search/                     # Search algorithms module
│   ├── __init__.py
│   ├── bfs.py                 # BFS breadth-first search
│   ├── ucs.py                 # UCS uniform cost search
│   ├── astar.py               # A* algorithm
│   └── utils.py               # Utility functions
├── ml/                         # Machine learning module
│   ├── __init__.py
│   ├── model.py               # ML models, feature extraction
│   ├── pseudo_labels.py       # Pseudo-label generation
│   └── learning_astar.py      # Learning-enhanced A*
├── csp/                        # CSP solving module
│   ├── __init__.py
│   └── tsptw_solver.py        # TSPTW time window solver
├── visualization/              # Visualization module
│   ├── __init__.py
│   └── ui.py                  # Interactive UI
├── experiments/                # Experiments module
│   ├── __init__.py
│   ├── runner.py              # Experiment runner
│   └── statistics.py          # Statistical analysis
├── tests/                      # Unit tests
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_search.py
│   ├── test_ml.py
│   └── test_csp.py
├── tsp_instances_dataset.csv   # TSP dataset
└── README.md                   # This document
```

## Environment Setup

### Dependencies

```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
scipy>=1.7.0
matplotlib>=3.4.0
```

### Installation

```bash
pip install numpy pandas scikit-learn scipy matplotlib
```

On Windows systems with multiple Python versions installed, Python 3.12 is
recommended:

```powershell
py -3.12 -m pip install -r requirements.txt
py -3.12 -X utf8 quick_test.py
```

## Quick Start

### 1. Load and Preprocess Data

```python
from data import load_tsp_instances

instances = load_tsp_instances(num_instances=100)
print(f"Loaded {len(instances)} instances")
```

### 2. Run Search Algorithms

```python
from search import AStarSolver

solver = AStarSolver(distance_matrix)
result = solver.solve()
print(f"Path: {result['path']}")
print(f"Cost: {result['cost']}")
print(f"Nodes expanded: {result['nodes_expanded']}")
```

### 3. Machine Learning Training

```python
from ml import TSPMLModel
from ml.pseudo_labels import generate_pseudo_labels_batch

X, y = generate_pseudo_labels_batch(instances)
model = TSPMLModel(model_type='rf')
metrics = model.train(X, y)
print(f"Accuracy: {metrics['accuracy']}")
```

### 4. Learning-enhanced A*

```python
from ml.learning_astar import LearningAStar

solver = LearningAStar(dist_matrix, model=ml_model, lambda_param=0.5)
result = solver.solve()
```

### 5. TSPTW Solving

```python
from csp import TSPTWInstance, TSPTWSolver

instance = TSPTWInstance(coords, time_windows)
solver = TSPTWSolver(instance)
result = solver.solve()
```

### 6. Run Experiments

```python
from experiments import ExperimentRunner

runner = ExperimentRunner(data_path="tsp_instances_dataset.csv")
runner.load_instances(num_instances=100)
runner.train_ml_models(train_instances)
runner.run_batch(algorithms=['astar', 'learning_astar_rf'])
```

### 7. Run Visualization

```python
from visualization import TSPVisualizer

app = TSPVisualizer()
app.run()
```

### 8. Run Unit Tests

```bash
python -m pytest tests/ -v
```

Or:

```bash
python -m unittest discover -s tests -v
```

For an end-to-end smoke test:

```bash
python -X utf8 full_test.py
```

## Module Details

### data/ - Data Processing

- **TSPInstance**: TSP problem instance class
- **DataProcessor**: Data loading and preprocessing
- **DataValidator**: Data validation and anomaly detection
- **load_tsp_instances()**: Convenience loading function

Features:
- Kaggle TSP dataset loading
- Outlier detection and filtering
- Coordinate normalization (zero-mean, unit variance)
- Euclidean distance matrix computation
- Dataset splitting (300 train, 100 validation, 100 test)

### search/ - Search Algorithms

**Algorithm Constraints**:
- BFS: N ≤ 10
- UCS: N ≤ 12
- A*: N ≤ 25

**Core Classes**:
- `BFSSolver`: Breadth-First Search
- `UCSSolver`: Uniform Cost Search
- `AStarSolver`: A* algorithm with MST heuristic

**Utility Functions**:
- `TSPState`: State representation (visited_mask, current_city)
- `held_karp()`: Dynamic programming optimal solution (N ≤ 15)
- `verify_solution()`: Solution correctness verification
- `prim_mst()`: Minimum Spanning Tree

### ml/ - Machine Learning

**Feature Engineering (10 features)**:
1. visited_ratio: Ratio of visited cities
2. mean_dist_from_current: Mean distance from current city to all cities
3. min_dist_from_current: Minimum distance from current city
4. min_candidate_dist: Minimum distance to unvisited cities
5. mean_candidate_dist: Mean distance to unvisited cities
6. unvisited_ratio: Ratio of unvisited cities
7. max_dist_to_unvisited: Maximum distance to unvisited cities
8. mst_lower_bound_ratio: MST lower bound / current cost
9. current_city_normalized: Current city index (normalized)
10. next_city_normalized: Candidate next city (normalized)

**Models**:
- Random Forest
- 2-layer MLP neural network

**Learning A***:
- Formula: `f' = f - lambda * prob`
- Supports lambda parameter tuning
- Caches predictions to reduce overhead

### csp/ - Constraint Satisfaction Solving

**TSPTW Problem**:
- Each city has a time window [earliest, latest]
- Constraint satisfaction solving

**Solution Methods**:
- Backtracking search
- Forward checking
- MRV (Minimum Remaining Values) heuristic

**Limitations**: N ≤ 10

### visualization/ - Visualization

**Features**:
- Load custom TSP instances
- Select algorithm (BFS/UCS/A*/Learning A*)
- Real-time search tree expansion display
- Final path animation
- Train an ML model from the loaded instance and adjust the Lambda parameter

**Performance Optimization**:
- Limit display to first 10000 nodes
- Hierarchical rendering to avoid lag

### experiments/ - Experiments Module

**Features**:
- Automated execution of multiple algorithms
- Result statistics and analysis
- Significance testing (paired t-test, p < 0.05)
- Performance comparison plots

**Metrics**:
- Nodes expanded
- Runtime
- Path length
- Optimality maintenance rate
- Relative error

### tests/ - Unit Tests

Coverage:
- Data preprocessing
- Distance matrix computation
- BFS, UCS, A* algorithms
- MST heuristic
- Machine learning models
- CSP solver

## Notes

1. **Scale Constraints**: Strictly follow algorithm scale limits to prevent memory overflow
2. **Optimality Guarantee**: A* uses a consistent MST heuristic. Learning A* keeps ML guidance admissible and requires a trained model.
3. **Data Format**: TSP dataset must be CSV format with City_i_X, City_i_Y columns
4. **Python Version**: Python 3.8-3.12 supported; Python 3.12 recommended

## Originality Statement

This code was developed with AI assistance according to course requirements. Core algorithm implementations reference classic textbooks:
- "Artificial Intelligence: A Modern Approach"
- "Introduction to Algorithms"

All reused code is properly cited in comments.
