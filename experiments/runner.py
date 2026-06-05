"""
Automated Experiment Runner

Features:
- Automated execution: BFS, UCS, standard A*, Learning A* (Random Forest/MLP)
- Result statistics: nodes expanded, runtime, path length, optimality rate, relative error
- Batch processing of multiple instances

Reference notes:
- Experiment metrics compare locally implemented search algorithms.
- Full bibliographic details and reuse rationale are in REFERENCES.md.
"""

import numpy as np
import pandas as pd
import time
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from data.data_processor import TSPInstance, load_tsp_instances
from search.bfs import BFSSolver
from search.ucs import UCSSolver
from search.astar import AStarSolver
from search.utils import held_karp
from ml.learning_astar import LearningAStar
from ml.model import TSPMLModel
from ml.pseudo_labels import generate_pseudo_labels_batch


@dataclass
class ExperimentResult:
    """Single experiment result"""
    algorithm: str
    instance_name: str
    n_cities: int
    path: List[int]
    cost: float
    optimal_cost: float
    nodes_expanded: int
    time_seconds: float
    optimality_maintained: bool
    relative_error: float
    success: bool = True


class ExperimentRunner:
    """
    Experiment Runner
    
    Automatically runs multiple algorithms and collects results
    """

    ALGORITHMS = ['bfs', 'ucs', 'astar', 'learning_astar_rf', 'learning_astar_mlp']

    def __init__(self, instances: List[TSPInstance] = None,
                 data_path: str = "tsp_instances_dataset.csv"):
        """
        Initialize experiment runner.
        
        Args:
            instances: Optional list of preloaded TSP instances
            data_path: Path to dataset CSV file
        """
        self.instances = instances
        self.data_path = data_path
        self.results: List[ExperimentResult] = []
        self.ml_models: Dict[str, TSPMLModel] = {}
        self._optimal_cost_cache: Dict[int, float] = {}

    def _get_optimal_cost(self, instance: TSPInstance) -> float:
        """Return an exact reference cost and cache it per instance."""
        cache_key = id(instance)
        if cache_key not in self._optimal_cost_cache:
            if instance.n <= 15:
                _, cost = held_karp(instance.dist_matrix)
            else:
                result = AStarSolver(instance.dist_matrix).solve()
                if not result['success']:
                    raise RuntimeError(f"Could not compute reference cost for {instance.name}")
                cost = result['cost']
            self._optimal_cost_cache[cache_key] = float(cost)
        return self._optimal_cost_cache[cache_key]

    def load_instances(self, num_instances: int = 100,
                      max_n: int = 25):
        """
        Load TSP instances from dataset.
        
        Args:
            num_instances: Maximum number of instances to load
            max_n: Maximum city count per instance
            
        Returns:
            List of loaded instances
        """
        if self.instances is None:
            self.instances = load_tsp_instances(
                num_instances=num_instances,
                data_path=self.data_path
            )
            self.instances = [inst for inst in self.instances if inst.n <= max_n]
        return self.instances

    def train_ml_models(self, training_instances: List[TSPInstance]):
        """
        Train ML models for learning-enhanced A*.
        
        Args:
            training_instances: List of instances for training data generation
        """
        print("Generating pseudo-labels...")
        X, y = generate_pseudo_labels_batch(training_instances)
        print(f"Generated {len(y)} training samples")
        if len(y) == 0:
            raise ValueError("No pseudo-label samples were generated")

        print("Training Random Forest...")
        rf_model = TSPMLModel(model_type='rf')
        rf_metrics = rf_model.train(X, y)
        print(f"Random Forest - Accuracy: {rf_metrics['accuracy']:.4f}")
        self.ml_models['rf'] = rf_model

        print("Training MLP...")
        mlp_model = TSPMLModel(model_type='mlp')
        mlp_metrics = mlp_model.train(X, y)
        print(f"MLP - Accuracy: {mlp_metrics['accuracy']:.4f}")
        self.ml_models['mlp'] = mlp_model

    def run_single_experiment(self, instance: TSPInstance,
                              algorithm: str) -> ExperimentResult:
        """
        Run single algorithm on single instance.
        
        Args:
            instance: TSPInstance to test
            algorithm: Algorithm name to execute
            
        Returns:
            ExperimentResult object
        """
        dist = instance.dist_matrix
        n = instance.n

        start_time = time.time()

        try:
            if algorithm == 'bfs' and n <= 10:
                solver = BFSSolver(dist)
                result = solver.solve()
            elif algorithm == 'ucs' and n <= 12:
                solver = UCSSolver(dist)
                result = solver.solve()
            elif algorithm == 'astar':
                solver = AStarSolver(dist)
                result = solver.solve()
            elif algorithm in ['ucs', 'bfs']:
                raise ValueError(f"{algorithm.upper()} does not support N={n}")
            elif algorithm == 'learning_astar_rf':
                model = self.ml_models.get('rf')
                if model is None:
                    raise ValueError("Random Forest model has not been trained")
                solver = LearningAStar(dist, model=model, lambda_param=0.5)
                result = solver.solve()
            elif algorithm == 'learning_astar_mlp':
                model = self.ml_models.get('mlp')
                if model is None:
                    raise ValueError("MLP model has not been trained")
                solver = LearningAStar(dist, model=model, lambda_param=0.5)
                result = solver.solve()
            else:
                raise ValueError(f"Unsupported algorithm or scale: {algorithm}")
        except Exception as e:
            print(f"Error running {algorithm}: {e}")
            result = {'success': False, 'path': [], 'cost': float('inf'),
                     'nodes_expanded': 0, 'time': 0}

        elapsed = time.time() - start_time

        try:
            optimal_cost = self._get_optimal_cost(instance)
        except Exception as e:
            print(f"Error computing reference cost for {instance.name}: {e}")
            optimal_cost = float('inf')

        cost = result['cost'] if result['success'] else float('inf')
        if np.isfinite(cost) and np.isfinite(optimal_cost):
            relative_error = (
                max(0.0, float(cost - optimal_cost) / optimal_cost)
                if optimal_cost else (0.0 if cost == 0 else float('inf'))
            )
            optimality_maintained = abs(cost - optimal_cost) < 1e-6
        else:
            relative_error = float('inf')
            optimality_maintained = False

        return ExperimentResult(
            algorithm=algorithm,
            instance_name=instance.name,
            n_cities=n,
            path=result.get('path', []),
            cost=cost,
            optimal_cost=optimal_cost,
            nodes_expanded=result.get('nodes_expanded', 0),
            time_seconds=result.get('time', elapsed),
            optimality_maintained=optimality_maintained,
            relative_error=relative_error,
            success=result['success']
        )

    def run_batch(self, algorithms: List[str] = None,
                 instances: List[TSPInstance] = None,
                 verbose: bool = True) -> List[ExperimentResult]:
        """
        Run batch experiments.
        
        Args:
            algorithms: List of algorithms to run
            instances: List of instances to test
            verbose: Print progress
            
        Returns:
            List of all experiment results
        """
        if algorithms is None:
            algorithms = self.ALGORITHMS
        if instances is None:
            instances = self.instances
        if instances is None:
            raise ValueError("No experiment instances configured")

        self.results = []
        total = len(algorithms) * len(instances)

        for i, inst in enumerate(instances):
            for j, algo in enumerate(algorithms):
                if verbose:
                    print(f"[{i*len(algorithms)+j+1}/{total}] "
                          f"Instance: {inst.name}, Algorithm: {algo}")

                result = self.run_single_experiment(inst, algo)
                self.results.append(result)

        return self.results

    def results_to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for r in self.results:
            data.append({
                'algorithm': r.algorithm,
                'instance': r.instance_name,
                'n_cities': r.n_cities,
                'cost': r.cost,
                'optimal_cost': r.optimal_cost,
                'nodes_expanded': r.nodes_expanded,
                'time_seconds': r.time_seconds,
                'success': r.success,
                'optimality_maintained': r.optimality_maintained,
                'relative_error': r.relative_error
            })
        return pd.DataFrame(data)

    def summarize(self) -> Dict:
        """Generate summary statistics."""
        df = self.results_to_dataframe()
        if df.empty:
            return {}

        summary = {}
        for algo in df['algorithm'].unique():
            algo_df = df[df['algorithm'] == algo]
            summary[algo] = {
                'avg_nodes': algo_df['nodes_expanded'].mean(),
                'avg_time': algo_df['time_seconds'].mean(),
                'avg_cost': algo_df['cost'].mean(),
                'success_rate': algo_df['success'].mean(),
                'optimality_rate': algo_df['optimality_maintained'].mean(),
                'n_instances': len(algo_df)
            }

        return summary

    def save_results(self, filename: str = "experiment_results.json"):
        """
        Save results to JSON file.
        
        Args:
            filename: Output file path
        """
        def json_float(value):
            return float(value) if np.isfinite(value) else None

        data = []
        for r in self.results:
            data.append({
                'algorithm': r.algorithm,
                'instance_name': r.instance_name,
                'n_cities': r.n_cities,
                'path': r.path,
                'cost': json_float(r.cost),
                'optimal_cost': json_float(r.optimal_cost),
                'nodes_expanded': r.nodes_expanded,
                'time_seconds': json_float(r.time_seconds),
                'success': bool(r.success),
                'optimality_maintained': bool(r.optimality_maintained),
                'relative_error': json_float(r.relative_error)
            })

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, allow_nan=False)

    def load_results(self, filename: str = "experiment_results.json"):
        """
        Load results from JSON file.
        
        Args:
            filename: Input file path
        """
        with open(filename, 'r') as f:
            data = json.load(f)

        self.results = []
        for item in data:
            for key in ('cost', 'optimal_cost', 'time_seconds', 'relative_error'):
                if item[key] is None:
                    item[key] = float('inf')
            item.setdefault('success', bool(item.get('path')))
            self.results.append(ExperimentResult(**item))


def run_standard_experiments(data_path: str = "tsp_small_instances.csv",
                            output_file: str = "experiment_results.json"):
    """
    Run the three required search strategies on comparable small instances.
    
    Args:
        data_path: Path to dataset CSV
        output_file: Path to save results
    """
    print("="*60)
    print("TSP Algorithm Experiments")
    print("="*60)

    runner = ExperimentRunner(data_path=data_path)
    print("\n[1/4] Loading instances...")
    instances = runner.load_instances(num_instances=20, max_n=10)
    print(f"Loaded {len(instances)} instances")

    print("\n[2/4] Running core search experiments...")
    algorithms = ['bfs', 'ucs', 'astar']
    runner.run_batch(algorithms=algorithms, instances=instances, verbose=True)

    print("\n[3/4] Saving results...")
    runner.save_results(output_file)

    print("\n[4/4] Result Summary:")
    summary = runner.summarize()
    for algo, stats in summary.items():
        print(f"\n{algo}:")
        print(f"  Avg Nodes: {stats['avg_nodes']:.1f}")
        print(f"  Avg Time: {stats['avg_time']:.4f}s")
        print(f"  Success Rate: {stats['success_rate']:.2%}")
        print(f"  Optimality Rate: {stats['optimality_rate']:.2%}")


if __name__ == "__main__":
    run_standard_experiments()
