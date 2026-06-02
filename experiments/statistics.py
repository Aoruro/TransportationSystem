"""
Statistical Analysis Module

Features:
- Significance testing: Paired t-test (p < 0.05)
- Performance comparison plots
- Result visualization charts

References:
- Statistical hypothesis testing for algorithm comparison
- Scipy stats module for t-tests
- Matplotlib for visualization
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy import stats
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass


@dataclass
class ComparisonResult:
    """Algorithm comparison result"""
    algorithm1: str
    algorithm2: str
    metric: str
    mean1: float
    mean2: float
    t_statistic: float
    p_value: float
    significant: bool
    better: str


class StatisticalAnalyzer:
    """
    Statistical Analyzer
    
    Features:
    - Paired t-test
    - Performance comparison
    - Plotting
    """

    METRICS = ['nodes_expanded', 'time_seconds', 'cost', 'relative_error']
    COLUMNS = [
        'algorithm', 'instance', 'n_cities', 'cost', 'nodes_expanded',
        'time_seconds', 'optimality_maintained', 'relative_error'
    ]

    def __init__(self, results: List):
        """
        Initialize analyzer.
        
        Args:
            results: List of ExperimentResult objects
        """
        self.results = results
        self.df = self._to_dataframe()

    def _to_dataframe(self) -> pd.DataFrame:
        """Convert results to pandas DataFrame."""
        data = []
        for r in self.results:
            data.append({
                'algorithm': r.algorithm,
                'instance': r.instance_name,
                'n_cities': r.n_cities,
                'cost': r.cost,
                'nodes_expanded': r.nodes_expanded,
                'time_seconds': r.time_seconds,
                'optimality_maintained': r.optimality_maintained,
                'relative_error': r.relative_error
            })
        return pd.DataFrame(data, columns=self.COLUMNS)

    def _validate_metric(self, metric: str):
        """Reject unsupported metrics before indexing the dataframe."""
        if metric not in self.METRICS:
            raise ValueError(f"Unsupported metric: {metric}")

    def paired_t_test(self, algo1: str, algo2: str,
                    metric: str = 'nodes_expanded') -> ComparisonResult:
        """
        Perform paired t-test between two algorithms.
        
        Args:
            algo1: First algorithm name
            algo2: Second algorithm name
            metric: Metric to compare
            
        Returns:
            ComparisonResult object
        """
        self._validate_metric(metric)
        df1 = self.df[self.df['algorithm'] == algo1].set_index('instance')
        df2 = self.df[self.df['algorithm'] == algo2].set_index('instance')

        common_instances = df1.index.intersection(df2.index)

        if len(common_instances) < 2:
            return ComparisonResult(
                algo1, algo2, metric, 0, 0, 0, 1.0, False, 'insufficient_data'
            )

        values1 = df1.loc[common_instances, metric].values.astype(float)
        values2 = df2.loc[common_instances, metric].values.astype(float)
        finite = np.isfinite(values1) & np.isfinite(values2)
        values1 = values1[finite]
        values2 = values2[finite]

        if len(values1) < 2:
            return ComparisonResult(
                algo1, algo2, metric, 0, 0, 0, 1.0, False, 'insufficient_data'
            )

        if np.allclose(values1, values2):
            return ComparisonResult(
                algo1, algo2, metric,
                np.mean(values1), np.mean(values2),
                0, 1.0, False, 'equal'
            )

        t_stat, p_value = stats.ttest_rel(values1, values2)

        mean1 = np.mean(values1)
        mean2 = np.mean(values2)

        significant = p_value < 0.05
        if significant:
            better = algo1 if mean1 < mean2 else algo2
        else:
            better = 'no_significant_difference'

        return ComparisonResult(
            algorithm1=algo1,
            algorithm2=algo2,
            metric=metric,
            mean1=mean1,
            mean2=mean2,
            t_statistic=t_stat,
            p_value=p_value,
            significant=significant,
            better=better
        )

    def compare_all_pairs(self, metric: str = 'nodes_expanded') -> List[ComparisonResult]:
        """
        Compare all algorithm pairs.
        
        Args:
            metric: Metric to compare
            
        Returns:
            List of ComparisonResult objects
        """
        self._validate_metric(metric)
        algorithms = self.df['algorithm'].unique()
        results = []

        for i in range(len(algorithms)):
            for j in range(i + 1, len(algorithms)):
                result = self.paired_t_test(algorithms[i], algorithms[j], metric)
                results.append(result)

        return results

    def get_summary_stats(self) -> Dict:
        """Get summary statistics for each algorithm."""
        summary = {}

        for algo in self.df['algorithm'].unique():
            algo_df = self.df[self.df['algorithm'] == algo]

            summary[algo] = {
                'n_instances': len(algo_df),
                'mean_nodes': algo_df['nodes_expanded'].mean(),
                'std_nodes': algo_df['nodes_expanded'].std(),
                'mean_time': algo_df['time_seconds'].mean(),
                'std_time': algo_df['time_seconds'].std(),
                'mean_cost': algo_df['cost'].mean(),
                'success_rate': algo_df['optimality_maintained'].mean(),
                'min_nodes': algo_df['nodes_expanded'].min(),
                'max_nodes': algo_df['nodes_expanded'].max()
            }

        return summary

    def plot_performance_comparison(self, metric: str = 'nodes_expanded',
                                  save_path: str = None):
        """
        Plot performance comparison bar chart and box plot.
        
        Args:
            metric: Metric to compare
            save_path: Optional path to save figure
        """
        self._validate_metric(metric)
        if self.df.empty:
            raise ValueError("No experiment results to plot")

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        algorithms = self.df['algorithm'].unique()

        ax = axes[0]
        means = []
        stds = []
        labels = []

        for algo in algorithms:
            algo_df = self.df[self.df['algorithm'] == algo]
            means.append(algo_df[metric].mean())
            stds.append(algo_df[metric].std())
            labels.append(algo)

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)
        ax.set_xlabel('Algorithm')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')

        ax = axes[1]
        data_to_plot = []
        for algo in algorithms:
            algo_df = self.df[self.df['algorithm'] == algo]
            data_to_plot.append(algo_df[metric].values)

        bp = ax.boxplot(data_to_plot, labels=algorithms, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor('lightblue')
            patch.set_alpha(0.7)

        ax.set_xlabel('Algorithm')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Distribution')
        ax.set_xticklabels(labels, rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def plot_scalability(self, save_path: str = None):
        """
        Plot scalability graph (nodes expanded vs city count).
        
        Args:
            save_path: Optional path to save figure
        """
        if self.df.empty:
            raise ValueError("No experiment results to plot")

        fig, ax = plt.subplots(figsize=(10, 6))

        algorithms = self.df['algorithm'].unique()

        for algo in algorithms:
            algo_df = self.df[self.df['algorithm'] == algo]
            grouped = algo_df.groupby('n_cities')[self.METRICS[0]].mean()
            ax.plot(grouped.index, grouped.values, 'o-', label=algo, alpha=0.7)

        ax.set_xlabel('Number of Cities')
        ax.set_ylabel('Average Nodes Expanded')
        ax.set_title('Algorithm Scalability')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

        plt.close()

    def generate_report(self, output_file: str = "analysis_report.txt"):
        """
        Generate analysis report.
        
        Args:
            output_file: Path to save report
            
        Returns:
            Report content as string
        """
        lines = []
        lines.append("="*60)
        lines.append("TSP Algorithm Experiment Analysis Report")
        lines.append("="*60)

        lines.append("\n1. Summary Statistics")
        lines.append("-"*40)
        summary = self.get_summary_stats()
        for algo, stats_dict in summary.items():
            lines.append(f"\n{algo}:")
            for key, value in stats_dict.items():
                if isinstance(value, float):
                    lines.append(f"  {key}: {value:.4f}")
                else:
                    lines.append(f"  {key}: {value}")

        lines.append("\n\n2. Significance Testing (Paired t-test, p < 0.05)")
        lines.append("-"*40)
        comparisons = self.compare_all_pairs('nodes_expanded')
        for comp in comparisons:
            lines.append(f"\n{comp.algorithm1} vs {comp.algorithm2}:")
            lines.append(f"  Nodes: {comp.mean1:.1f} vs {comp.mean2:.1f}")
            lines.append(f"  t-statistic: {comp.t_statistic:.4f}")
            lines.append(f"  p-value: {comp.p_value:.4f}")
            lines.append(f"  Significant: {'Yes' if comp.significant else 'No'}")
            lines.append(f"  Better: {comp.better}")

        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))

        return '\n'.join(lines)


def load_and_analyze(results_file: str = "experiment_results.json",
                    report_file: str = "analysis_report.txt") -> StatisticalAnalyzer:
    """
    Load results and perform analysis.
    
    Args:
        results_file: Path to experiment results JSON
        report_file: Path to save analysis report
        
    Returns:
        StatisticalAnalyzer instance
    """
    with open(results_file, 'r') as f:
        data = json.load(f)

    from .runner import ExperimentResult
    results = [ExperimentResult(**item) for item in data]

    analyzer = StatisticalAnalyzer(results)
    analyzer.generate_report(report_file)

    return analyzer
