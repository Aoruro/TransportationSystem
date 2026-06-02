"""
experiments module initialization file

Automated experiment running, comparison, statistics, and plotting:
- ExperimentRunner: Execute algorithms across instances
- StatisticalAnalyzer: Perform t-tests and generate reports

References:
- Experimental methodology for algorithm comparison
- Statistical hypothesis testing
"""
from .runner import ExperimentRunner
from .statistics import StatisticalAnalyzer

__all__ = ['ExperimentRunner', 'StatisticalAnalyzer']