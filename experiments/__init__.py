"""
experiments module initialization file

Automated experiment running, comparison, statistics, and plotting:
- ExperimentRunner: Execute algorithms across instances
- StatisticalAnalyzer: Perform t-tests and generate reports

Reference notes:
- Experiment comparison uses locally collected metrics and standard tests.
- Full bibliographic details and reuse rationale are in REFERENCES.md.
"""
from .runner import ExperimentRunner
from .statistics import StatisticalAnalyzer

__all__ = ['ExperimentRunner', 'StatisticalAnalyzer']
