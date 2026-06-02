"""
data module initialization file

Contains TSP dataset processing functionality:
- TSPInstance: TSP problem instance representation
- DataProcessor: Main data processing class
- load_tsp_instances: Convenience loading function
"""
from .data_processor import TSPInstance, DataProcessor, load_tsp_instances

__all__ = ['TSPInstance', 'DataProcessor', 'load_tsp_instances']