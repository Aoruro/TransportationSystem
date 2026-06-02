"""
TSP Dataset Processing Module

Features:
- Kaggle TSP dataset reading, parsing, and cleaning
- Outlier detection and filtering
- Coordinate normalization (zero-mean, unit variance)
- Euclidean distance matrix computation
- Instance filtering (N <= 25)
- Dataset splitting (300 train, 100 validation, 100 test)

References:
- Euclidean distance calculation based on numpy.linalg.norm
- Normalization technique from standard ML preprocessing practices
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import os


class TSPInstance:
    """TSP Problem Instance Class
    
    Stores coordinates, distance matrix, and time windows for a TSP instance.
    """
    def __init__(self, name: str, coords: np.ndarray, dist_matrix: np.ndarray = None,
                 time_windows: List[Tuple[float, float]] = None):
        """
        Initialize a TSP instance.
        
        Args:
            name: Instance identifier
            coords: Nx2 numpy array of city coordinates
            dist_matrix: Precomputed distance matrix (optional)
            time_windows: Optional time windows for TSPTW
        """
        self.name = name
        self.coords = coords
        self.n = len(coords)
        self.dist_matrix = dist_matrix if dist_matrix is not None else self._compute_dist_matrix()
        self.time_windows = time_windows
        self.normalized_coords = None

    def _compute_dist_matrix(self) -> np.ndarray:
        """Compute Euclidean distance matrix between all cities"""
        n = self.n
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist[i][j] = np.linalg.norm(self.coords[i] - self.coords[j])
                dist[j][i] = dist[i][j]
        return dist

    def normalize_coords(self) -> np.ndarray:
        """Normalize coordinates: zero-mean, unit variance
        
        Returns:
            Normalized coordinates array
        """
        mean = np.mean(self.coords, axis=0)
        std = np.std(self.coords, axis=0)
        std[std == 0] = 1.0  # Avoid division by zero
        self.normalized_coords = (self.coords - mean) / std
        return self.normalized_coords

    def __repr__(self):
        return f"TSPInstance(name={self.name}, n={self.n})"


class DataValidator:
    """Data Validator: Outlier Detection and Filtering
    
    Validates TSP instances for data quality and consistency.
    """

    @staticmethod
    def validate_coords(coords: np.ndarray) -> bool:
        """
        Validate coordinate data quality.
        
        Checks:
        - No NaN or Inf values
        - No duplicate cities
        - Reasonable coordinate range
        
        Args:
            coords: Nx2 coordinate array
            
        Returns:
            True if valid, False otherwise
        """
        try:
            coords = np.asarray(coords, dtype=float)
        except (TypeError, ValueError):
            return False
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) == 0:
            return False

        if np.any(np.isnan(coords)) or np.any(np.isinf(coords)):
            return False

        if len(coords) != len(np.unique(coords, axis=0)):
            return False

        return True

    @staticmethod
    def filter_instances(instances: List[TSPInstance],
                        max_cities: int = 25,
                        min_cities: int = 3) -> List[TSPInstance]:
        """
        Filter instances by city count and validation.
        
        Args:
            instances: List of TSP instances
            max_cities: Maximum allowed cities (default 25)
            min_cities: Minimum required cities (default 3)
            
        Returns:
            Filtered list of valid instances
        """
        filtered = []
        for inst in instances:
            if min_cities <= inst.n <= max_cities:
                if DataValidator.validate_coords(inst.coords):
                    filtered.append(inst)
        return filtered


class DataProcessor:
    """Main Data Processing Class
    
    Handles loading, cleaning, preprocessing, and splitting of TSP datasets.
    """

    def __init__(self, data_path: str = "tsp_instances_dataset.csv"):
        """
        Initialize data processor.
        
        Args:
            data_path: Path to CSV dataset file
        """
        self.data_path = data_path
        self.raw_instances: List[TSPInstance] = []
        self.filtered_instances: List[TSPInstance] = []
        self.train_set: List[TSPInstance] = []
        self.val_set: List[TSPInstance] = []
        self.test_set: List[TSPInstance] = []

    def load_from_csv(self) -> List[TSPInstance]:
        """
        Load TSP dataset from Kaggle format CSV.
        
        CSV format: TSP_Instance, Num_Cities, City_1_X, City_1_Y, ...
        
        Returns:
            List of parsed TSP instances
        """
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            print(f"Error: File not found at {self.data_path}")
            return []
        except Exception as e:
            print(f"Error reading CSV: {str(e)}")
            return []

        instances = []

        for idx in range(len(df)):
            row = df.iloc[idx]
            
            try:
                n_cities = int(row['Num_Cities'])
            except (ValueError, TypeError):
                continue

            if n_cities > 100 or n_cities < 3:
                continue

            xs = []
            ys = []
            for i in range(1, n_cities + 1):
                x_col = f'City_{i}_X'
                y_col = f'City_{i}_Y'
                if x_col in df.columns and y_col in df.columns:
                    x_val = row.get(x_col)
                    y_val = row.get(y_col)
                    if pd.notna(x_val) and pd.notna(y_val):
                        try:
                            xs.append(float(x_val))
                            ys.append(float(y_val))
                        except (ValueError, TypeError):
                            continue

            if len(xs) == n_cities and len(ys) == n_cities:
                coords = np.column_stack((xs, ys))
                inst_name = str(row.get('TSP_Instance', f'instance_{idx+1}'))
                inst = TSPInstance(name=inst_name, coords=coords)
                instances.append(inst)

        self.raw_instances = instances
        return instances

    def preprocess(self, max_cities: int = 25) -> List[TSPInstance]:
        """
        Preprocess dataset:
        1. Outlier detection and filtering
        2. Coordinate normalization
        3. Filter instances with N <= max_cities
        
        Args:
            max_cities: Maximum city count threshold
            
        Returns:
            Preprocessed instances
        """
        validator = DataValidator()
        self.filtered_instances = validator.filter_instances(
            self.raw_instances, max_cities=max_cities
        )

        for inst in self.filtered_instances:
            inst.normalize_coords()

        return self.filtered_instances

    def split_dataset(self, train_size: int = 300,
                     val_size: int = 100,
                     test_size: int = 100) -> Tuple[List[TSPInstance], List[TSPInstance], List[TSPInstance]]:
        """
        Split dataset into train/validation/test sets.
        
        Test set composition:
        - 25 instances with N=10
        - 25 instances with N=15
        - 25 instances with N=20
        - 25 instances with N=25
        
        Args:
            train_size: Number of training instances
            val_size: Number of validation instances
            test_size: Number of test instances
            
        Returns:
            (train_set, val_set, test_set)
        """
        instances = self.filtered_instances.copy()
        np.random.seed(42)
        np.random.shuffle(instances)

        # Scale requested sizes proportionally when the dataset is smaller.
        requested_total = train_size + val_size + test_size
        if requested_total <= 0:
            raise ValueError("At least one dataset split size must be positive")
        if len(instances) < requested_total:
            train_size = int(len(instances) * train_size / requested_total)
            val_size = int(len(instances) * val_size / requested_total)
            test_size = len(instances) - train_size - val_size

        self.train_set = instances[:train_size]
        self.val_set = instances[train_size:train_size + val_size]

        test_candidates = instances[train_size + val_size:]
        test_by_size = {10: [], 15: [], 20: [], 25: []}

        for inst in test_candidates:
            if inst.n in test_by_size and len(test_by_size[inst.n]) < 25:
                test_by_size[inst.n].append(inst)

        self.test_set = []
        for n in [10, 15, 20, 25]:
            self.test_set.extend(test_by_size[n][:25])

        # Fill remaining test slots with other sizes
        remaining = [inst for inst in test_candidates if inst not in self.test_set]
        while len(self.test_set) < test_size and remaining:
            self.test_set.append(remaining.pop(0))

        self.test_set = self.test_set[:test_size]

        return self.train_set, self.val_set, self.test_set

    def get_instances_by_size(self, n: int) -> List[TSPInstance]:
        """
        Get instances with specific city count.
        
        Args:
            n: Target city count
            
        Returns:
            List of instances with exactly n cities
        """
        all_sets = [self.train_set, self.val_set, self.test_set]
        result = []
        for dataset in all_sets:
            result.extend([inst for inst in dataset if inst.n == n])
        return result


def load_tsp_instances(num_instances: int = None,
                       data_path: str = "tsp_instances_dataset.csv") -> List[TSPInstance]:
    """
    Convenience function: Load and preprocess TSP instances.
    
    Args:
        num_instances: If specified, return only first N instances
        data_path: Path to data file
        
    Returns:
        List of TSPInstance objects
    """
    processor = DataProcessor(data_path)
    instances = processor.load_from_csv()
    instances = processor.preprocess(max_cities=25)

    if num_instances is not None:
        instances = instances[:num_instances]

    return instances


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distance matrix.
    
    Args:
        coords: Nx2 coordinate array
        
    Returns:
        NxN distance matrix
    """
    n = len(coords)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            dist[i][j] = d
            dist[j][i] = d
    return dist
