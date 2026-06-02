"""
Data Processing Module Tests

Test coverage:
- Data loading
- Distance matrix computation
- Coordinate normalization
- Dataset splitting
"""

import unittest
import numpy as np
import sys
import os
import tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_small_dataset import build_small_dataset
from data.data_processor import (
    TSPInstance, DataProcessor, DataValidator,
    compute_distance_matrix
)


class TestTSPInstance(unittest.TestCase):
    """TSPInstance Class Tests"""

    def setUp(self):
        self.coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        self.instance = TSPInstance(name="test", coords=self.coords)

    def test_instance_creation(self):
        """Test instance creation"""
        self.assertEqual(self.instance.name, "test")
        self.assertEqual(self.instance.n, 4)

    def test_distance_matrix_shape(self):
        """Test distance matrix shape"""
        self.assertEqual(self.instance.dist_matrix.shape, (4, 4))

    def test_distance_matrix_symmetry(self):
        """Test distance matrix symmetry"""
        dist = self.instance.dist_matrix
        np.testing.assert_array_almost_equal(dist, dist.T)

    def test_distance_matrix_diagonal(self):
        """Test distance matrix diagonal is zero"""
        np.testing.assert_array_equal(np.diag(self.instance.dist_matrix), 0)

    def test_normalize_coords(self):
        """Test coordinate normalization"""
        norm_coords = self.instance.normalize_coords()
        self.assertEqual(norm_coords.shape, self.coords.shape)
        self.assertAlmostEqual(np.mean(norm_coords[:, 0]), 0, places=5)
        self.assertAlmostEqual(np.mean(norm_coords[:, 1]), 0, places=5)


class TestDistanceMatrix(unittest.TestCase):
    """Distance Matrix Computation Tests"""

    def test_simple_square(self):
        """Test simple square configuration"""
        coords = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        dist = compute_distance_matrix(coords)

        self.assertEqual(dist.shape, (4, 4))
        self.assertAlmostEqual(dist[0][1], 1.0)
        self.assertAlmostEqual(dist[0][2], np.sqrt(2))
        self.assertAlmostEqual(dist[0][3], 1.0)

    def test_same_points(self):
        """Test identical points"""
        coords = np.array([[0, 0], [0, 0]])
        dist = compute_distance_matrix(coords)
        np.testing.assert_array_equal(dist, 0)


class TestDataValidator(unittest.TestCase):
    """Data Validator Tests"""

    def test_valid_coords(self):
        """Test valid coordinates"""
        coords = np.array([[0, 0], [1, 0], [1, 1]])
        self.assertTrue(DataValidator.validate_coords(coords))

    def test_nan_coords(self):
        """Test coordinates with NaN values"""
        coords = np.array([[0, 0], [np.nan, 0], [1, 1]])
        self.assertFalse(DataValidator.validate_coords(coords))

    def test_inf_coords(self):
        """Test coordinates with infinity values"""
        coords = np.array([[0, 0], [np.inf, 0], [1, 1]])
        self.assertFalse(DataValidator.validate_coords(coords))

    def test_duplicate_points(self):
        """Test duplicate points"""
        coords = np.array([[0, 0], [0, 0], [1, 1]])
        self.assertFalse(DataValidator.validate_coords(coords))

    def test_collinear_points(self):
        """Test that valid collinear TSP instances are accepted"""
        coords = np.array([[0, 0], [1, 0], [2, 0]])
        self.assertTrue(DataValidator.validate_coords(coords))

    def test_filter_instances(self):
        """Test instance filtering"""
        coords_small = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
        coords_large = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 2]] * 6)

        inst1 = TSPInstance("small", coords_small)
        inst2 = TSPInstance("large", coords_large)

        instances = [inst1, inst2]
        filtered = DataValidator.filter_instances(instances, max_cities=4)

        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].name, "small")


class TestDataProcessor(unittest.TestCase):
    """Data Processor Tests"""

    def setUp(self):
        self.test_csv = "tsp_instances_dataset.csv"

    @unittest.skipUnless(os.path.exists("tsp_instances_dataset.csv"),
                        "Test dataset not found")
    def test_load_from_csv(self):
        """Test CSV loading"""
        processor = DataProcessor(self.test_csv)
        instances = processor.load_from_csv()
        self.assertIsInstance(instances, list)
        if instances:
            self.assertIsInstance(instances[0], TSPInstance)

    @unittest.skipUnless(os.path.exists("tsp_instances_dataset.csv"),
                        "Test dataset not found")
    def test_preprocess(self):
        """Test preprocessing pipeline"""
        processor = DataProcessor(self.test_csv)
        processor.load_from_csv()
        filtered = processor.preprocess(max_cities=25)

        for inst in filtered:
            self.assertLessEqual(inst.n, 25)
            self.assertGreaterEqual(inst.n, 3)

    def test_incomplete_instance_is_skipped(self):
        """Test that a row cannot silently lose declared cities"""
        import pandas as pd

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
            csv_path = handle.name
        try:
            pd.DataFrame([{
                'TSP_Instance': 'broken',
                'Num_Cities': 4,
                'City_1_X': 0, 'City_1_Y': 0,
                'City_2_X': 1, 'City_2_Y': 0,
                'City_3_X': 0, 'City_3_Y': 1,
            }]).to_csv(csv_path, index=False)
            self.assertEqual(DataProcessor(csv_path).load_from_csv(), [])
        finally:
            os.unlink(csv_path)

    def test_small_dataset_split(self):
        """Test proportional splitting when fewer rows than requested exist"""
        processor = DataProcessor()
        coords = np.array([[0, 0], [1, 0], [0, 1]])
        processor.filtered_instances = [TSPInstance(str(i), coords) for i in range(10)]

        train, val, test = processor.split_dataset()

        self.assertEqual((len(train), len(val), len(test)), (6, 2, 2))


class TestSmallDatasetBuilder(unittest.TestCase):
    """Reproducible small-instance dataset generation."""

    def test_builds_prefix_city_subset(self):
        """Test that the derived dataset keeps a deterministic city prefix."""
        import pandas as pd

        source = pd.DataFrame([{
            "TSP_Instance": "route",
            "Num_Cities": 4,
            "City_1_X": 0, "City_1_Y": 0,
            "City_2_X": 1, "City_2_Y": 0,
            "City_3_X": 1, "City_3_Y": 1,
            "City_4_X": 0, "City_4_Y": 1,
        }])
        with tempfile.TemporaryDirectory() as directory:
            source_path = os.path.join(directory, "source.csv")
            output_path = os.path.join(directory, "small.csv")
            source.to_csv(source_path, index=False)

            count = build_small_dataset(source_path, output_path, city_count=3)
            result = pd.read_csv(output_path)

        self.assertEqual(count, 1)
        self.assertEqual(result.loc[0, "TSP_Instance"], "route_small")
        self.assertEqual(result.loc[0, "Num_Cities"], 3)
        self.assertNotIn("City_4_X", result.columns)


if __name__ == '__main__':
    unittest.main()
