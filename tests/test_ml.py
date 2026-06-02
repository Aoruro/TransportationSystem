"""
Machine Learning Module Tests

Test coverage:
- Feature extraction
- Model training and prediction
- Pseudo-label generation
- Learning-enhanced A*
"""

import unittest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.model import FeatureExtractor, TSPMLModel, ModelCache
from ml.pseudo_labels import PseudoLabelGenerator
from ml.learning_astar import LearningAStar


class TestFeatureExtractor(unittest.TestCase):
    """Feature Extractor Tests"""

    def setUp(self):
        self.n = 5
        self.dist = np.random.rand(self.n, self.n)
        self.dist = (self.dist + self.dist.T) / 2
        np.fill_diagonal(self.dist, 0)
        self.extractor = FeatureExtractor(self.n)

    def test_extract_features(self):
        """Test feature extraction"""
        path = [0, 1, 2]
        features = self.extractor.extract(path, self.dist, cost_so_far=1.0)

        self.assertEqual(len(features), 10)
        self.assertAlmostEqual(features[0], 3/5)

    def test_extract_with_next(self):
        """Test feature extraction with next city"""
        path = [0, 1]
        features = self.extractor.extract_with_next(path, self.dist, next_city=2, cost_so_far=0.5)

        self.assertEqual(len(features), 10)
        self.assertAlmostEqual(features[-1], 2/self.n)

    def test_unvisited_features(self):
        """Test unvisited city features"""
        path = [0]
        features = self.extractor.extract(path, self.dist)

        self.assertGreater(features[4], 0)
        self.assertGreater(features[5], 0)

    def test_min_distance_excludes_current_city(self):
        """Test that the nearest-city feature is not the diagonal zero"""
        path = [0]
        features = self.extractor.extract(path, self.dist)
        self.assertGreater(features[2], 0)


class TestTSPMLModel(unittest.TestCase):
    """ML Model Tests"""

    def setUp(self):
        np.random.seed(42)
        self.X = np.random.rand(100, 8)
        self.y = np.random.randint(0, 2, 100)
        self.model = TSPMLModel(model_type='rf')

    def test_train_rf(self):
        """Test Random Forest training"""
        metrics = self.model.train(self.X, self.y, cv_folds=3)

        self.assertIn('accuracy', metrics)
        self.assertIn('cv_mean', metrics)
        self.assertGreater(metrics['accuracy'], 0)

    def test_train_mlp(self):
        """Test MLP training"""
        mlp_model = TSPMLModel(model_type='mlp')
        metrics = mlp_model.train(self.X, self.y, cv_folds=3)

        self.assertIn('accuracy', metrics)

    def test_predict_proba(self):
        """Test probability prediction"""
        self.model.train(self.X, self.y)
        probs = self.model.predict_proba(self.X[:5])

        self.assertEqual(probs.shape, (5, 2))
        self.assertTrue(np.all(probs >= 0))
        self.assertTrue(np.all(probs <= 1))

    def test_predict(self):
        """Test class prediction"""
        self.model.train(self.X, self.y)
        preds = self.model.predict(self.X[:5])

        self.assertEqual(len(preds), 5)
        self.assertTrue(np.all((preds == 0) | (preds == 1)))

    def test_feature_importance(self):
        """Test feature importance calculation"""
        metrics = self.model.train(self.X, self.y)

        self.assertIn('feature_importance', metrics)
        importance = metrics['feature_importance']
        self.assertEqual(len(importance), 8)


class TestModelCache(unittest.TestCase):
    """Model Cache Tests"""

    def setUp(self):
        self.model = TSPMLModel(model_type='rf')
        X = np.random.rand(50, 8)
        y = np.random.randint(0, 2, 50)
        self.model.train(X, y)
        self.cache = ModelCache(self.model)

    def test_cache_miss(self):
        """Test cache miss scenario"""
        features = np.random.rand(8)
        prob1 = self.cache.predict(features)
        self.assertEqual(self.cache.misses, 1)
        self.assertEqual(self.cache.hits, 0)

    def test_cache_hit(self):
        """Test cache hit scenario"""
        features = np.random.rand(8)
        prob1 = self.cache.predict(features)
        prob2 = self.cache.predict(features)

        self.assertEqual(self.cache.hits, 1)
        self.assertAlmostEqual(prob1, prob2)

    def test_stats(self):
        """Test cache statistics"""
        features = np.random.rand(8)
        self.cache.predict(features)
        self.cache.predict(features)

        stats = self.cache.get_stats()
        self.assertEqual(stats['misses'], 1)
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['cache_size'], 1)


class TestPseudoLabelGenerator(unittest.TestCase):
    """Pseudo-label Generator Tests"""

    def setUp(self):
        np.random.seed(42)
        n = 6
        coords = np.random.rand(n, 2) * 100
        self.dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.dist[i][j] = np.linalg.norm(coords[i] - coords[j])
        self.generator = PseudoLabelGenerator(self.dist)

    def test_generate_optimal_solution(self):
        """Test optimal solution generation"""
        path, cost = self.generator.generate_optimal_solution()

        self.assertEqual(len(path), len(self.dist) + 1)
        self.assertEqual(path[0], path[-1])
        self.assertEqual(set(path[:-1]), set(range(len(self.dist))))

    def test_generate_training_data(self):
        """Test training data generation"""
        path, cost = self.generator.generate_optimal_solution()
        X, y = self.generator.generate_training_data(path, cost)

        self.assertGreater(len(X), 0)
        self.assertEqual(len(X), len(y))
        self.assertEqual(X.shape[1], 10)

    def test_positive_samples(self):
        """Test positive sample generation"""
        path, cost = self.generator.generate_optimal_solution()
        X, y = self.generator.generate_training_data(path, cost)

        self.assertEqual(np.sum(y), len(path) - 2)


class TestLearningAStar(unittest.TestCase):
    """Learning-enhanced A* Tests"""

    def setUp(self):
        np.random.seed(42)
        n = 8
        coords = np.random.rand(n, 2) * 100
        self.dist = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                self.dist[i][j] = np.linalg.norm(coords[i] - coords[j])
        self.solver = LearningAStar(self.dist)

    def test_solve_without_model(self):
        """Test solving without ML model"""
        result = self.solver.solve()

        self.assertIn('success', result)
        self.assertIn('cost', result)
        self.assertIn('nodes_expanded', result)

    def test_solve_with_model(self):
        """Test solving with ML model"""
        from ml.pseudo_labels import generate_pseudo_labels_batch
        from data.data_processor import TSPInstance

        np.random.seed(42)
        coords = np.random.rand(8, 2) * 100
        inst = TSPInstance("test", coords=coords, dist_matrix=self.dist)

        X, y = generate_pseudo_labels_batch([inst])

        if len(X) > 0:
            model = TSPMLModel(model_type='rf')
            model.train(X, y)

            solver = LearningAStar(self.dist, model=model, lambda_param=0.3)
            result = solver.solve()

            self.assertTrue(result['success'])

    def test_model_guidance_preserves_optimality(self):
        """Test that ML guidance cannot make a completed tour look cheaper"""
        from search.utils import held_karp

        class CandidateBiasedModel:
            def predict_proba(self, X):
                probs = X[:, 9]
                return np.column_stack((1 - probs, probs))

        _, optimal_cost = held_karp(self.dist)
        solver = LearningAStar(self.dist, model=CandidateBiasedModel(), lambda_param=10.0)
        result = solver.solve()

        self.assertAlmostEqual(result['cost'], optimal_cost)


if __name__ == '__main__':
    unittest.main()
