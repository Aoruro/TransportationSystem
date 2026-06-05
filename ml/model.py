"""
Machine Learning Model Module for TSP

Features:
- Feature engineering: 10 features (8 original + max_dist_to_unvisited + mst_lower_bound_ratio)
- Model training: Random Forest, 2-layer MLP
- 5-fold cross-validation
- Random Forest feature importance output
- Prediction caching for performance optimization

Reference notes:
- scikit-learn is used for model training and evaluation utilities.
- TSP-specific features are project-designed from the local state representation.
- Full bibliographic details and reuse rationale are in REFERENCES.md.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from search.utils import prim_mst


class FeatureExtractor:
    """
    TSP Feature Extractor
    
    Feature List (10 features total):
    1. visited_ratio: Ratio of visited cities
    2. mean_dist_from_current: Average distance from current city to all cities
    3. min_dist_from_current: Minimum distance from current city
    4. min_candidate_dist: Minimum distance to unvisited cities
    5. mean_candidate_dist: Average distance to unvisited cities
    6. unvisited_ratio: Ratio of unvisited cities
    7. max_dist_to_unvisited: Maximum distance to unvisited cities
    8. mst_lower_bound_ratio: MST lower bound / accumulated cost
    9. current_city: Current city index (normalized)
    10. next_city: Candidate next city index (normalized)
    """

    def __init__(self, n_cities: int):
        """
        Initialize feature extractor.
        
        Args:
            n_cities: Total number of cities
        """
        self.n_cities = n_cities

    def extract(self, path: List[int], dist_matrix: np.ndarray,
                cost_so_far: float = 0.0,
                next_city: Optional[int] = None) -> np.ndarray:
        """
        Extract features for a given state.
        
        Args:
            path: Current path
            dist_matrix: NxN distance matrix
            cost_so_far: Accumulated cost to current state
            
        Returns:
            10-dimensional feature vector
        """
        n = len(dist_matrix)
        current = path[-1] if path else 0
        visited = set(path)

        visited_ratio = len(path) / n

        all_dists = dist_matrix[current]
        mean_dist_from_current = np.mean(all_dists)
        other_dists = np.delete(all_dists, current)
        min_dist_from_current = np.min(other_dists) if len(other_dists) else 0.0

        unvisited = [i for i in range(n) if i not in visited]
        unvisited_ratio = len(unvisited) / n if n > 0 else 0

        if unvisited:
            candidate_dists = [dist_matrix[current][u] for u in unvisited]
            min_candidate_dist = min(candidate_dists)
            mean_candidate_dist = np.mean(candidate_dists)
            max_dist_to_unvisited = max(candidate_dists)
        else:
            min_candidate_dist = 0.0
            mean_candidate_dist = 0.0
            max_dist_to_unvisited = 0.0

        if len(unvisited) > 1 and current < len(dist_matrix):
            sub_matrix = np.array([[dist_matrix[u][v] for v in unvisited] for u in unvisited])
            mst_cost = prim_mst(len(unvisited), sub_matrix)
        else:
            mst_cost = 0.0

        mst_lower_bound_ratio = mst_cost / cost_so_far if cost_so_far > 0 else 0.0

        features = [
            visited_ratio,
            mean_dist_from_current,
            min_dist_from_current,
            min_candidate_dist,
            mean_candidate_dist,
            unvisited_ratio,
            max_dist_to_unvisited,
            mst_lower_bound_ratio,
            current / n if n > 0 else 0,
            next_city / n if next_city is not None and n > 0 else 0
        ]

        return np.array(features, dtype=np.float32)

    def extract_with_next(self, path: List[int], dist_matrix: np.ndarray,
                         next_city: int, cost_so_far: float = 0.0) -> np.ndarray:
        """
        Extract features including candidate next city.
        
        Args:
            path: Current path
            dist_matrix: NxN distance matrix
            next_city: Candidate next city
            cost_so_far: Accumulated cost to current state
            
        Returns:
            10-dimensional feature vector including the candidate city
        """
        return self.extract(path, dist_matrix, cost_so_far, next_city=next_city)


class TSPMLModel:
    """
    TSP Machine Learning Model
    
    Supports:
    - Random Forest classifier
    - 2-layer MLP classifier
    - 5-fold cross-validation
    - Feature importance output (Random Forest only)
    """

    FEATURE_NAMES = [
        'visited_ratio', 'mean_dist_from_current', 'min_dist_from_current',
        'min_candidate_dist', 'mean_candidate_dist', 'unvisited_ratio',
        'max_dist_to_unvisited', 'mst_lower_bound_ratio',
        'current_city_normalized', 'next_city_normalized'
    ]

    def __init__(self, model_type: str = 'rf'):
        """
        Initialize ML model.
        
        Args:
            model_type: 'rf' for Random Forest, 'mlp' for MLP
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.feature_extractor = None

    def _create_model(self):
        """Create model instance based on type."""
        if self.model_type == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation='relu',
                max_iter=500,
                random_state=42,
                early_stopping=True
            )
        else:
            return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

    def train(self, X: np.ndarray, y: np.ndarray,
             cv_folds: int = 5) -> Dict:
        """
        Train the model.
        
        Args:
            X: Feature matrix
            y: Label vector
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with training metrics
        """
        X = np.asarray(X)
        y = np.asarray(y)
        if X.ndim != 2 or len(X) != len(y) or len(y) < 2:
            raise ValueError("Training data must contain at least two feature rows")
        if len(np.unique(y)) < 2:
            raise ValueError("Training labels must contain both positive and negative samples")

        class_counts = np.unique(y, return_counts=True)[1]
        n_classes = len(class_counts)
        test_rows = int(np.ceil(len(y) * 0.2))
        train_rows = len(y) - test_rows
        stratify = (
            y if np.min(class_counts) >= 2
            and test_rows >= n_classes
            and train_rows >= n_classes
            else None
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        self.model = self._create_model()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        effective_folds = min(cv_folds, int(np.min(class_counts)))
        cv_scores = (
            cross_val_score(self.model, X, y, cv=effective_folds)
            if effective_folds >= 2 else np.array([float('nan')])
        )

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_samples': len(y)
        }

        if self.model_type == 'rf' and hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_
            metrics['feature_importance'] = dict(zip(self.FEATURE_NAMES, self.feature_importance))

        return metrics

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix
            
        Returns:
            Probabilities for each class
            
        Raises:
            ValueError: If model not trained
        """
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class labels
            
        Raises:
            ValueError: If model not trained
        """
        if self.model is None:
            raise ValueError("Model not trained")
        return self.model.predict(X)

    def get_feature_importance(self) -> Optional[Dict]:
        """
        Get feature importance (Random Forest only).
        
        Returns:
            Dictionary of feature importance scores, or None
        """
        if self.model_type != 'rf':
            return None
        if self.feature_importance is None:
            return None
        return dict(zip(self.FEATURE_NAMES, self.feature_importance))


class ModelCache:
    """
    Model Prediction Cache
    
    Reduces repeated prediction overhead by caching results.
    """

    def __init__(self, model: TSPMLModel):
        """
        Initialize cache.
        
        Args:
            model: TSPMLModel instance
        """
        self.model = model
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def predict_key(self, features: np.ndarray) -> str:
        """Generate cache key from features."""
        return tuple(np.round(features, 4).tolist())

    def predict(self, features: np.ndarray) -> float:
        """
        Predict with caching.
        
        Args:
            features: Feature vector
            
        Returns:
            Probability of positive class
        """
        key = self.predict_key(features)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        prob = self.model.predict_proba(features.reshape(1, -1))[0][1]
        self.cache[key] = prob
        return prob

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache)
        }

    def clear(self):
        """Clear cache."""
        self.cache = {}
        self.hits = 0
        self.misses = 0
