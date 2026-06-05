"""
ml module initialization file

Machine learning models and learning-enhanced A* algorithm:
- TSPMLModel: ML model wrapper (Random Forest/MLP)
- FeatureExtractor: 10-feature engineering
- PseudoLabelGenerator: Training data generation
- LearningAStar: ML-enhanced A* search

Reference notes:
- scikit-learn supports model training while routing features are local.
- Full bibliographic details and reuse rationale are in REFERENCES.md.
"""
from .model import TSPMLModel, FeatureExtractor
from .pseudo_labels import PseudoLabelGenerator
from .learning_astar import LearningAStar

__all__ = [
    'TSPMLModel', 'FeatureExtractor',
    'PseudoLabelGenerator', 'LearningAStar'
]
