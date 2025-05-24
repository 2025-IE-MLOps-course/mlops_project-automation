"""
features.py

Feature engineering transformers for MLOps pipelines.
- All new features should be implemented as scikit-learn compatible transformers
- Each transformer is documented for academic and production clarity
- Designed for integration into preprocessing and model training pipelines
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ICD-10 diagnostic group flags available in the raw dataset
ICD10_CHAPTER_FLAGS = [
    "A", "B", "C", "D", "E", "F", "H", "I", "J",
    "K", "L", "M", "N", "R", "S", "T", "V"
]


class RiskScore(BaseEstimator, TransformerMixin):
    """
    Adds a 'RiskScore_count' column: sum of all ICD-10 chapter flags per patient.

    Clinical motivation:
    - Multimorbidity is an established risk factor for opioid use disorder
    - Used to improve classification of opioid abuse disorder in this context

    Usage:
        pipeline = Pipeline([
            ('RiskScore', RiskScore()),
            ...
        ])
    """

    def fit(self, X, y=None):
        # No fitting necessary
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()
        X["risk_score"] = X[ICD10_CHAPTER_FLAGS].sum(axis=1)
        return X

# Template for future engineered features:
# class MyFeatureTransformer(BaseEstimator, TransformerMixin):
#     """
#     Short description and academic motivation
#     """
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         X = X.copy()
#         # Feature engineering logic here
#         return X


# Registry of feature transformers (for flexible pipeline construction)
FEATURE_TRANSFORMERS = {
    "risk_score": RiskScore
}

