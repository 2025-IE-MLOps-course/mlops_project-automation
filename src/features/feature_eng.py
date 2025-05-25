"""
feature_eng.py

Feature engineering transformers for MLOps pipelines.
- All new features should be implemented as scikit-learn compatible transformers
- Each transformer is documented for academic and production clarity
- Designed for integration into preprocessing and model training pipelines
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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

    def __init__(self, icd10_flags):
        self.icd10_flags = icd10_flags

    def fit(self, X, y=None):
        # No fitting necessary
        return self

    def transform(self, X: pd.DataFrame):
        X = X.copy()

        # Ensure every ICD-10 flag column exists
        for col in self.icd10_flags:
            if col not in X.columns:
                X[col] = 0

        # Coerce to numeric; strings become NaN → fill with 0
        icd_numeric = (
            X[self.icd10_flags if isinstance(self.icd10_flags, list) else [
                self.icd10_flags]]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
        )

        X["risk_score"] = icd_numeric.sum(axis=1)
        return X

# Template for future engineered features:
# class MyFeatureTransformer(BaseEstimator, TransformerMixin):
#     """
#     Short description and academic motivation.
#     If this transformer depends on config (e.g., feature names), pass them as constructor arguments.
#     """
#     def __init__(self, param_from_config):
#         self.param_from_config = param_from_config
#     def fit(self, X, y=None):
#         return self
#     def transform(self, X):
#         X = X.copy()
#         # Feature engineering logic here, using self.param_from_config if needed
#         return X


FEATURE_TRANSFORMERS = {
    "risk_score": lambda config: RiskScore(config["icd10_chapter_flags"])
}
