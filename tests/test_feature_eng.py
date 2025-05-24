"""
test_features.py

Unit test for feature engineering transformers.
- Validates correctness and robustness of all new engineered features
- Ensures each feature returns expected outputs for both typical and edge case inputs
- Required for reproducibility and for maintaining ML pipeline quality over time

All engineered features should be tested in isolation before pipeline integration to ensure:
- No hidden assumptions or side effects
- Consistent output types and shapes
- Reproducible, deterministic results for any input

This test covers the risk_score feature, a sum of ICD-10 flags (proxy for patient multimorbidity risk).
"""

import pandas as pd
from features.feature_eng import RiskScore

def test_risk_score_basic():
    """
    Test that RiskScore transformer correctly sums ICD-10 flags into the risk_score feature.

    - Input: minimal DataFrame with all ICD10 flags (1 for row 0, 0 for row 1)
    - Expected: row 0 gets risk_score = 16, row 1 gets risk_score = 0

    Rationale:
    ----------
    - Validates column presence and correct summation logic
    - Prevents silent errors if ICD-10 columns are changed or missing
    - Supports reproducible, tested ML code as required in production and academic workflows
    """
    # Construct minimal DataFrame: each ICD flag column, two rows (all 1s, all 0s)
    df = pd.DataFrame({k: [1, 0] for k in [
        "A", "B", "C", "D", "E", "F", "H", "I", "J",
        "K", "L", "M", "N", "R", "S", "T", "V"]})
    # Apply transformer
    out = RiskScore().fit_transform(df)
    # Test output
    assert "risk_score" in out.columns, "Output missing 'risk_score' column"
    assert out["risk_score"].iloc[0] == 17, "Sum should be 17 for all 1s"
    assert out["risk_score"].iloc[1] == 0, "Sum should be 0 for all 0s"
