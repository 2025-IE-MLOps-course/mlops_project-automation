"""
preprocessing.py
================
End-to-end, leakage-proof preprocessing for the MLOps course project.

Design principles
-----------------
1. **Single source of truth for column names**  
   Raw column names appear **only** in `config.yaml`.  
   Renaming is performed exactly once via the `ColumnRenamer` step
   at the very start of the sklearn `Pipeline`.

2. **100 % config-driven**  
   All choices (imputation, scaling, encoding, bucketing, etc.)
   are specified in `config.yaml`.  
   The code never hard-codes feature names or hyperparameters.

3. **Sklearn compatibility and composability**  
   The output of `build_preprocessing_pipeline` is a standard
   `sklearn.pipeline.Pipeline`, ready to be dropped into any model
   training routine or cross-validation loop.

Key sections in this file
-------------------------
* `ColumnRenamer` – minimal transformer that renames columns
* `build_preprocessing_pipeline` – assembles the full pipeline
* `get_output_feature_names` – retrieves final column names after fit
* `run_preprocessing_pipeline` – compatibility stub (returns input unchanged)

This file purposefully contains **no model-specific code**.  
Its single responsibility is to turn a raw `pandas.DataFrame` into
a numeric feature matrix suitable for machine learning.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)

logger = logging.getLogger(__name__)

class ColumnRenamer(BaseEstimator, TransformerMixin):
    """
    Simple, sklearn-compatible transformer that renames DataFrame columns.

    Parameters
    ----------
    rename_map : dict, optional
        Mapping from *old_name* → *new_name*.
        If a column is missing from the map, it is left unchanged.
    """

    def __init__(self, rename_map: dict | None = None):
        self.rename_map = rename_map or {}

    # No computation is needed during fit
    def fit(self, X, y=None):  # noqa: D401  (sklearn signature)
        return self

    # The transform step performs an out-of-place rename and returns a new DF
    def transform(self, X):
        return X.rename(columns=self.rename_map, inplace=False)


# Pipeline-construction helper function
def build_preprocessing_pipeline(config: Dict) -> Pipeline:
    """
    Build a complete sklearn preprocessing pipeline from the YAML config.

    The pipeline has two sequential steps:
    1. `ColumnRenamer` – applies `preprocessing.rename_columns`
    2. `ColumnTransformer` – numeric, categorical and passthrough handling

    Parameters
    ----------
    config : Dict
        Parsed YAML configuration loaded in `src.main`

    Returns
    -------
    sklearn.pipeline.Pipeline
        Ready-to-fit preprocessing pipeline
    """
    # Shorter aliases for frequently accessed sections
    pp_cfg = config.get("preprocessing", {})
    feats_cfg = config.get("features", {})

    continuous: List[str] = feats_cfg.get("continuous", [])
    categorical: List[str] = feats_cfg.get("categorical", [])
    rename_map: dict = pp_cfg.get("rename_columns", {})

    if not continuous:
        logger.warning("No continuous features specified in config")
    if not categorical:
        logger.info(
            "No categorical features specified in config If expected, ignore")

    transformers: list[tuple] = []

    # 1. Build numeric column branch
    for col in continuous:
        steps = []
        col_cfg = pp_cfg.get(col, {})  # column-specific overrides

        # Imputation (mean by default)
        if col_cfg.get("impute", True):
            strategy = col_cfg.get("imputer_strategy", "mean")
            steps.append(("imputer", SimpleImputer(strategy=strategy)))

        # Scaling
        scaler = col_cfg.get("scaler", "minmax")
        if scaler == "minmax":
            steps.append(("scaler", MinMaxScaler()))
        elif scaler == "standard":
            steps.append(("scaler", StandardScaler()))

        # Optional discretisation / bucketing
        if col_cfg.get("bucketize", False):
            steps.append(
                (
                    "bucketize",
                    KBinsDiscretizer(
                        n_bins=col_cfg.get("n_buckets", 4),
                        encode="onehot-dense",
                        strategy="quantile",
                    ),
                )
            )

        # Only add the branch if something actually happens
        if steps:
            transformers.append((f"{col}_num", Pipeline(steps), [col]))

    # 2. Build categorical column branch
    for col in categorical:
        steps = []
        col_cfg = pp_cfg.get(col, {})

        # Imputation (mode by default)
        if col_cfg.get("impute", True):
            strategy = col_cfg.get("imputer_strategy", "most_frequent")
            steps.append(("imputer", SimpleImputer(strategy=strategy)))

        # Encoding
        encoding = col_cfg.get("encoding", "onehot")
        if encoding == "onehot":
            steps.append(("encoder", OneHotEncoder(
                sparse_output=False, handle_unknown="ignore")))
        elif encoding == "ordinal":
            steps.append(("encoder", OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=-1)))

        if steps:
            transformers.append((f"{col}_cat", Pipeline(steps), [col]))

    # 3. Passthrough raw features that have no specific rules
    already_handled = set(continuous + categorical)
    passthrough: list[str] = []

    for raw_col in config.get("raw_features", []):
        # If the column will be renamed, reference its *new* name
        mapped_col = rename_map.get(raw_col, raw_col)
        # Avoid duplicates if the same column is also in numeric / categorical
        if mapped_col not in already_handled:
            passthrough.append(mapped_col)

    if passthrough:
        transformers.append(("passthrough", "passthrough", passthrough))

    # 4. Assemble the full preprocessing
    col_transformer = ColumnTransformer(
        transformers,
        remainder="drop",            # Drop any column not explicitly listed
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        steps=[
            ("rename", ColumnRenamer(rename_map)),  # Must run first
            ("col_transform", col_transformer),
        ]
    )
    return pipeline


# Utility to fetch output feature names after fit
def get_output_feature_names(
    preprocessor: Pipeline,
    input_features: List[str],
    config: Dict,
) -> List[str]:
    """
    Retrieve feature names produced by a *fitted* preprocessing pipeline.

    Useful for:
    * debugging shape mismatches
    * inspecting one-hot column expansion
    * exporting feature importance plots

    Parameters
    ----------
    preprocessor : sklearn.pipeline.Pipeline
        Pipeline returned by `build_preprocessing_pipeline` *after* .fit(...)
    input_features : List[str]
        Original list of columns fed into the pipeline
    config : Dict
        Parsed YAML configuration (not strictly required but handy)

    Returns
    -------
    List[str]
        Ordered list of column names in the transformed numpy array
    """
    feature_names: List[str] = []
    col_transform = preprocessor.named_steps["col_transform"]

    for _, transformer, cols in col_transform.transformers_:
        # Case 1 – transformer exposes its own get_feature_names_out
        if hasattr(transformer, "get_feature_names_out"):
            try:
                feature_names.extend(transformer.get_feature_names_out(cols))
                continue
            except Exception:  # noqa: BLE001  acceptable fallback
                pass

        # Case 2 – transformer is a pipeline; inspect its last step
        if hasattr(transformer, "named_steps"):
            last_step = list(transformer.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                try:
                    feature_names.extend(last_step.get_feature_names_out(cols))
                    continue
                except Exception:  # noqa: BLE001
                    pass

        # Case 3 – passthrough branch or unknown transformer
        if transformer == "passthrough":
            feature_names.extend(cols)
        else:
            # Fallback: use input column names verbatim
            feature_names.extend(cols)

    return feature_names


# CLI for standalone use
if __name__ == "__main__":
    """
    Quick CLI helper so students can run:

        python -m src.preprocess.preprocessing data/raw/opiod_raw_data.csv config.yaml

    and inspect the first few pre-processed rows.
    """
    import sys
    import yaml
    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )

    if len(sys.argv) < 3:
        logging.error(
            "Usage: python -m src.preprocess.preprocessing <raw_data.csv> <config.yaml>"
        )
        sys.exit(1)

    raw_data_path: str = sys.argv[1]
    config_path: str = sys.argv[2]

    # 1. Load raw data and configuration
    df_raw: pd.DataFrame = pd.read_csv(raw_data_path)
    with open(config_path, "r", encoding="utf-8") as fh:
        config: Dict = yaml.safe_load(fh)

    # 2. Build and fit the preprocessing pipeline
    pipeline: Pipeline = build_preprocessing_pipeline(config)
    X_transformed = pipeline.fit_transform(df_raw)

    # 3. Convert the numpy array back to a DataFrame for easy inspection
    feature_names: List[str] = get_output_feature_names(
        preprocessor=pipeline,
        input_features=df_raw.columns.tolist(),
        config=config,
    )
    df_preprocessed = pd.DataFrame(X_transformed, columns=feature_names)

    # 4. Show the result
    pd.set_option("display.width", 120)        # console formatting
    print(df_preprocessed.head())
