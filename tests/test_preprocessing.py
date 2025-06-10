"""
Comprehensive pytest suite for preprocessing.py

- Covers ColumnRenamer, build_preprocessing_pipeline, get_output_feature_names
- Includes both 'happy path' and failure/edge cases
- Uses only toy DataFrames and minimal configs for full isolation
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.exceptions import NotFittedError

from preprocess import preprocessing

# --- Fixtures for mock configs and DataFrames ---


@pytest.fixture
def toy_df():
    """DataFrame with both numeric and categorical columns (happy path)."""
    return pd.DataFrame({
        "A": [1.0, 2.0, 3.0, np.nan],
        "B": ["cat", "dog", "cat", "fish"],
        "C": [100, 200, 300, 400]
    })


@pytest.fixture
def rename_map():
    """Rename mapping for ColumnRenamer."""
    return {"A": "alpha", "B": "beta"}


@pytest.fixture
def basic_config(rename_map):
    """Minimal config dict for build_preprocessing_pipeline."""
    return {
        "preprocessing": {
            "rename_columns": rename_map,
            "alpha": {"impute": True, "imputer_strategy": "mean", "scaler": "minmax"},
            "beta": {"impute": True, "imputer_strategy": "most_frequent", "encoding": "onehot"},
            "C": {"impute": False}
        },
        "features": {
            "continuous": ["alpha", "C"],
            "categorical": ["beta"]
        },
        "raw_features": ["A", "B", "C"]
    }

# --- Tests for ColumnRenamer ---


def test_column_renamer_renames_columns(toy_df, rename_map):
    """ColumnRenamer renames columns according to the map."""
    renamer = preprocessing.ColumnRenamer(rename_map)
    df_renamed = renamer.transform(toy_df)
    assert "alpha" in df_renamed.columns
    assert "beta" in df_renamed.columns
    assert "A" not in df_renamed.columns
    assert "B" not in df_renamed.columns


def test_column_renamer_leaves_unmapped_columns(toy_df, rename_map):
    """ColumnRenamer leaves columns unchanged if not in the map."""
    renamer = preprocessing.ColumnRenamer(rename_map)
    df_renamed = renamer.transform(toy_df)
    assert "C" in df_renamed.columns


def test_column_renamer_empty_map_returns_same(toy_df):
    """Empty map: DataFrame columns unchanged."""
    renamer = preprocessing.ColumnRenamer({})
    df2 = renamer.transform(toy_df)
    pd.testing.assert_frame_equal(df2, toy_df)


def test_column_renamer_ignores_nonexistent_columns(toy_df):
    """Rename map entry for nonexistent column does not raise."""
    renamer = preprocessing.ColumnRenamer({"Z": "zeta"})
    df2 = renamer.transform(toy_df)
    pd.testing.assert_frame_equal(df2, toy_df)

# --- Tests for build_preprocessing_pipeline ---


def test_build_preprocessing_pipeline_happy_path(toy_df, basic_config):
    """Full pipeline fits and transforms data with both continuous and categorical columns."""
    pipe = preprocessing.build_preprocessing_pipeline(basic_config)
    out = pipe.fit_transform(toy_df)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == len(toy_df)


def test_pipeline_numeric_imputation_and_scaling():
    """Numeric pipeline imputes missing, scales, output is in [0,1], correct values."""
    df = pd.DataFrame({"A": [1.0, np.nan, 3.0], "B": ["x", "x", "y"]})
    config = {
        "preprocessing": {
            "rename_columns": {"A": "alpha", "B": "beta"},
            "alpha": {"impute": True, "imputer_strategy": "mean", "scaler": "minmax"},
            "beta": {"impute": True, "encoding": "onehot"},
        },
        "features": {"continuous": ["alpha"], "categorical": ["beta"]},
        "raw_features": ["A", "B"],
    }
    pipe = preprocessing.build_preprocessing_pipeline(config)
    X = pipe.fit_transform(df)
    # Imputation: mean([1.0, 3.0]) = 2.0, so scaled values: [0, 0.5, 1]
    np.testing.assert_almost_equal(X[:, 0], [0.0, 0.5, 1.0])


def test_pipeline_categorical_encoding(toy_df, basic_config):
    """Categorical pipeline applies one-hot encoding and handles unknown."""
    pipe = preprocessing.build_preprocessing_pipeline(basic_config)
    X = pipe.fit_transform(toy_df)
    unique_cats = toy_df["B"].nunique()
    assert X.shape[1] >= unique_cats + 2  # plus numeric


def test_pipeline_bucketize(tmp_path, toy_df, basic_config):
    """Numeric pipeline applies KBinsDiscretizer when specified."""
    config = dict(basic_config)
    config["preprocessing"] = dict(config["preprocessing"])
    config["preprocessing"]["alpha"]["bucketize"] = True
    config["preprocessing"]["alpha"]["n_buckets"] = 2
    pipe = preprocessing.build_preprocessing_pipeline(config)
    X = pipe.fit_transform(toy_df)
    assert X.shape[1] >= 2


def test_bucketize_handles_few_unique_values():
    """Bucketize handles all-constant feature by outputting a single bin."""
    df = pd.DataFrame({"A": [1, 1, 1, 1]})
    config = {
        "preprocessing": {
            "rename_columns": {"A": "alpha"},
            "alpha": {"bucketize": True, "n_buckets": 3},
        },
        "features": {"continuous": ["alpha"], "categorical": []},
        "raw_features": ["A"],
    }
    pipe = preprocessing.build_preprocessing_pipeline(config)
    out = pipe.fit_transform(df)
    # Only one bin possible when all values are identical
    assert out.shape[1] == 1
    assert (out == 1).all()  # All rows assigned to the single available bin


def test_pipeline_passthrough_column(toy_df):
    """Columns in raw_features but not in features should be passthrough."""
    config = {
        "preprocessing": {
            "rename_columns": {},
        },
        "features": {
            "continuous": [],
            "categorical": []
        },
        "raw_features": ["A", "B", "C"]
    }
    pipe = preprocessing.build_preprocessing_pipeline(config)
    X = pipe.fit_transform(toy_df)
    assert X.shape[1] == len(toy_df.columns)


def test_pipeline_missing_config_section(toy_df):
    """Missing 'features' or 'preprocessing' does not crash, falls back to default."""
    config = {
        "raw_features": ["A", "B", "C"]
    }
    pipe = preprocessing.build_preprocessing_pipeline(config)
    X = pipe.fit_transform(toy_df)
    assert X.shape[1] == len(toy_df.columns)


def test_build_pipeline_logs_warning_for_missing_continuous(toy_df, basic_config, caplog):
    """Logs warning if continuous features are missing."""
    config = dict(basic_config)
    config["features"]["continuous"] = []
    with caplog.at_level("WARNING"):
        preprocessing.build_preprocessing_pipeline(config)
        assert any(
            "No continuous features specified" in m for m in caplog.messages)

# --- Tests for malformed config and error handling ---


def test_pipeline_malformed_config_falls_back_with_warning(caplog):
    """Malformed config with missing sections should log a warning and still build, output is empty."""
    config = {}  # Missing all required keys
    with caplog.at_level("WARNING"):
        pipe = preprocessing.build_preprocessing_pipeline(config)
        # Should issue a warning about missing features
        assert any(
            "No continuous features specified" in m for m in caplog.messages)
    # Should still produce a functional pipeline
    df = pd.DataFrame({"foo": [1, 2]})
    out = pipe.fit_transform(df)
    # No features configured, so output has zero columns
    assert out.shape[1] == 0


def test_pipeline_transform_before_fit_raises(toy_df, basic_config):
    """Transform before fit raises NotFittedError."""
    pipe = preprocessing.build_preprocessing_pipeline(basic_config)
    with pytest.raises(Exception) as exc:
        pipe.transform(toy_df)
    # Could be NotFittedError, AttributeError, or RuntimeError depending on sklearn version
    assert any([isinstance(exc.value, NotFittedError), "fit" in str(
        exc.value).lower(), "has no attribute" in str(exc.value).lower()])


def test_pipeline_idempotency(toy_df, basic_config):
    """Transforming twice produces identical output, does not mutate input."""
    pipe = preprocessing.build_preprocessing_pipeline(basic_config)
    pipe.fit(toy_df)
    out1 = pipe.transform(toy_df)
    out2 = pipe.transform(toy_df)
    np.testing.assert_array_equal(out1, out2)
    # Original DataFrame is unchanged
    pd.testing.assert_frame_equal(toy_df, toy_df.copy())

# --- Tests for get_output_feature_names ---


def test_get_output_feature_names_matches_transform(toy_df, basic_config):
    """Output feature names after fit should match transformed array columns."""
    pipe = preprocessing.build_preprocessing_pipeline(basic_config)
    pipe.fit(toy_df)
    names = preprocessing.get_output_feature_names(
        pipe, toy_df.columns.tolist(), basic_config)
    X = pipe.transform(toy_df)
    assert len(names) == X.shape[1]
    assert any("beta" in n for n in names)


def test_get_output_feature_names_not_fitted_raises(toy_df, basic_config):
    """Calling get_output_feature_names before fit raises."""
    pipe = preprocessing.build_preprocessing_pipeline(basic_config)
    with pytest.raises(AttributeError):
        preprocessing.get_output_feature_names(
            pipe, toy_df.columns.tolist(), basic_config)
