"""
Comprehensive pytest suite for model.py

- Covers model training, evaluation, artifact saving/loading, and config error handling
- Uses only toy DataFrames and minimal configs
- All file artifacts saved to tmp_path for isolation
"""

import pytest
import numpy as np
import pandas as pd
import pickle
import os

from model.model import (
    train_model, evaluate_model, save_artifact, format_metrics, run_model_pipeline, MODEL_REGISTRY
)

import sys
print("PYTHONPATH:", sys.path)


# --- Fixtures for toy data and config ---


@pytest.fixture
def toy_df():
    """Tiny toy DataFrame for binary classification."""
    return pd.DataFrame({
        "f1": [0, 1, 0, 1, 0, 1],
        "f2": [2, 3, 1, 5, 2, 1],
        "target": [0, 1, 0, 1, 0, 1],
    })


@pytest.fixture
def minimal_config(tmp_path):
    """Minimal config dictionary for a full pipeline run, with tmp_path output."""
    return {
        "raw_features": ["f1", "f2", "target"],
        "target": "target",
        "data_split": {
            "test_size": 0.33,
            "valid_size": 0.33,
            "random_state": 0
        },
        "features": {
            "engineered": ["f1", "f2"]
        },
        "preprocessing": {
            "rename_columns": {},
            "f1": {"impute": False, "scaler": "minmax"},
            "f2": {"impute": False, "scaler": "minmax"}
        },
        "model": {
            "active": "decision_tree",
            "decision_tree": {"params": {"max_depth": 2}}
        },
        "artifacts": {
            "splits_dir": str(tmp_path / "splits"),
            "processed_dir": str(tmp_path / "processed"),
            "preprocessing_pipeline": str(tmp_path / "preproc.pkl"),
            "model_path": str(tmp_path / "model.pkl"),
            "metrics_path": str(tmp_path / "metrics.json")
        }
    }


@pytest.fixture
def Xy():
    """Toy X, y for model training."""
    X = np.array([[0, 2], [1, 3], [0, 1], [1, 5], [0, 2], [1, 1]])
    y = np.array([0, 1, 0, 1, 0, 1])
    return X, y

# --- Tests for model training and registry ---


def test_train_model_supported_types(Xy):
    """train_model trains and returns model for all supported types."""
    X, y = Xy
    for mtype in MODEL_REGISTRY:
        model = train_model(X, y, mtype, {})
        assert hasattr(model, "fit") and hasattr(model, "predict")


def test_train_model_unsupported_type(Xy):
    """Unsupported model type raises ValueError."""
    X, y = Xy
    with pytest.raises(ValueError):
        train_model(X, y, "foobar_model", {})


def test_train_model_with_params(Xy):
    """train_model passes params to sklearn model."""
    X, y = Xy
    model = train_model(X, y, "decision_tree", {"max_depth": 1})
    assert hasattr(model, "max_depth") and model.max_depth == 1

# --- Tests for evaluate_model ---


def test_evaluate_model_basic_metrics(Xy):
    """evaluate_model returns all requested metrics with correct structure."""
    X, y = Xy
    model = train_model(X, y, "decision_tree", {})
    metrics = ["accuracy", "precision", "recall", "f1 score", "roc auc"]
    out = evaluate_model(model, X, y, metrics)
    assert set(out).issuperset(
        {"Accuracy", "Precision (PPV)", "Recall (Sensitivity)", "F1 Score", "ROC AUC"})
    for v in out.values():
        assert isinstance(v, float) or np.isnan(v)


def test_evaluate_model_handles_no_predict_proba(Xy):
    """evaluate_model works even if model lacks predict_proba (force LogisticRegression with no probability)."""
    from sklearn.linear_model import LogisticRegression
    X, y = Xy
    model = LogisticRegression()
    model.fit(X, y)
    # Remove predict_proba artificially
    delattr(model, "predict_proba")
    metrics = ["roc auc"]
    out = evaluate_model(model, X, y, metrics)
    assert "ROC AUC" in out

# --- Tests for artifact saving/loading ---


def test_save_and_load_artifact(tmp_path):
    """save_artifact writes a file that can be loaded back as the same object."""
    arr = np.array([1, 2, 3])
    path = tmp_path / "artifact.pkl"
    save_artifact(arr, str(path))
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    np.testing.assert_array_equal(arr, loaded)

# --- Tests for format_metrics ---


def test_format_metrics_rounding():
    """format_metrics returns dict with all values rounded to specified ndigits."""
    metrics = {"Accuracy": 0.98765, "Precision": 1.0, "Label": "ok"}
    out = format_metrics(metrics, ndigits=3)
    assert out["Accuracy"] == 0.988
    assert out["Precision"] == 1.0
    assert out["Label"] == "ok"

# --- End-to-end pipeline test ---


def test_run_model_pipeline_toy(toy_df, minimal_config, tmp_path):
    """run_model_pipeline executes end-to-end with toy data and saves all expected artifacts."""
    run_model_pipeline(toy_df, minimal_config)
    # All expected files created
    for key, fname in [
        ("splits_dir", "train.csv"),
        ("processed_dir", "train_processed.csv"),
        ("preprocessing_pipeline", None),
        ("model_path", None),
        ("metrics_path", None)
    ]:
        path = minimal_config["artifacts"][key]
        if fname:
            assert os.path.exists(os.path.join(path, fname))
        else:
            assert os.path.exists(path)
    # Metrics file contains expected keys
    import json
    with open(minimal_config["artifacts"]["metrics_path"]) as f:
        metrics = json.load(f)
    assert "validation" in metrics and "test" in metrics

# --- Edge/error cases ---


def test_run_model_pipeline_unsupported_model(toy_df, minimal_config):
    """Unsupported model type in config raises ValueError."""
    cfg = dict(minimal_config)
    cfg["model"] = {"active": "not_real", "not_real": {"params": {}}}
    with pytest.raises(ValueError):
        run_model_pipeline(toy_df, cfg)


def test_train_model_with_missing_data():
    """train_model with missing values in X raises/sklearn error."""
    X = np.array([[0, np.nan], [1, 3], [0, 1]])
    y = np.array([0, 1, 0])
    # Most sklearn models will error on NaN (unless you impute)
    with pytest.raises(ValueError):
        train_model(X, y, "decision_tree", {})

# --- Serialization/Deserialization integration test ---


def test_model_and_preproc_roundtrip(tmp_path, toy_df, minimal_config):
    """Save, reload, and use model and preprocessor artifacts."""
    run_model_pipeline(toy_df, minimal_config)
    # Reload model and preprocessor
    with open(minimal_config["artifacts"]["model_path"], "rb") as f:
        model = pickle.load(f)
    with open(minimal_config["artifacts"]["preprocessing_pipeline"], "rb") as f:
        preproc = pickle.load(f)
    # Should be usable for transform/predict
    X = toy_df[["f1", "f2"]].values
    # Preproc is a Pipeline; should have transform
    Xt = preproc.fit_transform(toy_df[["f1", "f2"]])
    preds = model.predict(Xt)
    assert preds.shape[0] == toy_df.shape[0]
