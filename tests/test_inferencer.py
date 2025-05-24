"""
pytest suite for inferencer.py

- Covers config/artifact loading, feature checks, transformation, prediction, outputs, errors
- Uses toy data, pickled pipeline/model, and tmp_path for all I/O
"""

import pytest
import pandas as pd
import numpy as np
import pickle
import os
import yaml
from pathlib import Path

from inference import inferencer

@pytest.fixture
def toy_config(tmp_path):
    """Minimal valid config with engineered/feature columns and artifacts."""
    return {
        "raw_features": ["f1", "f2"],
        "features": {"engineered": ["f1", "f2"]},
        "artifacts": {
            "preprocessing_pipeline": str(tmp_path / "preproc.pkl"),
            "model_path": str(tmp_path / "model.pkl"),
        }
    }

@pytest.fixture
def toy_df():
    """Tiny test DataFrame with features."""
    return pd.DataFrame({
        "f1": [0, 1, 0, 1],
        "f2": [1, 2, 2, 1]
    })

@pytest.fixture
def toy_model():
    """Minimal scikit-learn model supporting predict and predict_proba."""
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression()
    X = np.array([[0, 1], [1, 2], [0, 2], [1, 1]])
    y = np.array([0, 1, 0, 1])
    model.fit(X, y)
    return model

@pytest.fixture
def toy_pipeline():
    """Pass-through pipeline for test (no transformation)."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    return Pipeline([('identity', FunctionTransformer())])

def write_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def write_yaml(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f)

def write_csv(df, path):
    df.to_csv(path, index=False)

# --- Test cases ---

def test_run_inference_happy_path(tmp_path, toy_config, toy_df, toy_model, toy_pipeline):
    """Full inference run writes expected predictions."""
    # Write pipeline and model artifacts
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(toy_pipeline, pp_path)
    write_pickle(toy_model, model_path)
    # Write config
    config_path = tmp_path / "config.yaml"
    config = dict(toy_config)
    config["artifacts"]["preprocessing_pipeline"] = str(pp_path)
    config["artifacts"]["model_path"] = str(model_path)
    write_yaml(config, config_path)
    # Write input CSV
    input_path = tmp_path / "input.csv"
    write_csv(toy_df, input_path)
    # Output path
    output_path = tmp_path / "out.csv"
    inferencer.run_inference(str(input_path), str(config_path), str(output_path))
    # Check output file
    df_out = pd.read_csv(output_path)
    assert "prediction" in df_out.columns
    assert df_out.shape[0] == toy_df.shape[0]
    # If model has predict_proba, check
    assert "prediction_proba" in df_out.columns

def test_missing_artifact_file_raises(tmp_path, toy_config, toy_df):
    """Missing pipeline/model file raises FileNotFoundError."""
    # Only write one artifact
    model_path = tmp_path / "model.pkl"
    write_pickle(toy_df, model_path)  # Not a real model but suffices for missing pipeline
    config = dict(toy_config)
    config["artifacts"]["preprocessing_pipeline"] = str(tmp_path / "does_not_exist.pkl")
    config["artifacts"]["model_path"] = str(model_path)
    config_path = tmp_path / "config.yaml"
    write_yaml(config, config_path)
    input_path = tmp_path / "input.csv"
    write_csv(toy_df, input_path)
    output_path = tmp_path / "out.csv"
    with pytest.raises(FileNotFoundError):
        inferencer.run_inference(str(input_path), str(config_path), str(output_path))

def test_missing_feature_column_triggers_exit(tmp_path, toy_config, toy_model, toy_pipeline, monkeypatch):
    """Missing required feature columns in input triggers sys.exit."""
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(toy_pipeline, pp_path)
    write_pickle(toy_model, model_path)
    config = dict(toy_config)
    config["artifacts"]["preprocessing_pipeline"] = str(pp_path)
    config["artifacts"]["model_path"] = str(model_path)
    config_path = tmp_path / "config.yaml"
    write_yaml(config, config_path)
    # Deliberately leave out one column
    df = pd.DataFrame({"f1": [0, 1, 0, 1]})
    input_path = tmp_path / "input.csv"
    write_csv(df, input_path)
    output_path = tmp_path / "out.csv"
    with pytest.raises(SystemExit):
        inferencer.run_inference(str(input_path), str(config_path), str(output_path))

def test_engineered_features_subset(tmp_path, toy_df, toy_model, toy_pipeline):
    """If engineered subset is configured, output is correct shape."""
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(toy_pipeline, pp_path)
    write_pickle(toy_model, model_path)
    config = {
        "raw_features": ["f1", "f2"],
        "features": {"engineered": ["f1"]},
        "artifacts": {
            "preprocessing_pipeline": str(pp_path),
            "model_path": str(model_path),
        }
    }
    config_path = tmp_path / "config.yaml"
    write_yaml(config, config_path)
    input_path = tmp_path / "input.csv"
    write_csv(toy_df, input_path)
    output_path = tmp_path / "out.csv"
    inferencer.run_inference(str(input_path), str(config_path), str(output_path))
    df_out = pd.read_csv(output_path)
    assert "prediction" in df_out.columns

def test_engineered_features_missing_exit(tmp_path, toy_df, toy_model, toy_pipeline):
    """If none of the engineered features are present, triggers sys.exit."""
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(toy_pipeline, pp_path)
    write_pickle(toy_model, model_path)
    config = {
        "raw_features": ["f1", "f2"],
        "features": {"engineered": ["something_not_present"]},
        "artifacts": {
            "preprocessing_pipeline": str(pp_path),
            "model_path": str(model_path),
        }
    }
    config_path = tmp_path / "config.yaml"
    write_yaml(config, config_path)
    input_path = tmp_path / "input.csv"
    write_csv(toy_df, input_path)
    output_path = tmp_path / "out.csv"
    with pytest.raises(SystemExit):
        inferencer.run_inference(str(input_path), str(config_path), str(output_path))

def test_run_inference_logs(monkeypatch, tmp_path, toy_config, toy_df, toy_model, toy_pipeline, caplog):
    """Logger emits info-level logs for all steps."""
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    write_pickle(toy_pipeline, pp_path)
    write_pickle(toy_model, model_path)
    config = dict(toy_config)
    config["artifacts"]["preprocessing_pipeline"] = str(pp_path)
    config["artifacts"]["model_path"] = str(model_path)
    config_path = tmp_path / "config.yaml"
    write_yaml(config, config_path)
    input_path = tmp_path / "input.csv"
    write_csv(toy_df, input_path)
    output_path = tmp_path / "out.csv"
    with caplog.at_level("INFO"):
        inferencer.run_inference(str(input_path), str(config_path), str(output_path))
        found = any("Loading preprocessing pipeline" in m for m in caplog.messages)
        assert found
        assert any("Generating predictions" in m for m in caplog.messages)

def test_inference_handles_model_without_predict_proba(tmp_path, toy_config, toy_df):
    """Inference does not fail if model lacks predict_proba."""
    # Use model without predict_proba
    from sklearn.svm import LinearSVC
    X = toy_df[["f1", "f2"]].values
    y = np.array([0, 1, 0, 1])
    model = LinearSVC()
    model.fit(X, y)
    pp_path = tmp_path / "preproc.pkl"
    model_path = tmp_path / "model.pkl"
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import FunctionTransformer
    pipeline = Pipeline([('identity', FunctionTransformer())])
    write_pickle(pipeline, pp_path)
    write_pickle(model, model_path)
    config = dict(toy_config)
    config["artifacts"]["preprocessing_pipeline"] = str(pp_path)
    config["artifacts"]["model_path"] = str(model_path)
    config_path = tmp_path / "config.yaml"
    write_yaml(config, config_path)
    input_path = tmp_path / "input.csv"
    write_csv(toy_df, input_path)
    output_path = tmp_path / "out.csv"
    inferencer.run_inference(str(input_path), str(config_path), str(output_path))
    df_out = pd.read_csv(output_path)
    assert "prediction" in df_out.columns
    assert "prediction_proba" not in df_out.columns
