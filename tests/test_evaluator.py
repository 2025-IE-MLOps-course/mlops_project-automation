"""
pytest suite for evaluator.py

- Covers happy path, config-driven metrics, save/load, logs, edge cases
- Uses toy data and minimal configs
"""

import pytest
import numpy as np
import pandas as pd
import os
import json
from sklearn.linear_model import LogisticRegression
from evaluation.evaluator import evaluate_classification

@pytest.fixture
def toy_data():
    """Balanced, separable binary data."""
    X = np.array([[0], [1], [2], [3]])
    y = np.array([0, 0, 1, 1])
    return X, y

@pytest.fixture
def minimal_config():
    return {"metrics": ["accuracy", "precision", "recall", "f1 score", "roc auc", "specificity", "negative predictive value (npv)"]}

def test_evaluate_classification_happy_path(toy_data, minimal_config):
    """Returns expected metrics dict with all requested keys and valid values."""
    X, y = toy_data
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    results = evaluate_classification(model, X, y, minimal_config)
    for metric in ["Accuracy", "Precision (PPV)", "Recall (Sensitivity)", "F1 Score", "ROC AUC", "Specificity", "Negative Predictive Value (NPV)", "Confusion Matrix"]:
        assert metric in results
    # Confusion Matrix values make sense
    cm = results["Confusion Matrix"]
    assert sum(cm.values()) == len(y)

def test_evaluate_classification_subset_metrics(toy_data):
    """Supports subset of metrics."""
    X, y = toy_data
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    config = {"metrics": ["accuracy", "specificity"]}
    results = evaluate_classification(model, X, y, config)
    assert set(results.keys()) == {"Accuracy", "Specificity", "Confusion Matrix"}

def test_evaluate_classification_save_to_json(tmp_path, toy_data, minimal_config):
    """Saves results to JSON if save_path provided."""
    X, y = toy_data
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    save_path = tmp_path / "metrics.json"
    results = evaluate_classification(model, X, y, minimal_config, save_path=str(save_path))
    # File exists and loads as dict with same keys
    assert os.path.exists(save_path)
    with open(save_path) as f:
        loaded = json.load(f)
    for k in results:
        assert k in loaded

def test_evaluate_classification_logs_metrics(toy_data, minimal_config, caplog):
    """Logger emits evaluation metrics."""
    X, y = toy_data
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    with caplog.at_level("INFO"):
        evaluate_classification(model, X, y, minimal_config, split="validation")
        found = any("Evaluation metrics" in m for m in caplog.messages)
        assert found

def test_evaluate_classification_zero_division_case():
    """Handles zero division for precision/recall gracefully."""
    # All y_true are 0, model always predicts 0
    X = np.zeros((6, 1))
    y = np.zeros(6)
    class DummyAllZero:
        def predict(self, X): return np.zeros(X.shape[0])
        def predict_proba(self, X): return np.tile([1, 0], (X.shape[0], 1))
    model = DummyAllZero()
    config = {"metrics": ["precision", "recall", "specificity", "negative predictive value (npv)", "f1 score"]}
    results = evaluate_classification(model, X, y, config)
    assert results["Precision (PPV)"] == 0
    assert results["Recall (Sensitivity)"] == 0
    assert results["Specificity"] == 1
    assert results["Negative Predictive Value (NPV)"] == 1
    assert results["F1 Score"] == 0

def test_evaluate_classification_handles_no_predict_proba(toy_data):
    """ROC AUC is nan if model lacks predict_proba."""
    class DummyNoProba:
        def predict(self, X): return np.ones(len(X))
    X, y = toy_data
    model = DummyNoProba()
    config = {"metrics": ["roc auc"]}
    results = evaluate_classification(model, X, y, config)
    assert "ROC AUC" in results and np.isnan(results["ROC AUC"])

def test_evaluate_classification_with_dataframe_input(minimal_config):
    """Accepts pandas DataFrames as input."""
    X = pd.DataFrame({"a": [0, 1, 2, 3]})
    y = pd.Series([0, 0, 1, 1])
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    results = evaluate_classification(model, X, y, minimal_config)
    assert "Accuracy" in results

def test_evaluate_classification_confusion_matrix_shape(toy_data, minimal_config):
    """Confusion matrix always has tn, fp, fn, tp as keys and correct type."""
    X, y = toy_data
    model = LogisticRegression(solver='liblinear')
    model.fit(X, y)
    results = evaluate_classification(model, X, y, minimal_config)
    cm = results["Confusion Matrix"]
    assert set(cm) == {"tn", "fp", "fn", "tp"}
    for v in cm.values():
        assert isinstance(v, int)
