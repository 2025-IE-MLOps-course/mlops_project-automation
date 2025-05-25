"""
evaluator.py

Binary classification evaluation utilities for MLOps pipelines.
- Computes configurable metrics for any estimator (scikit-learn API)
- Designed for modular use in model.py or as standalone analysis
- All metrics, output paths, and options are config-driven for reproducibility
- Returns results as dict and optionally saves artifact (JSON)
"""

import logging
import os
import json
import pickle
import pandas as pd
from typing import Dict, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

logger = logging.getLogger(__name__)


def evaluate_classification(
    model,
    X,
    y,
    config: Dict[str, Any],
    save_path: Optional[str] = None,
    split: Optional[str] = None
) -> Dict[str, float]:
    """
    Evaluate a binary classifier on given features and labels.
    Metrics and reporting are driven by config['metrics'].

    Args:
        model: Trained estimator (must implement predict/predict_proba)
        X: Features (array or DataFrame)
        y: Ground truth labels
        config: Full config dict, expects 'metrics' key
        save_path: Optional JSON file to save results (default: None)
        split: Optional label for reporting (e.g., "validation", "test")

    Returns:
        Dictionary of metric names and values
    """
    y_pred = model.predict(X)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
    else:
        y_prob = None

    # Calculate confusion matrix for specificity/NPV
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    results = {}
    for metric in config.get("metrics", []):
        m = metric.lower()
        if m == "accuracy":
            results["Accuracy"] = accuracy_score(y, y_pred)
        elif m in ["precision", "precision (ppv)", "positive predictive value (ppv)"]:
            results["Precision (PPV)"] = precision_score(
                y, y_pred, zero_division=0)
        elif m in ["recall", "sensitivity"]:
            results["Recall (Sensitivity)"] = recall_score(
                y, y_pred, zero_division=0)
        elif m == "specificity":
            results["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        elif m == "f1 score":
            results["F1 Score"] = f1_score(y, y_pred, zero_division=0)
        elif m == "negative predictive value (npv)":
            results["Negative Predictive Value (NPV)"] = tn / \
                (tn + fn) if (tn + fn) > 0 else 0.0
        elif m == "roc auc":
            results["ROC AUC"] = roc_auc_score(
                y, y_prob) if y_prob is not None else float("nan")
        # Add more custom metrics if needed

    # Optionally include the confusion matrix for teaching
    results["Confusion Matrix"] = {
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    # Optionally save to JSON
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Evaluation results saved to {save_path}")

    def round_metrics(metrics_dict, ndigits=2):
        rounded = {}
        for k, v in metrics_dict.items():
            if isinstance(v, dict):
                rounded[k] = {ik: (round(iv, ndigits) if isinstance(
                    iv, float) else iv) for ik, iv in v.items()}
            elif isinstance(v, float):
                rounded[k] = round(v, ndigits)
            else:
                rounded[k] = v
        return rounded

    rounded_results = round_metrics(results)

    # Log metrics
    split_label = f" [{split}]" if split else ""
    logger.info(f"Evaluation metrics{split_label}: {rounded_results}")

    return results


def generate_report(
    config: Dict[str, Any],
    model_path: Optional[str] = None,
    processed_dir: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Load artifacts, evaluate validation and test splits, save one JSON.
    """
    artifacts = config.get("artifacts", {})
    model_path = model_path or artifacts.get("model_path", "models/model.pkl")
    processed_dir = processed_dir or artifacts.get(
        "processed_dir", "data/processed")
    save_path = save_path or artifacts.get(
        "metrics_path", "models/metrics.json")
    target = config["target"]

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    valid_df = pd.read_csv(os.path.join(processed_dir, "valid_processed.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test_processed.csv"))

    valid_df = valid_df.dropna(subset=[target])
    test_df = test_df.dropna(subset=[target])

    Xv, yv = valid_df.drop(columns=[target]).values, valid_df[target].values
    Xt, yt = test_df.drop(columns=[target]).values, test_df[target].values

    rv = evaluate_classification(model, Xv, yv, config, split="validation")
    rt = evaluate_classification(model, Xt, yt, config, split="test")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as fh:
        json.dump({"validation": rv, "test": rt}, fh, indent=2)
    logger.info(f"Metrics report saved to {save_path}")
    return {"validation": rv, "test": rt}
