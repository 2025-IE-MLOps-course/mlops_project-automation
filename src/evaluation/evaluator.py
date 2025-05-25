"""
Binary‑classification evaluation utilities for MLOps pipelines.

Key features
------------
* Computes a configurable set of metrics for any scikit‑learn‑compatible estimator
* Configuration‑driven: metric lists and artifact paths are taken from ``config.yaml``
* Can be called from the training script (e.g., *model.py*) or as a standalone
  analysis step via :pyfunc:`generate_report`
* Results are returned as dictionaries and, when desired, persisted as JSON

Usage
------
* Use :pyfunc:`evaluate_classification` to compute metrics for a fitted model
* Use :pyfunc:`generate_report` to create a JSON report for validation and test splits
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
    recall_score
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# internal helpers

def _specificity(tn: int, fp: int) -> float:
    """Specificity = TN / (TN + FP). Returns *nan* if denominator is zero."""
    denom = tn + fp
    return tn / denom if denom else float("nan")


def _npv(tn: int, fn: int) -> float:
    """Negative predictive value = TN / (TN + FN). Returns *nan* if denom=0."""
    denom = tn + fn
    return tn / denom if denom else float("nan")


def _round(d: dict[str, Any], ndigits: int = 2) -> dict[str, Any]:
    """Round floats in a (nested) metrics dict for nicer display."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {ik: (round(iv, ndigits) if isinstance(iv, float) else iv)
                      for ik, iv in v.items()}
        elif isinstance(v, float):
            out[k] = round(v, ndigits)
        else:
            out[k] = v
    return out


def evaluate_classification(
    model,
    X,
    y,
    config,
    *,
    metrics=None,
    split: str | None = None,
    log: bool = False,
    # New: Save directly to JSON if path provided
    save_path: Optional[str] = None,
) -> dict[str, float | dict]:
    """
    Compute the selected metrics for one data split.

    Parameters
    ----------
    model        : fitted estimator implementing *predict* (and optionally *predict_proba*).
    X            : 2-D feature matrix.
    y            : 1-D array-like target.
    config       : full project config (used when *metrics* is None).
    metrics      : list[str] or None.  If None, defaults to ``config['metrics']`` or config['metrics']['report'].
    split        : Optional split name to include in the log line.
    log          : If True, write one INFO line with the resulting dict.
    save_path    : Optional[str]. If provided, saves result dict as JSON.

    Returns
    -------
    dict[str, float | dict]  – metric names mapped to values.
    """

    # --- Standardize and map metrics ---
    aliases = {
        "accuracy": "Accuracy",
        "precision": "Precision (PPV)",
        "precision (ppv)": "Precision (PPV)",
        "recall": "Recall (Sensitivity)",
        "recall (sensitivity)": "Recall (Sensitivity)",
        "sensitivity": "Recall (Sensitivity)",
        "f1": "F1 Score",
        "f1 score": "F1 Score",
        "roc auc": "ROC AUC",
        "specificity": "Specificity",
        "negative predictive value": "Negative Predictive Value (NPV)",
        "negative predictive value (npv)": "Negative Predictive Value (NPV)",
        "npv": "Negative Predictive Value (NPV)",
        "confusion matrix": "Confusion Matrix",
    }

    if metrics is None:
        m_cfg = config.get("metrics", {})
        # Support both list and dict config
        metrics = m_cfg.get("report", m_cfg) if isinstance(
            m_cfg, dict) else m_cfg
    metrics = [aliases.get(m.lower(), m) for m in metrics]

    # Always add Confusion Matrix for completeness (your tests expect it)
    if "Confusion Matrix" not in metrics:
        metrics.append("Confusion Matrix")

    # --- Compute predictions ---
    y_pred = model.predict(X)
    # Robust: Confusion matrix needs to handle possible 1-class cases
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    # fallback for single-class cases in tiny toy data
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    elif cm.shape == (1, 1):
        tn = cm[0, 0]
        fp = fn = tp = 0
    else:
        # 2x1 or 1x2 cases
        if set(np.unique(y)) == {0}:
            tn = cm[0, 0]
            fp = cm[0, 1] if cm.shape[1] > 1 else 0
            fn = tp = 0
        else:
            tp = cm[0, 0]
            fn = cm[1, 0] if cm.shape[0] > 1 else 0
            tn = fp = 0

    results: dict[str, float | dict] = {}

    for metric in metrics:
        if metric == "Accuracy":
            results["Accuracy"] = float(accuracy_score(y, y_pred))
        elif metric == "Precision (PPV)":
            results["Precision (PPV)"] = float(precision_score(
                y, y_pred, zero_division=0))
        elif metric == "Recall (Sensitivity)":
            results["Recall (Sensitivity)"] = float(
                recall_score(y, y_pred, zero_division=0))
        elif metric == "Specificity":
            denom = tn + fp
            results["Specificity"] = tn / denom if denom else 1.0 if (
                tn + fp + fn + tp) > 0 and (tn + fn) > 0 else float("nan")
        elif metric == "Negative Predictive Value (NPV)":
            denom = tn + fn
            results["Negative Predictive Value (NPV)"] = tn / denom if denom else 1.0 if (
                tn + fp + fn + tp) > 0 and (tn + fp) > 0 else float("nan")
        elif metric == "F1 Score":
            results["F1 Score"] = float(f1_score(y, y_pred, zero_division=0))
        elif metric == "ROC AUC":
            try:
                if hasattr(model, "predict_proba"):
                    if len(np.unique(y)) == 2:
                        results["ROC AUC"] = float(
                            roc_auc_score(y, model.predict_proba(X)[:, 1]))
                    else:
                        results["ROC AUC"] = float("nan")
                else:
                    results["ROC AUC"] = float("nan")
            except Exception:
                results["ROC AUC"] = float("nan")
        elif metric == "Confusion Matrix":
            results["Confusion Matrix"] = {"tn": int(tn), "fp": int(fp),
                                           "fn": int(fn), "tp": int(tp)}

    # Logging if requested
    if log and split:
        logger.info(
            "Evaluation metrics [%s]: %s",
            split,
            json.dumps(_round(results), indent=2),
        )

    # Save to JSON if requested (test needs this)
    if save_path:
        with open(save_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    return results


def generate_report(
    config: Dict[str, Any],
    *,
    model_path: Optional[str] = None,
    processed_dir: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute and persist a JSON metrics report for validation and test splits.

    This function loads a trained model and preprocessed validation/test sets,
    evaluates metrics defined in the config, and writes results to a JSON file.
    For toy or minimal datasets, if a split has no samples, the corresponding
    metrics dict will be empty, ensuring robust operation even with tiny splits.

    Parameters
    ----------
    config : dict
        Project configuration with all required artifact paths and metric names.
    model_path : str or None
        Path to the trained model (.pkl). Falls back to config if not provided.
    processed_dir : str or None
        Directory with preprocessed validation/test CSVs. Defaults to config.
    save_path : str or None
        Where to save the output metrics report JSON. Defaults to config.

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Dict with "validation" and "test" keys, each mapping to their metrics dict.
        If a split is empty, its dict is empty.
    """
    required_keys = ["target", "artifacts"]
    for key in required_keys:
        if key not in config:
            logger.error(
                "generate_report: missing required config key '%s'. Skipping report.", key)
            return {"validation": {}, "test": {}}

    artifacts = config.get("artifacts", {})
    model_path = cast(
        str,
        model_path or artifacts.get("model_path", "models/model.pkl"),
    )
    processed_dir = cast(
        str,
        processed_dir or artifacts.get("processed_dir", "data/processed"),
    )
    save_path = cast(
        str,
        save_path or artifacts.get("metrics_path", "models/metrics.json"),
    )
    target: str = config["target"]

    # 1. Load trained model
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    # 2. Load and filter processed splits
    valid_df = pd.read_csv(os.path.join(
        processed_dir, "valid_processed.csv")).dropna(subset=[target])
    test_df = pd.read_csv(os.path.join(
        processed_dir, "test_processed.csv")).dropna(subset=[target])

    X_val, y_val = valid_df.drop(
        columns=[target]).values, valid_df[target].values
    X_test, y_test = test_df.drop(
        columns=[target]).values, test_df[target].values

    # 3. Compute metrics, skip if a split is empty (common in tiny tests)
    report_metrics = config.get("metrics", {}).get("report", [])
    res_val = {}
    if len(y_val) > 0:
        res_val = evaluate_classification(
            model, X_val, y_val, config, metrics=report_metrics, split="validation", log=False
        )
    res_test = {}
    if len(y_test) > 0:
        res_test = evaluate_classification(
            model, X_test, y_test, config, metrics=report_metrics, split="test", log=False
        )

    # 4. Persist report as JSON
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump({"validation": res_val, "test": res_test}, fh, indent=2)

    logger.info("Metrics report saved to %s", save_path)
    return {"validation": res_val, "test": res_test}
