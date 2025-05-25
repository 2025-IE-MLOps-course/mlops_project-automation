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
    log: bool = False,            # NEW
) -> dict[str, float | dict]:
    """
    Compute the selected metrics for one data split.

    Parameters
    ----------
    model        : fitted estimator implementing *predict* (and optionally *predict_proba*).
    X            : 2-D feature matrix.
    y            : 1-D array-like target.
    config       : full project config (used when *metrics* is None).
    metrics      : list[str] or None.  If None, defaults to ``config['metrics']['report']``.
    split        : Optional split name to include in the log line.
    log          : If True, write one INFO line with the resulting dict.

    Returns
    -------
    dict[str, float | dict]  – metric names mapped to values.
    """

    # Resolve metric list -----------------------------------------------------
    if metrics is None:
        m_cfg = config.get("metrics", {})
        metrics = m_cfg.get("report", m_cfg) if isinstance(
            m_cfg, dict) else m_cfg

    # Predictions -------------------------------------------------------------
    y_pred = model.predict(X)
    tn, fp, fn, tp = confusion_matrix(y, y_pred, labels=[0, 1]).ravel()
    results: dict[str, float | dict] = {}

    for metric in metrics:
        if metric == "Accuracy":
            results["Accuracy"] = float(accuracy_score(y, y_pred))
        elif metric == "Precision (PPV)":
            results["Precision (PPV)"] = float(precision_score(
                y, y_pred, zero_division=0))
        elif metric == "Specificity":
            results["Specificity"] = float(tn / max(tn + fp, 1))
        elif metric == "F1 Score":
            results["F1 Score"] = float(f1_score(y, y_pred, zero_division=0))
        elif metric == "Negative Predictive Value (NPV)":
            results["Negative Predictive Value (NPV)"] = float(
                tn / max(tn + fn, 1))
        elif metric == "ROC AUC":
            if len(np.unique(y)) == 2:
                results["ROC AUC"] = float(roc_auc_score(
                    y, model.predict_proba(X)[:, 1]))
            else:
                results["ROC AUC"] = float("nan")
        elif metric == "Confusion Matrix":
            results["Confusion Matrix"] = {"tn": int(tn), "fp": int(fp),
                                           "fn": int(fn), "tp": int(tp)}

    if log and split:
        logger.info(
            "Evaluation metrics [%s]: %s",
            split,
            json.dumps(_round(results), indent=2),
        )
    return results


def generate_report(
    config: Dict[str, Any],
    *,
    model_path: Optional[str] = None,
    processed_dir: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Create and persist a JSON metrics report for validation and test splits.

    Workflow
    --------
    1. Load trained model and processed CSV files from ``config['artifacts']``.
    2. Compute metrics specified in ``config['metrics']['report']`` using
       :pyfunc:`evaluate_classification`.
    3. Write the combined report to *save_path* and return it.
    """
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

    # 1. Load artifacts
    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    valid_df = pd.read_csv(os.path.join(processed_dir, "valid_processed.csv"))
    test_df = pd.read_csv(os.path.join(processed_dir, "test_processed.csv"))

    # safeguard: remove rows without ground‑truth label
    valid_df = valid_df.dropna(subset=[target])
    test_df = test_df.dropna(subset=[target])

    X_val, y_val = valid_df.drop(
        columns=[target]).values, valid_df[target].values
    X_test, y_test = test_df.drop(
        columns=[target]).values, test_df[target].values

    # 2. Compute metrics
    report_metrics = config.get("metrics", {}).get("report", [])
    res_val = evaluate_classification(
        model, X_val, y_val, config, metrics=report_metrics, split="validation",
        log=False
    )
    res_test = evaluate_classification(
        model, X_test, y_test, config, metrics=report_metrics, split="test",
        log=False
    )

    # 3. Persist report
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump({"validation": res_val, "test": res_test}, fh, indent=2)

    logger.info("Metrics report saved to %s", save_path)
    return {"validation": res_val, "test": res_test}
