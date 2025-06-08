"""
evaluation/evaluator.py

Binary-classification evaluation utilities for your MLOps pipeline.

Key features
------------
* Computes configurable metrics for scikit-learn estimators
* Robust to NaNs in target columns
* Works with 1-D or 2-D predict_proba output
* Saves a JSON report (validation & test) and returns arrays for W&B plots
"""

from __future__ import annotations
import json, logging, pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np, pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ───────────────────────── helpers ──────────────────────────
def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p

def _specificity(tn: int, fp: int) -> float:
    return tn / (tn + fp) if (tn + fp) else float("nan")

def _npv(tn: int, fn: int) -> float:
    return tn / (tn + fn) if (tn + fn) else float("nan")


# ───────────────────── metric computation ───────────────────
def evaluate_classification(
    model,
    X: np.ndarray,
    y: np.ndarray,
    config: Dict[str, Any],
    *,
    metrics: Optional[list[str]] = None,
    split: str = "",
    save_path: Optional[str] = None,
    log: bool = False,
) -> dict[str, float | dict]:
    """Compute metrics for one data split (drops NaN targets)."""
    mask = ~pd.isna(y)
    if not mask.any():
        logger.warning("Split '%s' has only NaN targets—skipping metrics", split)
        return {}

    X, y = X[mask], y[mask]
    y_pred = model.predict(X)

    aliases = {
        "accuracy": "Accuracy",
        "precision": "Precision (PPV)",
        "recall": "Recall (Sensitivity)",
        "sensitivity": "Recall (Sensitivity)",
        "f1": "F1 Score",
        "f1 score": "F1 Score",
        "roc auc": "ROC AUC",
        "specificity": "Specificity",
        "npv": "Negative Predictive Value (NPV)",
        "negative predictive value": "Negative Predictive Value (NPV)",
        "negative predictive value (npv)": "Negative Predictive Value (NPV)",
        "confusion matrix": "Confusion Matrix",
    }
    if metrics is None:
        m_cfg = config.get("metrics", {})
        metrics = m_cfg.get("report", m_cfg) if isinstance(m_cfg, dict) else m_cfg
    metrics = [aliases.get(m.lower(), m) for m in metrics]
    if "Confusion Matrix" not in metrics:
        metrics.append("Confusion Matrix")

    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (cm.flatten().tolist() + [0, 0, 0, 0])[:4]

    out: dict[str, float | dict] = {}
    for m in metrics:
        if m == "Accuracy":
            out[m] = float(accuracy_score(y, y_pred))
        elif m == "Precision (PPV)":
            out[m] = float(precision_score(y, y_pred, zero_division=0))
        elif m == "Recall (Sensitivity)":
            out[m] = float(recall_score(y, y_pred, zero_division=0))
        elif m == "Specificity":
            out[m] = _specificity(tn, fp)
        elif m == "Negative Predictive Value (NPV)":
            out[m] = _npv(tn, fn)
        elif m == "F1 Score":
            out[m] = float(f1_score(y, y_pred, zero_division=0))
        elif m == "ROC AUC":
            if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
                proba = model.predict_proba(X)
                proba_pos = proba[:, 1] if proba.ndim == 2 else proba
                out[m] = float(roc_auc_score(y, proba_pos))
            else:
                out[m] = float("nan")
        elif m == "Confusion Matrix":
            out[m] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}

    if save_path:
        sp = Path(save_path)
        sp.parent.mkdir(parents=True, exist_ok=True)
        with open(sp, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)

    if log:
        msg_split = f" ({split})" if split else ""
        logger.info("Evaluation metrics%s: %s", msg_split, json.dumps(out))

    return out


# ────────────────── report generation ───────────────────────
def generate_report(
    config: Dict[str, Any],
    *,
    model_path: Optional[str] = None,
    processed_dir: Optional[str] = None,
    save_path: Optional[str] = None,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    np.ndarray | None, np.ndarray | None, np.ndarray | None
]:
    """
    Compute metrics for validation & test, save JSON, return arrays for W&B.

    Returns
    -------
    (report_dict, y_test, y_pred_test, y_proba_test)
    Arrays are None if the test split is empty or fully NaN.
    """
    for k in ("target", "artifacts"):
        if k not in config:
            logger.error("Config missing '%s'; abort report", k)
            return {"validation": {}, "test": {}}, None, None, None

    artifacts = config["artifacts"]
    model_path = _resolve(model_path or artifacts.get("model_path", "models/model.pkl"))
    processed_dir = _resolve(processed_dir or artifacts.get("processed_dir", "data/processed"))
    save_path = _resolve(save_path or artifacts.get("metrics_path", "models/metrics.json"))
    target = config["target"]

    with open(model_path, "rb") as fh:
        model = pickle.load(fh)

    valid_df = pd.read_csv(processed_dir / "valid_processed.csv", dtype={target: "float"})
    test_df  = pd.read_csv(processed_dir / "test_processed.csv",  dtype={target: "float"})

    report_metrics = config.get("metrics", {}).get("report", [])

    res_val = {}
    if not valid_df.empty:
        Xv, yv = valid_df.drop(columns=[target]).values, valid_df[target].values
        res_val = evaluate_classification(model, Xv, yv, config,
                                          metrics=report_metrics, split="validation")

    res_test, y_true, y_pred, y_proba = {}, None, None, None
    if not test_df.empty:
        Xt, yt = test_df.drop(columns=[target]).values, test_df[target].values
        res_test = evaluate_classification(model, Xt, yt, config,
                                           metrics=report_metrics, split="test")

        # Extract arrays for W&B plots (after dropping NaNs)
        mask = ~pd.isna(yt)
        if mask.sum() > 0:
            y_true = yt[mask]
            y_pred = model.predict(Xt[mask])
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(Xt[mask])
                y_proba = proba[:, 1] if proba.ndim == 2 else proba

    # Save JSON report
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as fh:
        json.dump({"validation": res_val, "test": res_test}, fh, indent=2)
    logger.info("Metrics report saved to %s", save_path)

    return {"validation": res_val, "test": res_test}, y_true, y_pred, y_proba
