"""
model.py

Leakage-proof, end-to-end MLOps pipeline:
- Splits raw data first
- Performs feature engineering and preprocessing inside this step
- Fits preprocessing pipeline ONLY on train split, applies to valid/test
- Trains model, evaluates, and saves model and preprocessing artifacts
"""

import os
import logging
import json
import pickle
from typing import Dict, Any
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from preprocess.preprocessing import build_preprocessing_pipeline, get_output_feature_names
from evaluation.evaluator import evaluate_classification


logger = logging.getLogger(__name__)

# Resolve project root two levels above this file so that artifact paths
# defined in config can be resolved relative to the repository.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve(path: str | Path) -> Path:
    """Return absolute path relative to PROJECT_ROOT if not already absolute."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p

MODEL_REGISTRY = {
    "decision_tree": DecisionTreeClassifier,
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
}


def train_model(X_train, y_train, model_type, params):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model type: {model_type}")
    model_cls = MODEL_REGISTRY[model_type]
    model = model_cls(**params)
    model.fit(X_train, y_train)
    logger.info(f"Trained model: {model_type}")
    return model


def save_artifact(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"Artifact saved to {path}")


def format_metrics(metrics: dict, ndigits: int = 2) -> dict:
    return {k: round(float(v), ndigits) if isinstance(v, (float, int)) else v for k, v in metrics.items()}


def run_model_pipeline(df: pd.DataFrame, config: Dict[str, Any]):
    """
    Train–validate–test workflow with strict train-only fitting for all
    preprocessing steps.

    Steps
    -----
    1. Split raw data into train / valid / test as configured.
    2. Fit preprocessing pipeline on train; transform all splits.
    3. Train the selected model type from *config['model']*.
    4. Persist artefacts (splits, pipeline, model).
    5. Log a configurable subset of metrics (config['metrics']['display']).

    The full metric set is **not** saved here; it is generated later by
    :pyfunc:`evaluation.evaluator.generate_report`.

    """
    # 1. Split data using only raw features (present in the original file)
    raw_features = config.get("raw_features", [])
    target = config["target"]
    split_cfg = config["data_split"]
    input_features_raw = [f for f in raw_features if f != target]

    X = df[input_features_raw]
    y = df[target]
    test_size = split_cfg.get("test_size", 0.2)
    valid_size = split_cfg.get("valid_size", 0.2)
    random_state = split_cfg.get("random_state", 42)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + valid_size), random_state=random_state, stratify=y
    )
    rel_valid = valid_size / (test_size + valid_size)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_valid, random_state=random_state, stratify=y_temp
    )
    # --- Save raw data splits ---
    splits_dir = _resolve(
        config.get("artifacts", {}).get("splits_dir", "data/splits")
    )
    splits_dir.mkdir(parents=True, exist_ok=True)
    X_train.assign(**{target: y_train}).to_csv(
        splits_dir / "train.csv", index=False
    )
    X_valid.assign(**{target: y_valid}).to_csv(
        splits_dir / "valid.csv", index=False
    )
    X_test.assign(**{target: y_test}).to_csv(
        splits_dir / "test.csv", index=False
    )

    # 2. Fit preprocessing pipeline on X_train, transform all splits
    preprocessor = build_preprocessing_pipeline(config)
    X_train_pp = preprocessor.fit_transform(X_train)
    X_valid_pp = preprocessor.transform(X_valid)
    X_test_pp = preprocessor.transform(X_test)

    # 3. Create DataFrames with engineered feature columns
    engineered_features = config.get("features", {}).get("engineered", [])
    out_cols = get_output_feature_names(
        preprocessor, input_features_raw, config)
    X_train_pp = pd.DataFrame(X_train_pp, columns=out_cols)
    X_valid_pp = pd.DataFrame(X_valid_pp, columns=out_cols)
    X_test_pp = pd.DataFrame(X_test_pp, columns=out_cols)

    # 4. Use only engineered features for modeling
    input_features = [
        f for f in engineered_features if f in X_train_pp.columns]
    X_train_pp = X_train_pp[input_features]
    X_valid_pp = X_valid_pp[input_features]
    X_test_pp = X_test_pp[input_features]

    # Save processed data splits
    processed_dir = _resolve(
        config.get("artifacts", {}).get("processed_dir", "data/processed")
    )
    processed_dir.mkdir(parents=True, exist_ok=True)
    X_train_pp.assign(**{target: y_train}).to_csv(
        processed_dir / "train_processed.csv", index=False
    )
    X_valid_pp.assign(**{target: y_valid}).to_csv(
        processed_dir / "valid_processed.csv", index=False
    )
    X_test_pp.assign(**{target: y_test}).to_csv(
        processed_dir / "test_processed.csv", index=False
    )

    # Save preprocessing pipeline artifact
    preproc_path = _resolve(
        config.get("artifacts", {}).get(
            "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
        )
    )
    save_artifact(preprocessor, str(preproc_path))

    # Train model
    model_config = config["model"]
    active = model_config.get("active", "decision_tree")
    active_model_cfg = model_config[active]
    model_type = active
    params = active_model_cfg.get("params", {})
    model = train_model(X_train_pp.values, y_train, model_type, params)

    # Save model artifact
    model_path = _resolve(
        config.get("artifacts", {}).get("model_path", "models/model.pkl")
    )
    save_artifact(model, str(model_path))

    active = model_config.get("active", "decision_tree")
    algo_model_path = _resolve(
        model_config.get(active, {}).get("save_path", f"models/{active}.pkl")
    )
    save_artifact(model, str(algo_model_path))

    # 5. Evaluate model and log metrics
    display_metrics = config.get("metrics", {}).get("display", [])

    results_valid = evaluate_classification(         # validation
        model, X_valid_pp.values, y_valid,
        config, metrics=display_metrics,
        split=None, log=False,
    )
    results_test = evaluate_classification(         # test
        model, X_test_pp.values, y_test,
        config, metrics=display_metrics,
        split=None, log=False,
    )

    # Round floats for nicer console reading
    def _round(d: dict[str, Any], ndigits: int = 2) -> dict[str, Any]:
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

    logger.info("Validation metrics: %s", json.dumps(
        _round(results_valid), indent=2))
    logger.info("Test metrics: %s",       json.dumps(
        _round(results_test),  indent=2))


# CLI for standalone training
if __name__ == "__main__":
    import sys
    import yaml
    import logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    try:
        from src.data_load.data_loader import get_data
        df = get_data(config_path=config_path, data_stage="raw")
    except ImportError:
        data_path = config["data_source"]["raw_path"]
        df = pd.read_csv(data_path)
    run_model_pipeline(df, config)
