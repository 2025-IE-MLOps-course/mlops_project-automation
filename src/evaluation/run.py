"""
evaluation/run.py

MLflow-compatible evaluation step with Hydra config and W&B logging.
Loads the trained model and processed validation/test data, computes metrics,
logs them, and saves schema, data hash, and plots for experiment tracking.
"""

import sys
import logging
import hashlib
import json
from datetime import datetime
from pathlib import Path

import hydra
import wandb
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.evaluator import generate_report, load_eval_data, plot_confusion_matrix, plot_roc_curve

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluation")

def compute_df_hash(df: pd.DataFrame) -> str:
    """Compute a hash for the DataFrame, including index."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for the evaluation MLflow step."""

    config_path = PROJECT_ROOT / "config.yaml"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"evaluation_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="evaluation",
            name=run_name,
            config=cfg_dict,
            tags=["evaluation"],
        )
        logger.info("Started WandB run: %s", run_name)

        # Load evaluation data for schema/hash logging (define load_eval_data as needed)
        df = load_eval_data(cfg_dict)
        if df is not None and not df.empty:
            df_hash = compute_df_hash(df)
            wandb.summary["eval_data_hash"] = df_hash
            eval_schema = {col: str(dtype) for col, dtype in df.dtypes.items()}
            wandb.summary["eval_data_schema"] = eval_schema
            schema_path = PROJECT_ROOT / "artifacts" / f"eval_schema_{run.id[:8]}.json"
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            with open(schema_path, "w") as f:
                json.dump(eval_schema, f, indent=2)
            schema_art = wandb.Artifact(f"eval_schema_{run.id[:8]}", type="schema")
            schema_art.add_file(str(schema_path))
            wandb.log_artifact(schema_art)
            # Optionally log a sample of evaluation data
            if cfg.data_load.get("log_sample_artifacts", True):
                sample_tbl = wandb.Table(dataframe=df.head(50))
                wandb.log({"eval_sample_rows": sample_tbl})

        # Generate report and log metrics
        report, y_true, y_pred, y_proba = generate_report(cfg_dict)  # update to return these as needed

        metrics_flat: dict[str, float] = {}
        for split, metrics in report.items():
            for key, val in metrics.items():
                if isinstance(val, dict):
                    for sub_k, sub_v in val.items():
                        metrics_flat[f"{split}_{key}_{sub_k}"] = sub_v
                else:
                    metrics_flat[f"{split}_{key}"] = val
        if metrics_flat:
            wandb.summary.update(metrics_flat)

        # Log confusion matrix and ROC if classification
        if y_true is not None and y_pred is not None:
            cm_fig = plot_confusion_matrix(y_true, y_pred)
            wandb.log({"confusion_matrix": wandb.Image(cm_fig)})
        if y_true is not None and y_proba is not None:
            roc_fig = plot_roc_curve(y_true, y_proba)
            wandb.log({"roc_curve": wandb.Image(roc_fig)})

        if cfg.data_load.get("log_artifacts", True):
            metrics_path = PROJECT_ROOT / cfg.artifacts.get("metrics_path", "models/metrics.json")
            if metrics_path.is_file():
                art = wandb.Artifact(f"metrics_{run.id[:8]}", type="metrics")
                art.add_file(str(metrics_path))
                wandb.log_artifact(art)
                logger.info("Logged metrics artifact to WandB")
    except Exception as e:
        logger.exception("Failed during evaluation step")
        if run is not None:
            run.alert(title="Evaluation Step Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")

if __name__ == "__main__":
    main()
