"""
model/run.py

MLflow-compatible model training step with Hydra config and W&B logging.
Trains the model using the config-defined preprocessing and model
settings, saves artifacts, and logs metrics to Weights & Biases.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_load.data_loader import get_data
from model.model import run_model_pipeline
from evaluation.evaluator import generate_report

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("model")


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config_path = PROJECT_ROOT / "config.yaml"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"model_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="model",
            name=run_name,
            config=cfg_dict,
            tags=["model"],
        )
        logger.info("Started WandB run: %s", run_name)

        df = get_data(config_path=str(config_path), data_stage="raw")
        if df.empty:
            logger.warning("Loaded dataframe is empty.")

        # Train model and save artifacts
        run_model_pipeline(df, cfg_dict)

        # Generate metrics report (validation/test)
        report = generate_report(cfg_dict)

        # Flatten metrics for summary logging
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

        # Log artifacts if configured
        if cfg.data_load.get("log_artifacts", True):
            model_path = PROJECT_ROOT / cfg.artifacts.get("model_path", "models/model.pkl")
            if model_path.is_file():
                art = wandb.Artifact(f"model_{run.id[:8]}", type="model")
                art.add_file(str(model_path))
                wandb.log_artifact(art)
                logger.info("Logged model artifact to WandB")

            pp_path = PROJECT_ROOT / cfg.artifacts.get(
                "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
            )
            if pp_path.is_file():
                art = wandb.Artifact(f"preprocessing_pipeline_{run.id[:8]}", type="pipeline")
                art.add_file(str(pp_path))
                wandb.log_artifact(art)
                logger.info("Logged preprocessing pipeline artifact to WandB")

            metrics_path = PROJECT_ROOT / cfg.artifacts.get("metrics_path", "models/metrics.json")
            if metrics_path.is_file():
                art = wandb.Artifact(f"metrics_{run.id[:8]}", type="metrics")
                art.add_file(str(metrics_path))
                wandb.log_artifact(art)
                logger.info("Logged metrics artifact to WandB")

    except Exception as e:
        logger.exception("Failed during model step")
        if run is not None:
            run.alert(title="Model Step Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
