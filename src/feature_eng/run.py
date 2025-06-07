"""
feature_eng/run.py

MLflow-compatible feature engineering step with Hydra config, W&B logging,
and robust error handling.
"""

import sys
import logging
from datetime import datetime
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

# Ensure project modules are importable when executed via MLflow
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_load.data_loader import get_data
from features.feature_eng import FEATURE_TRANSFORMERS

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("feature_eng")


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config_path = PROJECT_ROOT / "config.yaml"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"feature_eng_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="feature_eng",
            name=run_name,
            config=dict(cfg),
            tags=["feature_eng"],
        )
        logger.info("Started WandB run: %s", run_name)

        df = get_data(config_path=str(config_path), data_stage="raw")
        if df.empty:
            logger.warning("Loaded dataframe is empty.")

        for feat in cfg.features.get("engineered", []):
            builder = FEATURE_TRANSFORMERS.get(feat)
            if builder is None:
                logger.debug("No transformer registered for %s; skipping", feat)
                continue
            transformer = builder(cfg_dict)
            df = transformer.transform(df)
            logger.info("Applied transformer: %s", feat)

        processed_path = PROJECT_ROOT / cfg.data_source.processed_path
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(processed_path, index=False)
        logger.info("Saved engineered data to %s", processed_path)

        if cfg.data_load.get("log_artifacts", True):
            artifact = wandb.Artifact("processed_data", type="dataset")
            artifact.add_file(str(processed_path))
            wandb.log_artifact(artifact)
            logger.info("Logged processed data artifact to WandB")

        wandb.summary.update({"n_rows": df.shape[0], "n_cols": df.shape[1]})

    except Exception as e:
        logger.exception("Failed during feature engineering step")
        if run is not None:
            run.alert(title="Feature Eng Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
