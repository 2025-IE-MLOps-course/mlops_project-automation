"""
preprocess/run.py

MLflow-compatible preprocessing step with Hydra config and W&B logging.
Builds the preprocessing pipeline defined in ``config.yaml`` and saves it
as a pickle artifact for downstream stages.
"""

import sys
import logging
import pickle
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

from preprocess.preprocessing import build_preprocessing_pipeline
from data_load.data_loader import get_data

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("preprocess")


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for the preprocessing MLflow step."""
    config_path = PROJECT_ROOT / "config.yaml"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"preprocess_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="preprocess",
            name=run_name,
            config=cfg_dict,
            tags=["preprocess"],
        )
        logger.info("Started WandB run: %s", run_name)

        df = get_data(config_path=str(config_path), data_stage="processed")
        if df.empty:
            logger.warning("Loaded dataframe is empty.")

        pipeline = build_preprocessing_pipeline(cfg_dict)
        pipeline.fit(df)

        pp_path = PROJECT_ROOT / cfg.artifacts.get(
            "preprocessing_pipeline", "models/preprocessing_pipeline.pkl"
        )
        pp_path.parent.mkdir(parents=True, exist_ok=True)
        with pp_path.open("wb") as f:
            pickle.dump(pipeline, f)
        logger.info("Saved preprocessing pipeline to %s", pp_path)

        if cfg.data_load.get("log_artifacts", True):
            artifact = wandb.Artifact(
                f"preprocessing_pipeline_{run.id[:8]}", type="pipeline"
            )
            artifact.add_file(str(pp_path))
            wandb.log_artifact(artifact)
            logger.info("Logged preprocessing pipeline artifact to WandB")

    except Exception as e:
        logger.exception("Failed during preprocessing step")
        if run is not None:
            run.alert(title="Preprocess Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
