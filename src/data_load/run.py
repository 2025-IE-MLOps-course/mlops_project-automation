"""
data_load/run.py

MLflow-compatible, modular data loading step with Hydra config, W&B artifact logging, and robust error handling.
"""
import os
import sys
import logging
import hydra
import pandas as pd
import wandb
from omegaconf import DictConfig
from datetime import datetime
from data_loader import get_data


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("data_load")


@hydra.main(config_path=".", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Loads data for the MLOps pipeline with config and artifact tracking.
    Logs summary stats and sample artifacts to Weights & Biases.
    """

    # Get parameters from Hydra config (allowing CLI/MLflow overrides)
    output_dir = cfg.get("output_dir", "artifacts")
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = str(cfg.data_source.get("raw_path", "unknown")).split("/")[-1]
    run_name = f"data_load_{dt_str}_{data_file}"

    # Start WandB run with reproducibility tags
    run = wandb.init(
        project=cfg.main.WANDB_PROJECT,
        entity=cfg.main.WANDB_ENTITY,
        job_type="data_load",
        name=run_name,
        config=dict(cfg),
        tags=["data_load", data_file]
    )
    logger.info("Started WandB run: %s", run_name)

    # Ensure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "data_loaded.csv")
    sample_path = os.path.join(output_dir, "data_sample.csv")

    try:
        # Load data with environment variable fallback
        df = get_data(
            config_path=getattr(cfg, "config_path", "config.yaml"),
            env_path=getattr(cfg, "env_path", ".env"),
            data_stage=getattr(cfg, "data_stage", "raw"),
        )
        if df.empty:
            logger.warning("Loaded dataframe is empty: %s", output_path)
        if df.duplicated().any():
            logger.warning(
                "Duplicates found in data. Consider removing them before use.")

        df.to_csv(output_path, index=False)
        logger.info("Data saved to %s", output_path)

        # Log data metrics
        wandb.log({
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict()
        })
        # Log basic stats per column
        desc = df.describe(include="all").to_dict()
        for col, col_stats in desc.items():
            for stat, val in col_stats.items():
                wandb.log({f"stat/{col}/{stat}": val})

        # Save and log sample
        df.head(100).to_csv(sample_path, index=False)
        artifact = wandb.Artifact(
            "sample_data", type="dataset", description="First 100 rows of raw data")
        artifact.add_file(sample_path)
        wandb.log_artifact(artifact)
        logger.info("Sample artifact logged to WandB")

    except Exception as e:
        logger.exception("Failed during data loading step")
        run.alert(title="Data Load Error", text=str(e))
        sys.exit(1)

    finally:
        wandb.finish()
        logger.info("WandB run finished")


if __name__ == "__main__":
    main()
