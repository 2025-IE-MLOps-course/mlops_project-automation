"""
data_load/run.py

MLflow-compatible, modular data loading step with Hydra config, W&B artifact logging, and robust error handling.
"""

import os
import sys
import logging
import hydra
from hydra.utils import to_absolute_path
import pandas as pd
import wandb
from omegaconf import DictConfig
from datetime import datetime
from data_loader import get_data
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()  # Only loads secrets from .env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("data_load")


@hydra.main(config_path="../../", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config_path = to_absolute_path("../../config.yaml")
    output_dir = cfg.data_load.output_dir
    data_stage = cfg.data_load.data_stage

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = str(cfg.data_source.get("raw_path", "unknown")).split("/")[-1]
    run_name = f"data_load_{dt_str}_{data_file}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="data_load",
            name=run_name,
            config=dict(cfg),
            tags=["data_load", data_file]
        )
        logger.info("Started WandB run: %s", run_name)

        os.makedirs(output_dir, exist_ok=True)
        resolved_raw_path = Path(to_absolute_path(cfg.data_source.raw_path))

        # Load data
        df = get_data(
            config_path=config_path,
            data_stage=data_stage
        )
        if df.empty:
            logger.warning("Loaded dataframe is empty: %s", resolved_raw_path)
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            logger.warning(
                f"Duplicates found in data ({dup_count} rows). Consider removing them before use.")

        # --- W&B logging (honor config flags) ---
        # Log sample rows as wandb.Table
        if cfg.data_load.get("log_sample_artifacts", True):
            sample_tbl = wandb.Table(dataframe=df.head(100))
            wandb.log({"sample_rows": sample_tbl})

        # Log summary stats as wandb.Table
        if cfg.data_load.get("log_summary_stats", True):
            stats_tbl = wandb.Table(dataframe=df.describe(
                include="all").T.reset_index())
            wandb.log({"summary_stats": stats_tbl})

        # Log raw data as artifact (by reference, not duplicate upload)
        if cfg.data_load.get("log_artifacts", True):
            raw_art = wandb.Artifact(f"raw_data_{run.id[:8]}", type="dataset")
            # By reference: requires wandb>=0.16, otherwise fallback to add_file
            try:
                raw_art.add_reference(str(resolved_raw_path))
            except Exception:
                raw_art.add_file(str(resolved_raw_path))
            wandb.log_artifact(raw_art)
            logger.info("Logged raw data artifact to WandB")

        # Log simple metrics summary
        wandb.summary.update({
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "n_duplicates": dup_count,
            "columns": list(df.columns)
        })

    except Exception as e:
        logger.exception("Failed during data loading step")
        if run is not None:
            run.alert(title="Data Load Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
