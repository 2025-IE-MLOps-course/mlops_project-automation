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

load_dotenv()  # Only loads secrets from .env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("data_load")

# Hydra will look for config.yaml two levels up (project root)


@hydra.main(config_path="../../", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # critical for MLflow project step
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
        output_path = os.path.join(output_dir, "data_loaded.csv")
        sample_path = os.path.join(output_dir, "data_sample.csv")

        df = get_data(
            config_path=config_path,
            data_stage=data_stage
        )
        if df.empty:
            logger.warning("Loaded dataframe is empty: %s", output_path)
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            logger.warning(
                f"Duplicates found in data ({dup_count} rows). Consider removing them before use.")

        df.to_csv(output_path, index=False)
        logger.info("Data saved to %s", output_path)

        wandb.log({
            "n_rows": df.shape[0],
            "n_cols": df.shape[1],
            "columns": list(df.columns),
            "dtypes": df.dtypes.astype(str).to_dict()
        })
        desc = df.describe(include="all").to_dict()
        for col, col_stats in desc.items():
            for stat, val in col_stats.items():
                wandb.log({f"stat/{col}/{stat}": val})

        df.head(100).to_csv(sample_path, index=False)
        artifact = wandb.Artifact(
            "sample_data", type="dataset", description="First 100 rows of raw data")
        artifact.add_file(sample_path)
        wandb.log_artifact(artifact)
        logger.info("Sample artifact logged to WandB")

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
