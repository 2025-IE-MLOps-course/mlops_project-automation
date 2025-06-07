"""
data_validation/run.py

MLflow-compatible, modular data validation step with Hydra config, W&B logging, and robust error handling.
"""

import sys
import logging
import hydra
import wandb
from omegaconf import DictConfig
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
from data_loader import get_data
from data_validator import validate_data
import json
import yaml

load_dotenv() 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("data_validation")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    config_path = PROJECT_ROOT / "config.yaml"

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"data_validation_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="data_validation",
            name=run_name,
            config=dict(cfg),
            tags=["data_validation"]
        )
        logger.info("Started WandB run: %s", run_name)

        # Determine what data to validate (raw by default)
        data_stage = getattr(cfg.data_load, "data_stage", "raw")
        df = get_data(
            config_path=str(config_path),
            data_stage=data_stage
        )
        if df.empty:
            logger.warning("Loaded dataframe is empty.")

        # Load config dict for validation
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)

        validate_data(df, config=config_dict)

        # Always log validation report to W&B (even if validation fails)
        val_report_path = cfg.data_validation.get(
            "report_path", "logs/validation_report.json")
        val_report_full_path = PROJECT_ROOT / val_report_path
        if val_report_full_path.is_file():
            artifact = wandb.Artifact("validation_report", type="report")
            artifact.add_file(str(val_report_full_path))
            wandb.log_artifact(artifact)
            logger.info("Logged validation report to WandB")
            with open(val_report_full_path) as f:
                report = json.load(f)
            wandb.summary.update({
                "validation_result": report.get("result", "unknown"),
                "validation_errors": len(report.get("errors", [])),
                "validation_warnings": len(report.get("warnings", [])),
            })
        else:
            logger.warning("Validation report not found for logging.")

    except Exception as e:
        logger.exception("Failed during data validation step")
        if run is not None:
            run.alert(title="Data Validation Error", text=str(e))
        # Always attempt to log the artifact if it exists
        val_report_path = cfg.data_validation.get(
            "report_path", "logs/validation_report.json")
        val_report_full_path = PROJECT_ROOT / val_report_path
        if val_report_full_path.is_file() and wandb.run is not None:
            artifact = wandb.Artifact("validation_report", type="report")
            artifact.add_file(str(val_report_full_path))
            wandb.log_artifact(artifact)
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
