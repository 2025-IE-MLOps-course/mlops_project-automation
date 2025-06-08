"""inference/run.py

MLflow-compatible batch inference step with Hydra config and W&B logging.
Uses the trained model and preprocessing pipeline to generate predictions
for new data.
"""

from __future__ import annotations

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

from inference.inferencer import run_inference

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("inference")


@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Entry point for the inference MLflow step."""
    config_path = PROJECT_ROOT / "config.yaml"
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"inference_{dt_str}"

    run = None
    try:
        run = wandb.init(
            project=cfg.main.WANDB_PROJECT,
            entity=cfg.main.WANDB_ENTITY,
            job_type="inference",
            name=run_name,
            config=cfg_dict,
            tags=["inference"],
        )
        logger.info("Started WandB run: %s", run_name)

        input_path = PROJECT_ROOT / cfg.inference.input_csv
        output_path = PROJECT_ROOT / cfg.inference.output_csv

        run_inference(str(input_path), str(config_path), str(output_path))

        if cfg.data_load.get("log_artifacts", True) and output_path.is_file():
            artifact = wandb.Artifact(f"predictions_{run.id[:8]}", type="predictions")
            artifact.add_file(str(output_path))
            wandb.log_artifact(artifact)
            logger.info("Logged predictions artifact to WandB")

        if cfg.data_load.get("log_sample_artifacts", True) and output_path.is_file():
            import pandas as pd

            sample_tbl = wandb.Table(dataframe=pd.read_csv(output_path).head(50))
            wandb.log({"prediction_sample": sample_tbl})

        if output_path.is_file():
            import pandas as pd

            out_df = pd.read_csv(output_path)
            wandb.summary.update(
                {
                    "n_predictions": len(out_df),
                    "prediction_columns": list(out_df.columns),
                }
            )

    except Exception as e:
        logger.exception("Failed during inference step")
        if run is not None:
            run.alert(title="Inference Step Error", text=str(e))
        sys.exit(1)
    finally:
        if wandb.run is not None:
            wandb.finish()
            logger.info("WandB run finished")


if __name__ == "__main__":
    main()
