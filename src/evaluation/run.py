"""evaluation/run.py

MLflow-compatible evaluation step with Hydra configuration and W&B logging.
Loads the trained model and processed validation/test data to compute metrics
and logs them as artifacts for experiment tracking.
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

from evaluation.evaluator import generate_report

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluation")


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

        try:
            report = generate_report(cfg_dict)
        except FileNotFoundError as e:
            logger.error("%s", e)
            if run is not None:
                run.alert(title="Evaluation Step Error", text=str(e))
            sys.exit(1)

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
