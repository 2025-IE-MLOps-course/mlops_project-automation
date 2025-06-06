import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

load_dotenv()  # Only for secrets

PIPELINE_STEPS = [
    "data_load",
    "data_validation",
    "feature_eng",
    "preprocessing",
    "train_model",
    "evaluate_model",
    # "inference"
]


@hydra.main(config_name="config", config_path=".", version_base=None)
def main(cfg: DictConfig):
    os.environ["WANDB_PROJECT"] = cfg.main.WANDB_PROJECT
    os.environ["WANDB_ENTITY"] = cfg.main.WANDB_ENTITY

    steps_raw = cfg.main.steps
    active_steps = [s.strip() for s in steps_raw.split(",") if s.strip()] \
        if steps_raw != "all" else PIPELINE_STEPS

    hydra_override = cfg.main.hydra_options if hasattr(
        cfg.main, "hydra_options") else ""

    with tempfile.TemporaryDirectory() as tmp_dir:
        for step in active_steps:
            step_dir = os.path.join(
                hydra.utils.get_original_cwd(), "src", step)

            # Only pass hydra_options if it is non-empty
            params = {}
            if hydra_override:
                params["hydra_options"] = hydra_override

            print(f"Running step: {step}")
            mlflow.run(step_dir, "main", parameters=params)


if __name__ == "__main__":
    main()
