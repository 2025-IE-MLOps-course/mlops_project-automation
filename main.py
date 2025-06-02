"""
main.py
========
Central entry-point to orchestrate the entire modular MLOps pipeline using MLflow.

This script does not directly import or call pipeline modules. Instead, each pipeline stage is executed
as an isolated MLflow project by invoking `mlflow.run` and passing configuration and artifacts as parameters.
Each submodule (under src/) is responsible for its own logic, configuration, logging, and experiment tracking (e.g., via Wandb).

Pipeline stages
--------------
1. **Data loading**
   - Loads raw data using `src/data_load` (as an MLflow project)
2. **Data validation**
   - Validates schema and data quality using `src/data_validation`
3. **Preprocessing**
   - Builds and applies preprocessing pipelines using `src/preprocessing`
4. **Feature engineering**
   - Generates new features using `src/feature_eng`
5. **Training**
   - Splits data, fits models, and persists artifacts using `src/train_model`
6. **Evaluation**
   - Evaluates trained models and generates metrics via `src/evaluate_model`
7. **Inference**
   - Runs batch inference using the trained model and preprocessing pipeline via `src/inference`

Typical usage
-------------
Recommended (for full reproducibility, including remote runs):

Full pipeline run (all steps):
    mlflow run . 

Run only selected steps:
    mlflow run . -P steps="data_load,train_model"

Override configuration (with Hydra-style CLI overrides):
    mlflow run . -P steps="data_load,train_model" -P hydra_options="model.active=random_forest"

Advanced:
    # Pass any Hydra-compatible override through hydra_options (spaces must be escaped if needed)
    mlflow run . -P hydra_options="preprocessing.rename_columns.rx\\ ds=rx_ds"

Local Python execution (for dev/debug only):
    python main.py main.steps="data_load,train_model" model.active="random_forest"

Configuration is managed via Hydra and config.yaml. All metrics and artifacts are tracked in each step via Wandb.

How to extend
-------------
- To add a new pipeline step, add a new `if` block for the step below, following the established pattern.
- Ensure each step has its own MLproject, conda.yml, and main.py in its respective subdirectory under `src/`.
- All steps must accept a `config` parameter and (optionally) a temporary working directory.

Author: Ivan Diaz
"""


import json
import mlflow
import tempfile
import os
import hydra
from omegaconf import DictConfig
from dotenv import load_dotenv

load_dotenv()  # Loads .env file if present

# Define pipeline steps in order. Extend this list as new steps are added.
_steps = [
    "data_load",
    "data_validation",
    "preprocessing",
    "feature_eng",
    "train_model",
    "evaluate_model",
    "inference"
]


@hydra.main(config_name='config', config_path='.', version_base=None)
def main(config: DictConfig):
    """
    Orchestrates the modular MLflow pipeline.
    Each pipeline step is run as an isolated MLflow project.
    Configuration and workspace isolation are handled per step.
    All experiment tracking is handled inside each step via Wandb.
    """
    # Setup Wandb environment variables (ensure these are in config.yaml or .env)
    os.environ["WANDB_PROJECT"] = config["main"]["WANDB_PROJECT"]
    os.environ["WANDB_ENTITY"] = config["main"]["WANDB_ENTITY"]
    if os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]

    # Determine which steps to execute
    steps_par = config["main"]["steps"]
    active_steps = [s.strip() for s in steps_par.split(",")
                    ] if steps_par != "all" else _steps

    # Use a temporary directory for any intermediate artifacts if needed by modules
    with tempfile.TemporaryDirectory() as tmp_dir:
        if "data_load" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(),
                             "src", "data_load"),
                "main",
                parameters={
                    "config": os.path.abspath(hydra.utils.get_original_cwd() + "/config.yaml"),
                    "tmp_dir": tmp_dir
                }
            )

        if "data_validation" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(),
                             "src", "data_validation"),
                "main",
                parameters={
                    "config": os.path.abspath(hydra.utils.get_original_cwd() + "/config.yaml"),
                    "tmp_dir": tmp_dir
                }
            )

        if "preprocessing" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(),
                             "src", "preprocessing"),
                "main",
                parameters={
                    "config": os.path.abspath(hydra.utils.get_original_cwd() + "/config.yaml"),
                    "tmp_dir": tmp_dir
                }
            )

        if "feature_eng" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(),
                             "src", "feature_eng"),
                "main",
                parameters={
                    "config": os.path.abspath(hydra.utils.get_original_cwd() + "/config.yaml"),
                    "tmp_dir": tmp_dir
                }
            )

        if "train_model" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(),
                             "src", "train_model"),
                "main",
                parameters={
                    "config": os.path.abspath(hydra.utils.get_original_cwd() + "/config.yaml"),
                    "tmp_dir": tmp_dir
                }
            )

        if "evaluate_model" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(),
                             "src", "evaluate_model"),
                "main",
                parameters={
                    "config": os.path.abspath(hydra.utils.get_original_cwd() + "/config.yaml"),
                    "tmp_dir": tmp_dir
                }
            )

        if "inference" in active_steps:
            _ = mlflow.run(
                os.path.join(hydra.utils.get_original_cwd(),
                             "src", "inference"),
                "main",
                parameters={
                    "config": os.path.abspath(hydra.utils.get_original_cwd() + "/config.yaml"),
                    "tmp_dir": tmp_dir
                }
            )


if __name__ == "__main__":
    main()
