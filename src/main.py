"""
main.py
========
Single entry-point that orchestrates **every** module in the project:

1. **Data stage**  
   * Loads raw data (src.data_load.data_loader)  
   * Runs schema & quality checks (src.data_validation.data_validator)

2. **Training stage**  
   * Splits data, builds preprocessing, trains model  
     (src.model.model.run_model_pipeline)  
   * Saves artefacts and evaluation metrics automatically

3. **Inference stage**  
   * Applies the *persisted* preprocessing pipeline + model to a new CSV  
     (src.inference.inferencer.run_inference)

Add new stages or artefacts by extending the `if/elif` block below; the
core modules are already fully decoupled.

Typical commands
----------------
Full rebuild (data + train):

    python -m src.main --config config.yaml --stage all

Only data validation:

    python -m src.main --stage data

Batch inference:

    python -m src.main --stage infer \
        --input_csv data/inference/new_data.csv \
        --output_csv data/inference/predictions.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Dict

import yaml

from src.data_load.data_loader import get_data
from src.data_validation.data_validator import validate_data
from src.model.model import run_model_pipeline
from src.inference.inferencer import run_inference

logger = logging.getLogger(__name__)


# helpers
def _setup_logging(cfg: Dict) -> None:
    """Configure root logger from config.yaml → logging section."""
    log_level = cfg.get("level", "INFO").upper()
    log_file = cfg.get("log_file", "logs/main.log")
    fmt = cfg.get("format", "%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    datefmt = cfg.get("datefmt", "%Y-%m-%d %H:%M:%S")

    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=fmt,
        datefmt=datefmt,
        filename=log_file,
        filemode="a",
    )
    # echo to console as well
    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(fmt, datefmt))
    console.setLevel(getattr(logging, log_level, logging.INFO))
    logging.getLogger().addHandler(console)


def _load_config(path: str) -> Dict:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# main routine
def main() -> None:
    parser = argparse.ArgumentParser(description="MLOps pipeline orchestrator")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    parser.add_argument(
        "--env",
        default=".env",
        help="Optional .env with credentials / environment vars",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "data", "train", "infer"],
        help="Pipeline stage to execute",
    )
    parser.add_argument("--input_csv", help="Raw CSV for inference stage")
    parser.add_argument("--output_csv", help="Output CSV for inference stage")
    args = parser.parse_args()

    # 1 – config & logging -------------------------------------------------
    try:
        cfg = _load_config(args.config)
    except Exception as exc:
        print(f"[main] Unable to read config: {exc}", file=sys.stderr)
        sys.exit(1)

    _setup_logging(cfg.get("logging", {}))
    logger.info("Pipeline started | stage=%s", args.stage)

    try:
        # 2 – data loading + validation -----------------------------------
        if args.stage in ("all", "data"):
            df_raw = get_data(config_path=args.config, env_path=args.env, data_stage="raw")
            logger.info("Raw data loaded | shape=%s", df_raw.shape)
            validate_data(df_raw, cfg)

        # 3 – training -----------------------------------------------------
        if args.stage in ("all", "train"):
            # reuse dataframe if already loaded in "all"
            if args.stage == "train":
                df_raw = get_data(config_path=args.config, env_path=args.env, data_stage="raw")
                validate_data(df_raw, cfg)
            run_model_pipeline(df_raw, cfg)

        # 4 – batch inference --------------------------------------------
        if args.stage == "infer":
            if not args.input_csv or not args.output_csv:
                logger.error("Inference stage requires --input_csv and --output_csv")
                sys.exit(1)
            run_inference(args.input_csv, args.config, args.output_csv)

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)

    logger.info("Pipeline completed successfully")


# CLI wrapper
if __name__ == "__main__":
    main()
