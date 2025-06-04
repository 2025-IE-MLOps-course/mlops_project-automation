"""
data_loader.py

Modular data ingestion utility for CSV and Excel files.
- Loads configuration from config.yaml
- Loads secrets from .env (using python-dotenv)
- Robust error handling and logging (configured by main.py)
- Production-ready and MLOps teaching example
"""

import os
import logging
import pandas as pd
import yaml
from dotenv import load_dotenv
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    """
    Loads configuration from the given YAML file.
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    logger.info(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_env(env_path: str = ".env"):
    """
    Loads environment variables from .env file.
    """
    load_dotenv(dotenv_path=env_path, override=True)
    logger.info(f"Loaded environment from: {env_path}")


def load_data(
    path: str,
    file_type: str = "csv",
    sheet_name: Optional[str] = None,
    delimiter: str = ",",
    header: int = 0,
    encoding: str = "utf-8"
) -> pd.DataFrame:
    """
    Loads data from CSV or Excel, with error handling and logging.
    """
    if not path or not isinstance(path, str):
        logger.error("No valid data path specified in configuration.")
        raise ValueError("No valid data path specified in configuration.")
    if not os.path.isfile(path):
        logger.error(f"Data file does not exist: {path}")
        raise FileNotFoundError(f"Data file not found: {path}")
    try:
        if file_type == "csv":
            df = pd.read_csv(path, delimiter=delimiter,
                             header=header, encoding=encoding)
        elif file_type == "excel":
            df = pd.read_excel(path, sheet_name=sheet_name,
                               header=header, engine="openpyxl")
            if isinstance(df, dict):
                raise ValueError(
                    "Multiple sheets detected in Excel file. Please specify a single 'sheet_name' in the configuration."
                )
        else:
            logger.error(f"Unsupported file type: {file_type}")
            raise ValueError(f"Unsupported file type: {file_type}")
        logger.info(f"Loaded data from {path} ({file_type}), shape={df.shape}")
        return df
    except Exception as e:
        logger.exception(f"Failed to load data: {e}")
        raise


def get_data(
    config_path: str = "config.yaml",
    env_path: str = ".env",
    data_stage: str = "raw"
) -> pd.DataFrame:
    """
    Main entry: Loads env, config, resolves path, loads data for the requested stage.
    """
    load_env(env_path)
    config = load_config(config_path)
    data_cfg = config.get("data_source", {})
    if data_stage == "raw":
        path = data_cfg.get("raw_path")
    elif data_stage == "processed":
        path = data_cfg.get("processed_path")
    else:
        logger.error(f"Unknown data_stage: {data_stage}")
        raise ValueError(f"Unknown data_stage: {data_stage}")
    if not path or not isinstance(path, str):
        logger.error(
            "No valid data path specified in configuration for data_stage='%s'.", data_stage)
        raise ValueError(
            f"No valid data path specified in configuration for data_stage='{data_stage}'.")

    base_dir = Path(config_path).resolve().parent
    resolved_path = (
        base_dir / path).resolve() if not Path(path).is_absolute() else Path(path)

    # Only pass sheet_name if Excel
    if data_cfg.get("type", "csv") == "excel":
        df = load_data(
            path=str(resolved_path),
            file_type=data_cfg.get("type", "csv"),
            sheet_name=data_cfg.get("sheet_name"),
            delimiter=data_cfg.get("delimiter", ","),
            header=data_cfg.get("header", 0),
            encoding=data_cfg.get("encoding", "utf-8"),
        )
    else:
        df = load_data(
            path=str(resolved_path),
            file_type=data_cfg.get("type", "csv"),
            delimiter=data_cfg.get("delimiter", ","),
            header=data_cfg.get("header", 0),
            encoding=data_cfg.get("encoding", "utf-8"),
        )
    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    try:
        df = get_data(data_stage="raw")
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
    except Exception as e:
        logging.exception(f"Failed to load data: {e}")
