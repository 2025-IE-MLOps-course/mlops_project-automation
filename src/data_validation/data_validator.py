"""
Reusable, config-driven, production-quality data validation module for MLOps pipelines.

- Validates data against a schema defined in the config.yaml file
- Checks column presence, type, missing values, value ranges, and allowed sets
- Behavior (raise or warn) is configurable; results are logged and written as JSON
"""

import logging
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# _is_dtype_compatible


def _is_dtype_compatible(series, expected_dtype: str) -> bool:
    """
    Checks whether a pandas Series is compatible with the expected data type.

    Args:
        series: pandas.Series object to check
        expected_dtype: Data type as a string ('int', 'float', 'str', 'bool')

    Returns:
        bool: True if compatible, False otherwise

    Explanation:
        This function uses pandas dtype 'kind' codes to map Python types to
        pandas data types (e.g. 'i' for integer, 'f' for float, etc).
    """
    kind = series.dtype.kind
    if expected_dtype == "int":
        return kind in ("i", "u")
    elif expected_dtype == "float":
        return kind == "f"
    elif expected_dtype == "str":
        return kind in ("O", "U", "S")
    elif expected_dtype == "bool":
        return kind == "b"
    return False

# _validate_column


def _validate_column(
    df: pd.DataFrame,
    col_schema: Dict[str, Any],
    errors: List[str],
    warnings: List[str],
    report: Dict[str, Any]
) -> None:
    """
    Validates a single column of the DataFrame according to schema rules.

    Args:
        df: pandas DataFrame to validate
        col_schema: Schema dictionary for this column
        errors: List to collect error messages
        warnings: List to collect warning messages
        report: Dict for storing per-column validation details

    Educational Notes:
        - Checks for column presence, type, missing values, min/max bounds,
          and allowed values (as configured).
        - Stops further checks if dtype mismatch is detected.
        - Populates a report dictionary for later inspection or audit.
    """
    col = col_schema["name"]
    col_report = {}

    # Check if the column is present in the DataFrame
    if col not in df.columns:
        if col_schema.get("required", True):
            msg = f"Missing required column: {col}"
            errors.append(msg)
            col_report["status"] = "missing"
            col_report["error"] = msg
        else:
            col_report["status"] = "not present (optional)"
        report[col] = col_report
        return

    col_series = df[col]
    col_report["status"] = "present"

    # Type check: Stop further validation if dtype does not match
    dtype_expected = col_schema.get("dtype")
    if dtype_expected and not _is_dtype_compatible(col_series, dtype_expected):
        msg = f"Column '{col}' has dtype '{col_series.dtype}', expected '{dtype_expected}'"
        errors.append(msg)
        col_report["dtype"] = str(col_series.dtype)
        col_report["dtype_expected"] = dtype_expected
        col_report["error"] = msg
        report[col] = col_report
        return  # Don't run further checks to avoid pandas TypeError

    # Missing values check
    missing_count = col_series.isnull().sum()
    if missing_count > 0:
        if col_schema.get("required", True):
            msg = f"Column '{col}' has {missing_count} missing values (required)"
            errors.append(msg)
        else:
            msg = f"Column '{col}' has {missing_count} missing values (optional)"
            warnings.append(msg)
        col_report["missing_count"] = int(missing_count)

    # Minimum value check (if configured)
    if "min" in col_schema:
        min_val = col_schema["min"]
        below = (col_series < min_val).sum()
        if below > 0:
            msg = f"Column '{col}' has {below} values below min ({min_val})"
            errors.append(msg)
            col_report["below_min"] = int(below)

    # Maximum value check (if configured)
    if "max" in col_schema:
        max_val = col_schema["max"]
        above = (col_series > max_val).sum()
        if above > 0:
            msg = f"Column '{col}' has {above} values above max ({max_val})"
            errors.append(msg)
            col_report["above_max"] = int(above)

    # Allowed values check (if configured)
    if "allowed_values" in col_schema:
        allowed = set(col_schema["allowed_values"])
        invalid = ~col_series.isin(allowed)
        n_invalid = invalid.sum()
        if n_invalid > 0:
            msg = f"Column '{col}' has {n_invalid} values not in allowed set {allowed}"
            errors.append(msg)
            col_report["invalid_values_count"] = int(n_invalid)

    # Sample values (for report)
    try:
        col_report["sample_values"] = col_series.dropna().unique()[:5].tolist()
    except Exception:
        col_report["sample_values"] = "unavailable"

    report[col] = col_report

# validate_data


def validate_data(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> None:
    """
    Run full data validation as per config.yaml rules.

    Args:
        df: DataFrame to validate
        config: Full config dict (expects config['data_validation'])

    Behavior:
        - Loads schema and error behavior from config
        - Calls _validate_column for each column in schema
        - Writes a validation report as a JSON artifact
        - Raises or warns depending on configuration

    Educational Notes:
        This function is the main entry point for data validation in MLOps pipelines.
        It enforces data contracts at the earliest possible stage, which is a critical
        production and compliance best practice.
    """
    dv_cfg = config.get("data_validation", {})
    if not dv_cfg.get("enabled", True):
        dv_cfg["enabled"] = True
        logger.info("Data validation is disabled in config.")
        return

    schema = dv_cfg.get("schema", {}).get("columns", [])
    if not schema:
        logger.warning(
            "No data_validation.schema.columns defined in config. Skipping validation.")
        return

    action_on_error = dv_cfg.get("action_on_error", "raise").lower()
    report_path = dv_cfg.get("report_path", "logs/validation_report.json")
    errors, warnings = [], []
    report = {}

    # Validate each column according to the schema
    for col_schema in schema:
        _validate_column(df, col_schema, errors, warnings, report)

    # Write validation report as a JSON artifact before any error is raised or warning is logged
    report_path = Path(report_path)
    if report_path.parent != Path():
        report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w") as f:
        json.dump(
            {
                "result": "fail" if errors else "pass",
                "errors": errors,
                "warnings": warnings,
                "details": report,
            },
            f,
            indent=2,
        )

    # Logging for transparency and debugging
    if errors:
        logger.error(
            f"Data validation failed with {len(errors)} errors. See {report_path}")
        for e in errors:
            logger.error(e)
    if warnings:
        logger.warning(f"Data validation warnings: {len(warnings)}")
        for w in warnings:
            logger.warning(w)

    # Behavior: Strict in production, more relaxed in research
    if errors:
        if action_on_error == "raise":
            # Report has already been written
            raise ValueError(
                f"Data validation failed with errors. See {report_path} for details")
        elif action_on_error == "warn":
            logger.warning(
                "Data validation errors detected but proceeding as per config.")
        else:
            logger.warning(
                f"Unknown action_on_error '{action_on_error}'. Proceeding but data may be invalid.")
    else:
        logger.info(f"Data validation passed. Details saved to {report_path}")


# CLI support for running as a script
if __name__ == "__main__":
    """
    Optional CLI entry point for validating a CSV with a YAML config from the command line.

    Usage:
        python -m src.data_validation.data_validator <data.csv> <config.yaml>
    """
    import sys
    import yaml
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    if len(sys.argv) < 3:
        logger.error(
            "Usage: python -m src.data_validation.data_validator <data.csv> <config.yaml>")
        sys.exit(1)
    data_path, config_path = sys.argv[1], sys.argv[2]
    df = pd.read_csv(data_path)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    validate_data(df, config)
