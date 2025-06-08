"""
Comprehensive pytest suite for data_validator.py

- Demonstrates both "happy path" and failure/edge cases for robust educational coverage.
- All tests check the report artifact for validation output, not just exception strings.
- Each fixture and test is commented to support student learning.

How to run:
    pytest tests/test_data_validation.py
"""

import os
import pytest
import pandas as pd
import json
import copy
from data_validation import data_validator

# Basic schema definition for test coverage
BASIC_SCHEMA = [
    {"name": "ID", "dtype": "int", "required": True, "min": 1},
    {"name": "OD", "dtype": "int", "required": True, "allowed_values": [0, 1]},
    {"name": "rx ds", "dtype": "int", "required": True, "min": 0, "max": 2000},
]

# Minimal working config for testing
BASIC_CONFIG = {
    "data_validation": {
        "enabled": True,
        "action_on_error": "raise",
        "report_path": "logs/test_validation_report.json",
        "schema": {"columns": BASIC_SCHEMA}
    }
}

# valid_df: DataFrame matching the schema, for "happy path" validation


@pytest.fixture
def valid_df():
    """DataFrame with valid data for all schema columns (happy path)."""
    return pd.DataFrame({
        "ID": [1, 2, 3],
        "OD": [0, 1, 1],
        "rx ds": [10, 150, 400]
    })

# df_missing_col: DataFrame missing a required column


@pytest.fixture
def df_missing_col():
    """DataFrame missing a required column ("OD")."""
    return pd.DataFrame({
        "ID": [1, 2, 3],
        "rx ds": [10, 150, 400]
    })

# df_invalid_type: DataFrame where "ID" column is string, not int


@pytest.fixture
def df_invalid_type():
    """DataFrame with wrong dtype for 'ID' (should be int, is str)."""
    return pd.DataFrame({
        "ID": ["A", "B", "C"],
        "OD": [0, 1, 1],
        "rx ds": [10, 150, 400]
    })

# df_out_of_range: DataFrame with value(s) outside schema bounds


@pytest.fixture
def df_out_of_range():
    """DataFrame with values out of allowed range/set for 'OD' and 'rx ds'."""
    return pd.DataFrame({
        "ID": [1, 2, 3],
        "OD": [0, 2, 1],
        "rx ds": [10, 150, 4000]
    })

# Happy path: all validation passes, report marks "pass"


def test_validate_data_happy_path(valid_df, tmp_path):
    """Test successful validation (happy path, all data correct)."""
    config = copy.deepcopy(BASIC_CONFIG)
    report_path = tmp_path / "validation_report.json"
    config["data_validation"]["report_path"] = str(report_path)
    data_validator.validate_data(valid_df, config)
    assert os.path.exists(report_path)
    with open(report_path) as f:
        report = json.load(f)
        assert report["result"] == "pass"
        assert not report["errors"]

# Fails for missing required column


def test_validate_data_missing_required_column(df_missing_col, tmp_path):
    """Test missing required column triggers error and is reported."""
    config = copy.deepcopy(BASIC_CONFIG)
    config["data_validation"]["report_path"] = str(
        tmp_path / "validation_report.json")
    with pytest.raises(ValueError) as e:
        data_validator.validate_data(df_missing_col, config)
    assert "Data validation failed with errors" in str(e.value)
    with open(config["data_validation"]["report_path"]) as f:
        report = json.load(f)
    errors = report["errors"]
    assert any("Missing required column" in err for err in errors)

# Fails for dtype error (str instead of int)


def test_validate_data_invalid_dtype(df_invalid_type, tmp_path):
    """Test dtype mismatch triggers error and stops further checks."""
    config = copy.deepcopy(BASIC_CONFIG)
    config["data_validation"]["report_path"] = str(
        tmp_path / "validation_report.json")
    with pytest.raises(ValueError) as e:
        data_validator.validate_data(df_invalid_type, config)
    with open(config["data_validation"]["report_path"]) as f:
        report = json.load(f)
    errors = report["errors"]
    assert any("has dtype" in err for err in errors)

# Fails for out-of-range and out-of-set values


def test_validate_data_out_of_range(df_out_of_range, tmp_path):
    """Test out-of-range and not-in-set values trigger errors."""
    config = copy.deepcopy(BASIC_CONFIG)
    config["data_validation"]["report_path"] = str(
        tmp_path / "validation_report.json")
    with pytest.raises(ValueError) as e:
        data_validator.validate_data(df_out_of_range, config)
    with open(config["data_validation"]["report_path"]) as f:
        report = json.load(f)
    errors = report["errors"]
    assert any("above max" in err for err in errors)
    assert any("not in allowed set" in err for err in errors)

# Validation is skipped if disabled in config


def test_validate_data_disabled(valid_df, tmp_path):
    """Test validation is skipped when disabled in config."""
    config = copy.deepcopy(BASIC_CONFIG)
    config["data_validation"]["enabled"] = False
    config["data_validation"]["report_path"] = str(
        tmp_path / "validation_report.json")
    data_validator.validate_data(valid_df, config)
    assert not os.path.exists(config["data_validation"]["report_path"])

# Warn mode: errors do not raise, but report file records failure


def test_validate_data_action_on_error_warn(df_out_of_range, tmp_path, caplog):
    """Test action_on_error=warn records errors, but does not raise."""
    config = copy.deepcopy(BASIC_CONFIG)
    config["data_validation"]["action_on_error"] = "warn"
    config["data_validation"]["report_path"] = str(
        tmp_path / "validation_report.json")
    data_validator.validate_data(df_out_of_range, config)
    with open(config["data_validation"]["report_path"]) as f:
        report = json.load(f)
    assert report["result"] == "fail"
    assert len(report["errors"]) > 0

# Skips validation if no schema provided


def test_validate_data_missing_schema_key(valid_df, tmp_path, caplog):
    """Test validation is skipped if no schema is provided in config."""
    config = {"data_validation": {"enabled": True}}
    data_validator.validate_data(valid_df, config)
    # No exception, so test passes

# Passes if optional column is missing


def test_validate_data_optional_column(valid_df, tmp_path):
    """Test missing optional column does not cause validation failure."""
    schema = [
        {"name": "ID", "dtype": "int", "required": True, "min": 1},
        {"name": "foo", "dtype": "int", "required": False}
    ]
    config = copy.deepcopy(BASIC_CONFIG)
    config["data_validation"]["schema"]["columns"] = schema
    config["data_validation"]["report_path"] = str(
        tmp_path / "validation_report.json")
    data_validator.validate_data(valid_df, config)
    with open(config["data_validation"]["report_path"]) as f:
        report = json.load(f)
    assert report["result"] == "pass"
    assert "foo" in report["details"]
    assert report["details"]["foo"]["status"].startswith("not present")
