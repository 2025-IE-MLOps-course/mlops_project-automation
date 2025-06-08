import os
import pytest
from evaluation.evaluator import generate_report


def test_generate_report_missing_model(tmp_path):
    cfg = {
        "target": "target",
        "artifacts": {
            "model_path": str(tmp_path / "missing_model.pkl"),
            "processed_dir": str(tmp_path / "proc"),
            "metrics_path": str(tmp_path / "metrics.json"),
        },
    }
    with pytest.raises(FileNotFoundError):
        generate_report(cfg)
