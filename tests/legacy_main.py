import importlib
import sys
from typing import Dict
import pandas as pd
import pytest

# Import the target module only once so monkeypatching is consistent across tests
main_mod = importlib.import_module("main")


# Helper: run `main.main()` with a temporary sys.argv
def _run_main(argv):
    """Invoke *src.main.main()* as if called from the command‑line."""
    original_argv = sys.argv.copy()
    sys.argv = ["python -m src.main"] + argv
    try:
        main_mod.main()
    finally:
        sys.argv = original_argv


# Fixtures / reusable monkey‑patch helpers
@pytest.fixture(autouse=True)
def _patch_logging(monkeypatch):
    """Disable real logging setup (fast & silent test runs)."""
    monkeypatch.setattr(main_mod, "_setup_logging", lambda cfg: None)


@pytest.fixture
def dummy_df():
    """A 1×1 DataFrame (has a `.shape` like real data)."""
    return pd.DataFrame({"x": [1]})


@pytest.fixture
def mock_config(tmp_path):
    """Create an empty YAML file – its contents are irrelevant once _load_config is patched."""
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("logging: {}\n")
    return cfg_path


# Test cases
def test_stage_data_runs_data_load_and_validation(monkeypatch, mock_config, dummy_df):
    """`--stage data` must call **get_data** *and* **validate_data**, nothing else."""
    called: Dict[str, bool] = {}

    monkeypatch.setattr(main_mod, "_load_config", lambda path: {"logging": {}})
    monkeypatch.setattr(main_mod, "get_data", lambda **
                        kwargs: called.update(data=True) or dummy_df)
    monkeypatch.setattr(main_mod, "validate_data", lambda df,
                        cfg: called.update(validate=True))
    monkeypatch.setattr(main_mod, "run_model_pipeline",
                        lambda *a, **kw: called.update(model=True))
    monkeypatch.setattr(main_mod, "run_inference", lambda *a,
                        **kw: called.update(infer=True))

    _run_main(["--config", str(mock_config), "--stage", "data"])

    assert called == {"data": True, "validate": True}


def test_stage_train_runs_all_pipeline(monkeypatch, mock_config, dummy_df):
    """`--stage train` must call data‑load, validation and **run_model_pipeline**."""
    called = {}
    monkeypatch.setattr(main_mod, "_load_config", lambda path: {"logging": {}})
    monkeypatch.setattr(main_mod, "get_data", lambda **
                        kw: called.update(data=True) or dummy_df)
    monkeypatch.setattr(main_mod, "validate_data", lambda df,
                        cfg: called.update(validate=True))
    monkeypatch.setattr(main_mod, "run_model_pipeline",
                        lambda df, cfg: called.update(model=True))

    _run_main(["--config", str(mock_config), "--stage", "train"])

    assert called == {"data": True, "validate": True, "model": True}


def test_stage_infer_requires_input_output(monkeypatch, mock_config):
    """`--stage infer` *without* mandatory paths should exit with code 1."""
    monkeypatch.setattr(main_mod, "_load_config", lambda path: {"logging": {}})

    with pytest.raises(SystemExit):
        _run_main(["--config", str(mock_config), "--stage", "infer"])


def test_stage_all_runs_all_steps(monkeypatch, mock_config, dummy_df):
    """`--stage all` must execute data‑load, validation **and** training."""
    called = {}
    monkeypatch.setattr(main_mod, "_load_config", lambda path: {"logging": {}})
    monkeypatch.setattr(main_mod, "get_data", lambda **
                        kw: called.update(data=True) or dummy_df)
    monkeypatch.setattr(main_mod, "validate_data", lambda df,
                        cfg: called.update(validate=True))
    monkeypatch.setattr(main_mod, "run_model_pipeline",
                        lambda df, cfg: called.update(model=True))

    _run_main(["--config", str(mock_config), "--stage", "all"])

    assert called == {"data": True, "validate": True, "model": True}


def test_file_not_found_config(monkeypatch, tmp_path):
    """If `_load_config` raises *FileNotFoundError*, the CLI should exit(1)."""
    monkeypatch.setattr(
        main_mod,
        "_load_config",
        lambda path: (_ for _ in ()).throw(
            FileNotFoundError("Config file not found")),
    )

    with pytest.raises(SystemExit):
        _run_main(
            ["--config", str(tmp_path / "nonexistent.yaml"), "--stage", "data"])


def test_pipeline_exception_logging(monkeypatch, mock_config, dummy_df):
    """Any unhandled downstream exception bubbles up → logged & exits(1)."""
    monkeypatch.setattr(main_mod, "_load_config", lambda path: {"logging": {}})

    # Force validate_data to crash
    def _boom(*a, **kw):
        raise RuntimeError("validation failed")

    monkeypatch.setattr(main_mod, "get_data", lambda **kw: dummy_df)
    monkeypatch.setattr(main_mod, "validate_data", _boom)

    with pytest.raises(SystemExit):
        _run_main(["--config", str(mock_config), "--stage", "data"])
