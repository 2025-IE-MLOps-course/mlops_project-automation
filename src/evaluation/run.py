"""
evaluation/run.py

MLflow-compatible evaluation step with Hydra & W&B logging.
Logs scalar metrics and artifacts for every run, and only logs
confusion-matrix / ROC / PR plots when the test split has ≥2
non-NaN samples and the probability vector is at least length 2.
"""

import sys, logging, hashlib, json
from datetime import datetime
from pathlib import Path

import hydra, wandb, pandas as pd
from omegaconf import DictConfig, OmegaConf
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluation.evaluator import generate_report
from data_load.data_loader import get_data

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("evaluation")

def df_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()

def _len_safe(x):
    try:
        return len(x)
    except Exception:
        return 0

@hydra.main(config_path=str(PROJECT_ROOT), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    run_name = f"evaluation_{datetime.now():%Y%m%d_%H%M%S}"

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
        # ─── Dataset traceability ───────────────────────────────
        df = get_data(config_path=str(PROJECT_ROOT / "config.yaml"), data_stage="processed")
        if not df.empty:
            wandb.summary["eval_data_hash"] = df_hash(df)
            schema = {c: str(t) for c, t in df.dtypes.items()}
            wandb.summary["eval_data_schema"] = schema
            schema_path = PROJECT_ROOT / "artifacts" / f"eval_schema_{run.id[:8]}.json"
            schema_path.parent.mkdir(parents=True, exist_ok=True)
            json.dump(schema, open(schema_path, "w"), indent=2)
            wandb.log_artifact(wandb.Artifact(f"eval_schema_{run.id[:8]}", type="schema")
                               .add_file(str(schema_path)))
            if cfg.data_load.get("log_sample_artifacts", True):
                wandb.log({"eval_sample_rows": wandb.Table(dataframe=df.head(50))})

        # ─── Metrics and arrays ─────────────────────────────────
        report, y_true, y_pred, y_proba = generate_report(cfg_dict)

        flat = {}
        for split, metrics in report.items():
            for k, v in metrics.items():
                if isinstance(v, dict):
                    for sk, sv in v.items():
                        flat[f"{split}_{k}_{sk}"] = sv
                else:
                    flat[f"{split}_{k}"] = v
        wandb.summary.update(flat)

        # ─── Plot only if we have ≥2 samples after NaN filter ───
        if y_true is not None and _len_safe(y_true) > 1:
            # Confusion matrix
            if y_pred is not None and _len_safe(y_pred) == _len_safe(y_true):
                class_names = getattr(cfg, "dataset", {}).get("class_names", None)
                cm_panel = wandb.plot.confusion_matrix(
                    y_true=y_true, preds=y_pred, class_names=class_names
                )
                wandb.log({"confusion_matrix": cm_panel})

            # ROC & PR curves
            if y_proba is not None and _len_safe(y_proba) == _len_safe(y_true):
                if _len_safe(y_proba) > 1:  # need >1 point
                    wandb.log({
                        "roc_curve": wandb.plot.roc_curve(y_true, y_proba),
                        "pr_curve":  wandb.plot.pr_curve(y_true, y_proba),
                    })

        # ─── Metrics JSON artifact ─────────────────────────────
        if cfg.data_load.get("log_artifacts", True):
            m_path = PROJECT_ROOT / cfg.artifacts.get("metrics_path", "models/metrics.json")
            if m_path.is_file():
                wandb.log_artifact(wandb.Artifact(f"metrics_{run.id[:8]}", type="metrics")
                                   .add_file(str(m_path)))

    except Exception as e:
        logger.exception("Evaluation step failed")
        if wandb.run is not None:
            run.alert(title="Evaluation Step Error", text=str(e))
        sys.exit(1)
    finally:
        wandb.finish()
        logger.info("WandB run finished")

if __name__ == "__main__":
    main()
