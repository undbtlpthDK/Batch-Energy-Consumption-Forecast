import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path.cwd()
ARTIFACTS = ROOT / "artifacts"


def make_run_dir(
    category: str,
    model_name: str,
) -> Path:
    """Creates a Run directory for model artifacts storage with
    add datetime parameters
    Parameters
    ----------
    category : str
        model category name
    model_name : str
        model name

    Returns
    -------
    Path
       Path to created directory
    """
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = ARTIFACTS / category / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_artifacts(
    run_dir: Path,
    metrics: dict,
    per_customer_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
):
    """Stores all the model related artifacts

    Parameters
    ----------
    run_dir : Path
        Path to the model run directory
    metrics : dict
        evaluation results
    per_customer_df : pd.DataFrame
        DataFrame with per customer evaluation
    forecasts_df : pd.DataFrame
        DataFrame with all forecasted data
    config : dict
        Model configuration
    """
    for k, v in metrics.items():
        metrics[k] = str(v)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    per_customer_df.to_parquet(
        run_dir / "per_customer.parquet",
        index=False,
    )

    forecasts_df.to_parquet(
        run_dir / "forecasts.parquet",
        index=False,
    )


def conf_to_params(conf) -> dict:
    """Convert dataclass config to dictionary

    Parameters
    ----------
    conf : _type_
        conf dataclass object of model configuration

    Returns
    -------
    dict
        MLflow acceptable configuration
    """
    d = asdict(conf)
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                out[f"{k}.{kk}"] = vv
        elif isinstance(v, list):
            out[k] = ",".join(map(str, v))
        else:
            out[k] = v
    return out
