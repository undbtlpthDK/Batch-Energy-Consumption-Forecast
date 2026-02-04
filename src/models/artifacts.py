import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml


def make_run_dir(
    artifacts_root: Path,
    category: str,
    model_name: str,
) -> Path:
    """Creates a Run directory for model artifacts storage with
    add datetime parameters
    Parameters
    ----------
    artifacts_root : Path
        artifacts directory
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
    run_dir = artifacts_root / category / model_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run_artifacts(
    run_dir: Path,
    metrics: dict,
    per_customer_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    config: dict,
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

    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
