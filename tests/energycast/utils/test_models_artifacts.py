import json

import pandas as pd
import pytest

from energycast.utils import model_artifacts


@pytest.fixture
def sample_dfs():
    per_customer = pd.DataFrame({"id": ["A", "B"], "MAE": [1.0, 2.0]})
    forecasts = pd.DataFrame({"ts": [1, 2], "value": [10.0, 12.0]})
    return per_customer, forecasts


def test_make_run_dir_creates_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(model_artifacts, "ARTIFACTS", tmp_path / "artifacts")

    run_dir = model_artifacts.make_run_dir(
        category="multi-horizon",
        model_name="lightgbm",
    )

    assert run_dir.exists()
    assert run_dir.parent == (
        tmp_path / "artifacts" / "multi-horizon" / "lightgbm"
    )
    assert run_dir.name.startswith("run_")  # timestamped folder


def test_save_run_artifacts(tmp_path, sample_dfs):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    per_customer_df, forecasts_df = sample_dfs
    metrics = {"MAE": 1.23, "RMSE": 2.34}

    model_artifacts.save_run_artifacts(
        run_dir=run_dir,
        metrics=metrics,
        per_customer_df=per_customer_df,
        forecasts_df=forecasts_df,
    )

    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "per_customer.parquet").exists()
    assert (run_dir / "forecasts.parquet").exists()
    assert not (run_dir / "config.yaml").exists()

    # Metrics content (strings)
    with open(run_dir / "metrics.json") as f:
        loaded_metrics = json.load(f)
    assert loaded_metrics["MAE"] == "1.23"
    assert loaded_metrics["RMSE"] == "2.34"

    # DataFrames readable
    pd.testing.assert_frame_equal(
        pd.read_parquet(run_dir / "per_customer.parquet"),
        per_customer_df,
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(run_dir / "forecasts.parquet"),
        forecasts_df,
    )
