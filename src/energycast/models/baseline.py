from pathlib import Path

import mlflow
import pandas as pd

from energycast.utils import model_artifacts
from src.energycast.evaluation import metrics

ROOT = Path.cwd()
ARTIFACTS = ROOT / "artifacts"
PROCESSED = ROOT / "data" / "processed"

config = {
    "model_type": "baseline",
    "baseline_type": "seasonal_naive",
    "season": 168,
    "horizon": 168,
    "target_col": "energy_kwh",
    "evaluation_protocol": "rolling_origin_shift",
}


def seasonal_naive_forecast(
    df_train: pd.DataFrame,
    df_dev: pd.DataFrame,
    *,
    target_col: str,
    time_col: str = "timestamp",
    group_col: str = "object_id",
    season: int = 168,
) -> pd.DataFrame:
    """Simple Naive Baseline implementation based on pandas shift.
    Default value for season is 1 week, as it's main scope  of
    project

    Parameters
    ----------
    df_train : pd.DataFrame
        history values data, only needed for the first iteration
    df_dev : pd.DataFrame
        dataframe to predict value on
    target_col : str
        name of target column
    time_col : str, optional
        timestamp column, by default "timestamp"
    group_col : str, optional
        identifier column name
    season : int, optional
        length of a season, by default 168

    Returns
    -------
    pd.DataFrame
       predicted and original values
    """
    df_all = (
        pd.concat([df_train, df_dev])
        .sort_values([group_col, time_col])
        .reset_index(drop=True)
    )

    df_all["y_pred"] = df_all.groupby(group_col)[target_col].shift(season)

    return (
        df_all.loc[df_all[time_col].isin(df_dev[time_col])]
        .dropna(subset=["y_pred"])
        .assign(y_true=lambda x: x[target_col])[
            [group_col, time_col, "y_true", "y_pred"]
        ]
    )


def run_seasonal_naive_baseline(
    df_train: pd.DataFrame,
    df_dev: pd.DataFrame,
    *,
    target_col: str,
    time_col: str = "timestamp",
    group_col: str = "object_id",
    season: int = 168,
):
    """_summary_

    Parameters
    ----------
    df_train : pd.DataFrame
        history values data, only needed for the first iteration
    df_dev : pd.DataFrame
        dev dataframe with targets
    target_col : str
        target column name
    time_col : str, optional
        timestamp column name, by default "timestamp"
    group_col : str, optional
        identifier column name, by default "object_id"
    season : int, optional
        season horizon length, by default 168

    Returns
    -------
    results
        overall model results
    per_customer_metrics
        accuracy metrics per id
    df_forecast: pd.DataFrame
        forecasted data
    """
    df_forecast = seasonal_naive_forecast(
        df_train,
        df_dev,
        target_col=target_col,
        time_col=time_col,
        group_col=group_col,
        season=season,
    )

    results, per_customer_metrics = metrics.evaluate_forecasts(
        df_forecast,
        pred_col="y_pred",
        actual_col="y_true",
        id_col=group_col,
    )

    return results, per_customer_metrics, df_forecast


def main():
    df_train_name = "stats_models_168_train.parquet"
    df_dev_name = "stats_models_168_dev.parquet"
    df_train = pd.read_parquet(PROCESSED / "stats_models_168_train.parquet")
    df_dev = pd.read_parquet(PROCESSED / "stats_models_168_dev.parquet")

    data_tracking = {
        "train_data": str(PROCESSED / df_train_name),
        "train_dvc": str(PROCESSED / f"{df_train_name}.dvc"),
        "dev_data": str(PROCESSED / df_dev_name),
        "dev_dvc": str(PROCESSED / f"{df_dev_name}.dvc"),
    }

    with mlflow.start_run(run_name="seasonal_naive"):
        mlflow.set_experiment("EnergyCast-Baselines")

        mlflow.log_params(
            {
                "train_data": data_tracking["train_data"],
                "dev_data": data_tracking["dev_data"],
            }
        )
        config["data_tracking"] = data_tracking

        results, per_customer, forecasts = run_seasonal_naive_baseline(
            df_train,
            df_dev,
            target_col="energy_kwh",
        )

        mlflow.log_metrics(results)

        run_dir = model_artifacts.make_run_dir(  # type: ignore
            artifacts_root=ARTIFACTS,
            category="baseline",
            model_name="seasonal_naive",
        )

        model_artifacts.save_run_artifacts(  # type: ignore
            run_dir=run_dir,
            metrics=results,
            per_customer_df=per_customer,
            forecasts_df=forecasts,
            config=config,
        )

        mlflow.log_artifacts(str(run_dir))


if __name__ == "__main__":
    main()
