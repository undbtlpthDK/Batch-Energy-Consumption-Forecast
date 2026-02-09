import pandas as pd

from energycast.evaluation import metrics


def seasonal_naive_forecast(
    df_train: pd.DataFrame,
    df_dev: pd.DataFrame,
    *,
    target_col: str,
    time_col: str = "timestamp",
    group_col: str = "object_id",
    season: int = 24,
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
    season: int = 24,
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
        season horizon length, by default 24

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
