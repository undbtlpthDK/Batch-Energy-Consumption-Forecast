from typing import List

import pandas as pd


def user_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates the mean and std baseline for the last 90 days

    Parameters
    ----------
    df : pd.DataFrame
        Smart Meters reading DataFrame where
        energy_import_kwh and energy_export_kwh columns should exists
    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with user consumption baseline
    """
    df_norm = df.sort_values(["object_id", "timestamp"]).copy()
    df_norm = _calculate_rolling_mean(df_norm, 24 * 90, mode="normalization")
    df_norm = _calculate_rolling_std(df_norm, 24 * 90, mode="normalization")
    return df_norm


def convert_to_multi_horizon(
    df: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    df = add_horizon_index(df, window)
    df = freeze_history_features(df, window)
    return df


def add_horizon_index(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    df = df.sort_values(["object_id", "timestamp"]).copy()

    df[f"horizon_index_{horizon}"] = (
        df.groupby("object_id").cumcount() % horizon
    )
    return df.reset_index(drop=True)


def freeze_history_features(
    df: pd.DataFrame, window: int, extra_freeze_cols: List[str]
) -> pd.DataFrame:
    df = df.sort_values(["object_id", "timestamp"]).copy()

    freeze_cols = [
        c
        for c in df.columns
        if c.startswith(
            (
                "import_lag",
                "export_lag",
                "import_rolling",
                "export_rolling",
            )
        )
    ]

    def _freeze_block(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        for start in range(0, len(g), window):
            end = min(start + window, len(g))
            base_row = g.iloc[start]

            # freeze lag / rolling features
            g.loc[g.index[start:end], freeze_cols] = base_row[
                freeze_cols
            ].values

            # add forecast_origin_time
            g.loc[g.index[start:end], "forecast_origin_time"] = base_row[
                "timestamp"
            ] - pd.Timedelta(hours=1)

        return g

    return (
        df.groupby("object_id", group_keys=False)
        .apply(_freeze_block)
        .reset_index(drop=True)
    )


def add_rolling_stats(
    df: pd.DataFrame, windows: List[int], stats: List["str"]
) -> pd.DataFrame:
    """Adds the rolling aggregated statistics
    columns added:
        import_rolling_sum_n, import_export_sum_n per each window
        import_rolling_mean_n, import_export_mean_n per each window
        import_rolling_std_n, import_export_std_n per each window
        where n is the size of the rolling value
    Parameters
    ----------
    df : pd.DataFrame
        Smart Meters reading DataFrame where
        energy_import_kwh and energy_export_kwh columns should exists
    windows : List[int]
        rolling window sizes
    stats : List[str]
        rolling statistics to compute (e.g. ["mean", "sum"])

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with energy rolling statistics
    """
    df_rolled = df.sort_values(["object_id", "timestamp"]).copy()

    for window in windows:
        if "sum" in stats:
            df_rolled = _calculate_rolling_sum(
                df_rolled, window, mode="rolling"
            )
        if "mean" in stats:
            df_rolled = _calculate_rolling_mean(
                df_rolled, window, mode="rolling"
            )
        if "std" in stats:
            df_rolled = _calculate_rolling_std(
                df_rolled, window, mode="rolling"
            )

    return df_rolled


def _calculate_rolling_stat(
    df: pd.DataFrame,
    window: int,
    mode: str,
    stat: str,
) -> pd.DataFrame:
    agg = getattr(
        df.groupby("object_id")[["energy_import_kwh", "energy_export_kwh"]]
        .shift(1)
        .rolling(window=window),
        stat,
    )

    values = agg().reset_index(level=0, drop=True)

    if mode == "rolling":
        cols = [
            f"import_rolling_{stat}_{window}",
            f"export_rolling_{stat}_{window}",
        ]
    elif mode == "normalization":
        days = window // 24
        cols = [
            f"import_{stat}_for_{days}_days",
            f"export_{stat}_for_{days}_days",
        ]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    df[cols] = values
    return df


def _calculate_rolling_sum(df, window, mode):
    return _calculate_rolling_stat(df, window, mode, "sum")


def _calculate_rolling_mean(df, window, mode):
    return _calculate_rolling_stat(df, window, mode, "mean")


def _calculate_rolling_std(df, window, mode):
    return _calculate_rolling_stat(df, window, mode, "std")


def add_lagged_energy_values(
    df: pd.DataFrame, lags: List[int]
) -> pd.DataFrame:
    """Adds the lagged energy consumption and generation features to DataFrame
    columns added:
        import_lag__n, export_lag_n per each lag
        where n is the size of the lag horizon
    Parameters
    ----------
    df : pd.DataFrame
        Smart Meters reading DataFrame where
        energy_import_kwh and energy_export_kwh columns should exists
    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with energy lags
    """

    df_lagged = df.sort_values(["object_id", "timestamp"]).copy()

    for lag in lags:
        shifted = df_lagged.groupby("object_id")[
            ["energy_import_kwh", "energy_export_kwh"]
        ].shift(lag)

        df_lagged[[f"import_lag_{lag}", f"export_lag_{lag}"]] = shifted

    return df_lagged
