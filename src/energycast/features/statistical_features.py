from typing import List

import pandas as pd


def convert_to_multi_horizon(
    df: pd.DataFrame, window: int, extra_freeze_cols: List[str]
) -> pd.DataFrame:
    """Add frozen columns and horizon index

    Parameters
    ----------
    df : pd.DataFrame
        smart readings dataframe
    window : int
        size of the frozen window

    Returns
    -------
    pd.DataFrame
        dataframe with added horizon index
        and all the log and roll columns freezed
    """

    df = add_horizon_index(df, window)
    df = freeze_history_features(df, window, [])
    return df


def add_horizon_index(
    df: pd.DataFrame,
    horizon: int,
) -> pd.DataFrame:
    """Adds horizon index that indicates the
    position relatively known data

    Parameters
    ----------
    df : pd.DataFrame
        smart readings dataframe
    horizon : int
        length of horizon cycle

    Returns
    -------
    pd.DataFrame
        dataframe with added horizon index
    """

    df = df.sort_values(["object_id", "timestamp"]).copy()

    df[f"horizon_index_{horizon}"] = (
        df.groupby("object_id").cumcount() % horizon
    )
    return df.reset_index(drop=True)


def freeze_history_features(
    df: pd.DataFrame, window: int, extra_freeze_cols: List[str]
) -> pd.DataFrame:
    """Add frozen lag and roll columns to the dataframe.
    That is used for multi-horizon forecast to prevent
    data leakage and utilize data in more realistic way

    Parameters
    ----------
    df : pd.DataFrame
        smart readings dataframe
    window : int
        length of frozen cycle
    extra_freeze_cols : List[str]
        name of columns to freeze in addition to the ones
        that starts with:
         - lag
         - roll

    Returns
    -------
    pd.DataFrame
        dataframe with frozen columns
    """
    df = (
        df.sort_values(["object_id", "timestamp"])
        .copy()
        .reset_index(drop=True)
    )

    freeze_cols = [
        c for c in df.columns if c.startswith(("lag", "roll"))
    ] + extra_freeze_cols

    frozen_cols = [f"{c}_frozen" for c in freeze_cols]
    for c, fc in zip(freeze_cols, frozen_cols):
        df[fc] = df[c]

    out_blocks = []

    for object_id, g in df.groupby("object_id", sort=False):
        g = g.copy()

        for start in range(0, len(g), window):
            end = min(start + window, len(g))
            base_row = g.iloc[start]

            # freeze values
            g.loc[g.index[start:end], frozen_cols] = base_row[
                freeze_cols
            ].values

            g.loc[g.index[start:end], "forecast_origin_time"] = base_row[
                "timestamp"
            ] - pd.Timedelta(hours=1)

        out_blocks.append(g)

    return (
        pd.concat(out_blocks, ignore_index=True)
        .sort_values(["object_id", "timestamp"])
        .reset_index(drop=True)
    )


def add_customer_averages(
    df: pd.DataFrame, shift_hours: int = 24, windows_days: list[int] = [30, 90]
) -> pd.DataFrame:
    """Calculates customer specific long horizon features:
    - dynamic history long averages
    - dynamic 30 and 90 days average
    Parameters
    ----------
    df : pd.DataFrame
        Smart Meters reading DataFrame where
        energy column should exists
    history_window_days : int, optional
        length of history window, by default 90
    shift_hours : int, optional
        data leakage prevention mechanism, by default 24
    days : list[int]
        ranges that we need to calculate dynamic means,  by default 30 and 90

    Returns
    -------
    pd.DataFrame
        _description_

    Raises
    ------
    ValueError
        raised if there no enerhy columns in df
    """
    df_out = (
        df.sort_values(["object_id", "timestamp"])
        .copy()
        .reset_index(drop=True)
    )

    energy_cols = [c for c in df_out.columns if c.startswith("energy")]
    if not energy_cols:
        raise ValueError("No columns starting with 'energy' found")

    shifted = df_out.groupby("object_id", sort=False)[energy_cols].shift(
        shift_hours
    )
    # History long mean
    dynamic_means = (
        shifted.groupby(df_out["object_id"], sort=False)
        .expanding()
        .mean()
        .reset_index(level=0, drop=True)
    )
    dynamic_means.columns = [f"customer_mean_{col}" for col in energy_cols]

    df_out[dynamic_means.columns] = dynamic_means

    # Defined period long mean
    for days in windows_days:
        window = days * 24

        rolling_means = (
            shifted.groupby(df_out["object_id"], sort=False)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

        rolling_means.columns = [
            f"customer_{days}d_mean_{col}" for col in energy_cols
        ]

        df_out[rolling_means.columns] = rolling_means

    return df_out


def add_rolling_stats(
    df: pd.DataFrame, windows: List[int], aggs: List["str"]
) -> pd.DataFrame:
    """Adds the rolling aggregated statistics
    columns added:
        energy_x_rolling_agg_n,
        where x are columns starting with "energy"
        where agg is aggregation function
        where n is the window size
    Parameters
    ----------
    df : pd.DataFrame
        Smart Meters reading DataFrame where
        energy column should exists
    windows : List[int]
        rolling window sizes
    aggs : List[str]
        aggregation functions to compute (e.g. ["mean", "sum"])

    Returns
    -------
    pd.DataFrame
        data with energy rolling statistics
    """
    df_rolled = (
        df.sort_values(["object_id", "timestamp"])
        .copy()
        .reset_index(drop=True)
    )

    energy_cols = [c for c in df_rolled.columns if c.startswith("energy")]
    if not energy_cols:
        raise ValueError(
            "No columns starting with 'energy' found in DataFrame"
        )

    for window in windows:
        for func in aggs:
            df_rolled = _calculate_rolling_stat(
                df=df_rolled,
                energy_cols=energy_cols,
                window=window,
                func=func,
            )

    return df_rolled


def _calculate_rolling_stat(
    df: pd.DataFrame,
    energy_cols: List[str],
    window: int,
    func: str,
) -> pd.DataFrame:

    grouped = (
        df.groupby("object_id", sort=False)[energy_cols]
        .shift(1)
        .rolling(window=window)
    )

    # Dynamic pic of agg function, converts grouped.func
    agg_fn = getattr(grouped, func)
    values = agg_fn().reset_index(level=0, drop=True)

    cols = [f"rolling_{func}_{col}_{window}" for col in energy_cols]

    values.columns = cols
    df[cols] = values

    return df


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
        energy column should exists
    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with energy lags
    """

    df_lagged = (
        df.sort_values(["object_id", "timestamp"])
        .copy()
        .reset_index(drop=True)
    )
    energy_cols = [
        col for col in df_lagged.columns if col.startswith("energy")
    ]

    for lag in lags:
        shifted = df_lagged.groupby("object_id")[energy_cols].shift(lag)

        shifted.columns = [f"lagged_{lag}_{col}" for col in energy_cols]
        df_lagged[shifted.columns] = shifted

    return df_lagged
