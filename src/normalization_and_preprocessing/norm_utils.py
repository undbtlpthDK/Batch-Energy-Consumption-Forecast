from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path.cwd()
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"


def drop_warmup_rows(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Removes the first n rows of each entity.

    That function is needed to remove all the rows affected by
    rolling and log values calculation because in the first
    iteration on entity they create missing values.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to remove from
    window : int
        Variable that specify the amount of first rows to remove
        Should have same length as the longest horizon of lag
        or roll features.
    Returns
    -------
    pd.DataFrame
        DataFrame without rows that was affected by rolling and
        lag values calculation.
    """
    df_drop = (
        df.groupby("object_id")
        .nth(slice(n, None))
        .reset_index(drop=True)
    )
    return df_drop  # type: ignore


def validate_no_missing_after_warmup(
    df: pd.DataFrame,
) -> None:
    """
    Validates that no missing values remain after warm-up removal.

    This function should be called AFTER `drop_warmup_rows`.
    It fails fast if any NaNs are present.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame after warm-up rows were removed.

    Raises
    ------
    ValueError
        If any missing values are detected.
    """
    nan_summary = (
        df.isna()
        .sum()
        .loc[lambda x: x > 0]
    )

    if not nan_summary.empty:
        raise ValueError(
            "Missing values detected after warm-up removal:\n"
            f"{nan_summary}"
        )


def create_data_split(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Split the DataFrame from start to the end date, end date not included.

    Required date format "YYYY-MM-DD HH:MM:SS"
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to split
    start : str
        Starting date of the split.
    end : str
        End date of the split

    Returns
    -------
    pd.DataFrame
        DataFrame split
    """
    df_split = df.loc[
        (df["timestamp"] >= start)
        & (df["timestamp"] < end)
    ]
    return df_split


def load_processed_parquet(df_name: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DATA_DIR / f"{df_name}.parquet")
    return df


def write_normalized_parquet(df: pd.DataFrame, df_name: str):
    output_path = PROCESSED_DATA_DIR / f"{df_name}.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


def downcast_float_in_df(
    df: pd.DataFrame,
    columns_to_exclude: List[str]
) -> pd.DataFrame:
    """
    Downcast float64 columns to float32 to reduce memory usage.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns_to_exclude : List[str]
        Columns that must not be reduced (e.g. timestamps, ids, targets).

    Returns
    -------
    pd.DataFrame
        DataFrame with float64 columns converted to float32 where applicable.
    """
    df = df.copy()

    for col in df.columns:
        if col in columns_to_exclude:
            continue

        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")

    return df


def load_splits() -> dict:
    start = pd.Timestamp("2017-04-01 01:00:00")
    roll_and_lag_end = start + pd.DateOffset(days=7)
    train_end = roll_and_lag_end + pd.DateOffset(days=364)
    dev_end = train_end + pd.DateOffset(days=91)
    test_end = dev_end + pd.DateOffset(days=364)
    end = pd.Timestamp("2020-04-01 00:00:00")
    splits = {
        "Rolling offset": [start, roll_and_lag_end],
        "Train set": [roll_and_lag_end, train_end],
        "Dev set": [train_end, dev_end],
        "Test set": [dev_end, test_end],
        "Left": [test_end, end],
    }
    return splits


def visualize_splits(splits: dict) -> None:
    plt.figure(figsize=(12, 2.5))

    plt.hlines(
        y=1,
        xmin=splits["Rolling offset"][0],
        xmax=splits["Rolling offset"][1],
        linewidth=12,
        label="ROLL & LAG",
        color="tab:red",
    )

    plt.hlines(
        y=1,
        xmin=splits["Train set"][0],
        xmax=splits["Train set"][1],
        linewidth=12,
        label="TRAIN",
        color="tab:blue",
    )

    plt.hlines(
        y=1,
        xmin=splits["Dev set"][0],
        xmax=splits["Dev set"][1],
        linewidth=12,
        label="DEV",
        color="tab:orange",
    )

    plt.hlines(
        y=1,
        xmin=splits["Test set"][0],
        xmax=splits["Test set"][1],
        linewidth=12,
        label="TEST",
        color="tab:green",
    )

    plt.hlines(
        y=1,
        xmin=splits["Left"][0],
        xmax=splits["Left"][1],
        linewidth=12,
        label="DATA LEFT",
        color="tab:gray",
    )

    plt.yticks([])
    plt.xlabel("Time")
    plt.title("Time-based Split: Rolling / Train / Dev / Test for 168 Hours long multi-horizon forecast")
    plt.legend(loc="upper center", ncol=5, frameon=False)
    plt.tight_layout()
    plt.show()


def energy_import_export_split(df: pd.DataFrame):
    """
    Split a combined import/export feature table into two independent datasets:
    one for energy import and one for energy export.

    Returns
    -------
    df_import : pd.DataFrame
        Dataset for energy import forecasting
    df_export : pd.DataFrame
        Dataset for energy export forecasting
    """

    base_cols = [
        "object_id",
        "timestamp",
        "forecast_origin_time",
        "horizon_index",
        "day",
        "hour",
        "hour_sin",
        "hour_cos",
        "week",
        "week_sin",
        "week_cos",
        "weekday",
        "is_weekend",
        "month",
        "month_sin",
        "month_cos",
        "is_holiday",
        "region_id",
        "temperature_2m",
        "rain",
        "snowfall",
        "cloud_cover",
        "weather_code",
        "is_day",
        "wind_speed_10m",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
    ]

    import_cols = base_cols + [
        "energy_import_kwh",
        "import_lag_1",
        "import_lag_24",
        "import_lag_168",
        "import_rolling_mean_24",
        "import_rolling_mean_168",
        "import_rolling_std_168",
    ]

    export_cols = base_cols + [
        "energy_export_kwh",
        "export_lag_1",
        "export_lag_24",
        "export_lag_168",
        "export_rolling_mean_24",
        "export_rolling_mean_168",
        "export_rolling_std_168",
    ]

    df_import = (
        df[import_cols]
        .rename(
            columns={
                "energy_import_kwh": "energy_kwh",
                "import_lag_1": "lag_1",
                "import_lag_24": "lag_24",
                "import_lag_168": "lag_168",
                "import_rolling_mean_24": "rolling_mean_24",
                "import_rolling_mean_168": "rolling_mean_168",
                "import_rolling_std_168": "rolling_std_168",
            }
        )
        .copy()
    )

    df_export = (
        df[export_cols]
        .rename(
            columns={
                "energy_export_kwh": "energy_kwh",
                "export_lag_1": "lag_1",
                "export_lag_24": "lag_24",
                "export_lag_168": "lag_168",
                "export_rolling_mean_24": "rolling_mean_24",
                "export_rolling_mean_168": "rolling_mean_168",
                "export_rolling_std_168": "rolling_std_168",
            }
        )
        .copy()
    )

    return df_import, df_export


def prepare_splits(df: pd.DataFrame, splits: dict, mode: str):

    if mode == "naive" or mode == "ARIMA":
        df_to_split = df[['object_id', 'timestamp', 'energy_kwh']].copy()
    elif mode == "LightGBM" or mode == "PR":
        df_to_split = df.drop(columns=[
            'month', "hour", "week", "weather_code", "region_id"]
        )
    else:
        raise ValueError

    train = create_data_split(
        df_to_split,
        start=splits['Train set'][0],
        end=splits['Train set'][1]
        )
    dev = create_data_split(
        df_to_split,
        start=splits['Dev set'][0],
        end=splits['Dev set'][1]
    )
    test = create_data_split(
        df_to_split,
        start=splits['Test set'][0],
        end=splits['Test set'][1]
    )

    return [train, dev, test]


def normalize_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype("int8")
    return df


def zscore_scale_float_columns(
    df: pd.DataFrame,
    exclude: Tuple[str] = ("energy_kwh",),
    stats: Dict[str, Tuple[float, float]] | None = None,
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Apply z-score scaling to all float columns.

    If stats is None:
        - compute (mean, std) from df (TRAIN split)
    If stats is provided:
        - reuse them (DEV / TEST)

    Parameters
    ----------
    df : pd.DataFrame
    exclude : Tuple[str]
        Columns to exclude from scaling (e.g. target)
    stats : Dict[str, Tuple[float, float]]
        Optional dict {col: (mean, std)}

    Returns
    -------
    df_scaled : pd.DataFrame
        Scalled DataFrame
    stats : dict {col: (mean, std)}
        Dictionary to reapply scalaing without data leakage 
    """

    df = df.copy()

    float_cols = (
        df.select_dtypes(include="float")
        .columns
        .difference(exclude)
    )

    if stats is None:
        stats = {}
        for col in float_cols:
            mean = df[col].mean()
            std = df[col].std()
            stats[col] = (mean, std)
            if std > 0:
                df[col] = (df[col] - mean) / std
    else:
        for col in float_cols:
            mean, std = stats[col]
            if std > 0:
                df[col] = (df[col] - mean) / std

    return df, stats
