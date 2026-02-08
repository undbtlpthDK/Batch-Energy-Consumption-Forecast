from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path.cwd()
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"


# move to data_utils, drop nan rows
# Splits


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
    df_split = df.loc[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    return df_split


def load_splits() -> dict:
    start = pd.Timestamp("2017-04-01 01:00:00")
    roll_and_lag_end = start + pd.DateOffset(days=7)
    train_end = roll_and_lag_end + pd.DateOffset(days=637)
    dev_end = train_end + pd.DateOffset(days=91)
    test_end = dev_end + pd.DateOffset(days=357)
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
    plt.title(
        "Time-based Split: Rolling / Train / Dev / Test for \
              168 Hours long multi-horizon forecast"
    )
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


def prepare_splits(df: pd.DataFrame, splits: dict, model_type: str):

    if model_type == "naive" or model_type == "ARIMA":
        df_to_split = df[["object_id", "timestamp", "energy_kwh"]].copy()
    elif model_type == "LightGBM" or model_type == "PR":
        df_to_split = df.drop(
            columns=["month", "hour", "week", "weather_code", "region_id"]
        )
    else:
        raise ValueError

    train = create_data_split(
        df_to_split, start=splits["Train set"][0], end=splits["Train set"][1]
    )
    dev = create_data_split(
        df_to_split, start=splits["Dev set"][0], end=splits["Dev set"][1]
    )
    test = create_data_split(
        df_to_split, start=splits["Test set"][0], end=splits["Test set"][1]
    )

    return [train, dev, test]


# Normalization


def normalize_bool_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converts boolean column values to int

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe

    Returns
    -------
    pd.DataFrame
        dataframe with normalized columns
    """
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype("int8")
    return df


def normalize_categorical_columns(
    df: pd.DataFrame,
    categorical_cols: list[str],
) -> pd.DataFrame:
    """One-hot encodes selected categorical columns without dropping originals.

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    categorical_cols : list[str]
        columns to one-hot encode

    Returns
    -------
    pd.DataFrame
        dataframe with added one-hot encoded columns

    Raises
    ------
    KeyError
        raised if provided column isn't preset id dataframe
    """
    df_out = df.copy()

    for col in categorical_cols:
        if col not in df_out.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

        dummies = pd.get_dummies(
            df_out[col],
            prefix=col,
            dtype="int8",
        )

        df_out = pd.concat([df_out, dummies], axis=1)

    return df_out


# Scaling


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
        input dataframe
    exclude : Tuple[str]
        columns to exclude from scaling (e.g. target)
    stats : Dict[str, Tuple[float, float]]
        optional dict {col: (mean, std)}

    Returns
    -------
    df_scaled : pd.DataFrame
        scaled DataFrame
    stats : dict {col: (mean, std)}
        dictionary to reapply scaling without data leakage
    """

    df = df.copy()

    float_cols = df.select_dtypes(include="float").columns.difference(
        exclude  # type: ignore
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


# Rows with nan removing


def remove_warmup_and_validate(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Removes raws with nan values that appear due the rolling and
    lag value calculation

    Parameters
    ----------
    df : pd.DataFrame
        input dataframe
    n : int
        length of the greatest lag or roll horizon

    Returns
    -------
    pd.DataFrame
        dataframe with removed

    Raises
    ------
    ValueError
       is raised if there some nan values left in the dataframe
    """

    df = (
        df.groupby("object_id").nth(slice(n, None)).reset_index(drop=True)
    )  # type: ignore

    nan_summary = df.isna().sum().loc[lambda x: x > 0]
    if not nan_summary.empty:
        raise ValueError(
            "Unexpected NaNs after warm-up removal:\n" f"{nan_summary}"
        )

    return df
