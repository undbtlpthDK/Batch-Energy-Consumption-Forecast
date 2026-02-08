from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]


ORIGINAL_LV_DATA_DIR = ROOT_DIR / "original" / "lv_smart_meters"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"


def downcast_float_in_df(
    df: pd.DataFrame, columns_to_exclude: List[str]
) -> pd.DataFrame:
    """
    Downcast float columns to reduce memory usage.

    Logic:
     - float64 -> float16 if will fit the range of  values
    else:
     - float64 -> float32
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

    float16_min = np.finfo(np.float16).min
    float16_max = np.finfo(np.float16).max

    for col in df.columns:
        if col in columns_to_exclude:
            continue

        if not pd.api.types.is_float_dtype(df[col]):
            continue

        col_min = df[col].min()
        col_max = df[col].max()

        # Skip NaN-only columns
        if pd.isna(col_min) or pd.isna(col_max):
            continue

        # Prefer float16 if safe
        if float16_min <= col_min <= col_max <= float16_max:
            df[col] = df[col].astype(np.float16)
        else:
            df[col] = df[col].astype(np.float32)

    return df


def load_parquet(dir: str, df_name: str) -> pd.DataFrame:
    """Loads parquet as DataFrame

    Parameters
    ----------
    dir : {"original", "raw", "processed"}
        - original - original lv smart meters data
        - raw - raw data from loaded from db storage
        - processed - data with some preprocessing already applied
    df_name : str
       name of parquet file, there should be no .parquet in it

    Returns
    -------
    pd.DataFrame
        requested dataframe
    Raises
    ------
    ValueError
        if dir is not one of the existing
    FileNotFoundError
        if requested file doesn't exist
    """
    dir = dir.lower()

    data_dirs = {
        "raw": RAW_DATA_DIR,
        "original": ORIGINAL_LV_DATA_DIR,
        "processed": PROCESSED_DATA_DIR,
    }

    # If not one of the allowed directories
    if dir not in data_dirs:
        raise ValueError(
            f"Invalid dir '{dir}'. Expected one of: {list(data_dirs.keys())}"
        )

    file_path: Path = data_dirs[dir] / f"{df_name}.parquet"

    if not file_path.exists():
        raise FileNotFoundError(f"Parquet file '{file_path}' does not exist")

    return pd.read_parquet(file_path)


def write_parquet(df: pd.DataFrame, dir: str, name: str) -> Path:
    """Saves DataFrame under specified name in the provided directory

    Parameters
    ----------
    df : pd.DataFrame
        data to save
    dir : {"original", "raw", "processed"}
        directory where to save parquet
        - original - original lv smart meters data
        - raw - raw data from loaded from db storage
        - processed - data with some preprocessing already applied
    name : str
        save as name provided

    Returns
    -------
    Path
        path to the saved file

    Raises
    ------
    ValueError
        if dir is not one of the existing
    ValueError
        if dataframe is not one of the existing
    """
    dir = dir.lower()

    base_dirs = {
        "raw": RAW_DATA_DIR,
        "original": ORIGINAL_LV_DATA_DIR,
        "processed": PROCESSED_DATA_DIR,
    }

    if dir not in base_dirs:
        raise ValueError(
            f"Invalid dir '{dir}'. Expected one of: {list(base_dirs.keys())}"
        )

    if df.empty:
        raise ValueError("DataFrame is empty, nothing to write")

    output_path: Path = base_dirs[dir] / f"{name}.parquet"

    df.to_parquet(output_path, index=False)

    return output_path


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
