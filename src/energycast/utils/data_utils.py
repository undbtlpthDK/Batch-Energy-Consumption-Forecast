from pathlib import Path
from typing import List

import pandas as pd

ROOT_DIR = Path.cwd()
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"


def downcast_float_in_df(
    df: pd.DataFrame, columns_to_exclude: List[str]
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


def load_raw_parquet(df_name: str) -> pd.DataFrame:
    """Converts raw data parquet file to DataFrame

    Parameters
    ----------
    df_name : str
        Name of raw parquet file

    Returns
    -------
    pd.DataFrame
        Parquet file as pandas DataFrame
    """
    print(RAW_DATA_DIR / f"{df_name}.parquet")
    df = pd.read_parquet(RAW_DATA_DIR / f"{df_name}.parquet")
    return df


def write_processed_sm_readings(df: pd.DataFrame, name: str) -> Path:
    """
    Write smart meter readings to raw parquet.
    """
    output_path = PROCESSED_DATA_DIR / f"{name}.parquet"
    df.to_parquet(output_path, index=False)

    return output_path
