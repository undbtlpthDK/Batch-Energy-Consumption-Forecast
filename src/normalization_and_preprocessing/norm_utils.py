from pathlib import Path
from typing import List

import pandas as pd

ROOT_DIR = Path.cwd()
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"


def read_processed_parquet() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DATA_DIR / "smart_meter_readings.parquet")
    return df


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
