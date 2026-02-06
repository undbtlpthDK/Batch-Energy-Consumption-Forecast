from datetime import date
from pathlib import Path
from typing import List

import pandas as pd

ROOT_DIR = Path.cwd()
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"


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


def is_weekend(country: str, weekday: int) -> bool:
    """Checks is the weekday weekend

    Parameters
    ----------
    country : str
        Origin of the smart meter dataset
    weekday : int
        Weekday to classify

    Returns
    -------
    bool
        True - This weekday is considered as weekend in specified country
    """
    if country == "lv":
        if weekday in (5, 6):
            return True

    return False


def latvia_public_holidays(year: int) -> set[date]:
    """Generates set of Latvian year dates that are holidays

    Parameters
    ----------
    year : int
        Year to base set on

    Returns
    -------
    set[date]
        Set of dates that fixed-date holidays in Latvian region
    """
    return {
        # Fixed-date holidays
        date(year, 1, 1),
        date(year, 5, 1),
        date(year, 6, 23),
        date(year, 6, 24),
        date(year, 11, 18),
        date(year, 12, 24),
        date(year, 12, 25),
        date(year, 12, 26),
    }


def is_lv_holiday(d: pd.Timestamp) -> bool:
    """Checks is the date a Latvian holiday

    Parameters
    ----------
    d : date
        date to check

    Returns
    -------
    bool
        True - is holiday
    """
    d_local = d.normalize().date()
    return d_local in latvia_public_holidays(d_local.year)
