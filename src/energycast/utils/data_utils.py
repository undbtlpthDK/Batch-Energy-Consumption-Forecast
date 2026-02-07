from pathlib import Path
from typing import List

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[3]


ORIGINAL_LV_DATA_DIR = ROOT_DIR / "original" / "lv_smart_meters"
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
