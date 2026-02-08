from typing import Dict, Tuple

import pandas as pd

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
