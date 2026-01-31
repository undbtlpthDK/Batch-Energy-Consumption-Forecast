import numpy as np
import pandas as pd
from feature_utils import is_lv_holiday, is_weekend


def add_lagged_energy_values(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the lagged energy consumption and generation features to DataFrame

    Parameters
    ----------
    df : pd.DataFrame
        Smart Meters reading DataFrame where
        energy_import_kwh and energy_export_kwh columns should exists
    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with energy related lags
    """
    # Specifies the range of lags applied
    lags = [1, 2, 3, 4, 5, 24, 48, 168]

    df_lagged = df.sort_values(["object_id", "timestamp"]).copy()

    for lag in lags:
        shifted = (
            df_lagged
            .groupby("object_id")[["energy_import_kwh", "energy_export_kwh"]]
            .shift(lag)
        )

        df_lagged[[f"import_lag_{lag}", f"export_lag_{lag}"]] = shifted

    return df_lagged


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the calendar values based on "timestamp" to DataFrame

    columns added:
        date
        hour, hour_sin, hour_coos of the day
        week, week_sin, week_coos of the year
        weekday, weekday_sin, weekday_coos of the week
        is_weekend, is_holiday

    Parameters
    ----------
    df : pd.DataFrame
        Smart Meters reading DataFrame with "timestamp" column

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame
    """
    df = _add_dates(df)
    df = _add_hours(df)
    df = _add_weeks(df)
    df = _add_weekdays(df)
    df = _add_months(df)
    df = _add_holidays(df)
    return df


def _add_hours(df: pd.DataFrame) -> pd.DataFrame:
    df['hour'] = df['timestamp'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    return df


def _add_weeks(df: pd.DataFrame) -> pd.DataFrame:
    df['week'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
    return df


def _add_weekdays(df: pd.DataFrame) -> pd.DataFrame:
    df['weekday'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['weekday'].apply(lambda d: is_weekend("lv", d))
    return df


def _add_months(df: pd.DataFrame) -> pd.DataFrame:
    df['month'] = df['timestamp'].dt.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    return df


def _add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    df['is_holiday'] = df['timestamp'].apply(is_lv_holiday)
    return df


def _add_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["day"] = df["timestamp"].dt.day
    return df
