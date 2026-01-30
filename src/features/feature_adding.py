import numpy as np
import pandas as pd
from feature_utils import is_lv_holiday, is_weekend


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
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
