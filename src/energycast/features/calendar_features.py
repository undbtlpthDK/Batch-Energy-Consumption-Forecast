from datetime import date

import numpy as np
import pandas as pd


def add_calendar_features(
    df: pd.DataFrame,
    *,
    hours: bool = True,
    dates: bool = True,
    weekdays: bool = True,
    weekends: bool = True,
    holidays: bool = True,
    weeks: bool = True,
    months: bool = True,
) -> pd.DataFrame:
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
    df = df.sort_values(["object_id", "timestamp"]).copy()
    if hours:
        df = add_hours(df)
    if dates:
        df = add_dates(df)
    if weekdays:
        if weekends:
            df = add_weekdays(df, weekends=True)
        else:
            df = add_weekdays(df)
    if holidays:
        df = add_holidays(df)
    if weeks:
        df = add_weeks(df)
    if months:
        df = add_months(df)
    return df


def add_hours(df: pd.DataFrame) -> pd.DataFrame:
    df["hour"] = df["timestamp"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    return df


def add_weeks(df: pd.DataFrame) -> pd.DataFrame:
    df["week"] = df["timestamp"].dt.isocalendar().week.astype(int)
    df["week_sin"] = np.sin(2 * np.pi * df["week"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week"] / 52)
    return df


def add_weekdays(df: pd.DataFrame, weekends: bool = False) -> pd.DataFrame:
    df["weekday"] = df["timestamp"].dt.dayofweek
    if weekends:
        df["is_weekend"] = df["weekday"].apply(lambda d: is_weekend("lv", d))
    return df


def add_months(df: pd.DataFrame) -> pd.DataFrame:
    df["month"] = df["timestamp"].dt.month
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    return df


def add_holidays(df: pd.DataFrame) -> pd.DataFrame:
    df["is_holiday"] = df["timestamp"].apply(is_lv_holiday)
    return df


def add_dates(df: pd.DataFrame) -> pd.DataFrame:
    df["day"] = df["timestamp"].dt.day
    return df


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
