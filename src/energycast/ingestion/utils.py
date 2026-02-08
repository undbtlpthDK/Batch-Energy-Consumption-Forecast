from pathlib import Path
from typing import Optional

import pandas as pd

from energycast.ingestion.weather_utils import (
    fetch_and_save_weather_for_regions,
)

ROOT_DIR = Path.cwd()
ORIGINAL_SM_DIR = ROOT_DIR / "data" / "original" / "lv_smart_meters"
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"


def pull_meta_data():
    df_id = pd.read_csv(ORIGINAL_SM_DIR / "smart_meter_metadata.csv")
    print(ORIGINAL_SM_DIR, ROOT_DIR)
    return df_id


def pull_regions_data():
    regions = pd.read_csv(ORIGINAL_SM_DIR / "regions_coordinates.csv")
    return regions


def pull_smart_meter_data():

    dfs = [
        pd.read_csv(ORIGINAL_SM_DIR / "smart_meter_readings_1.csv"),
        pd.read_csv(ORIGINAL_SM_DIR / "smart_meter_readings_2.csv"),
        pd.read_csv(ORIGINAL_SM_DIR / "smart_meter_readings_3_R.csv"),
        pd.read_csv(ORIGINAL_SM_DIR / "smart_meter_readings_4.csv"),
        pd.read_csv(ORIGINAL_SM_DIR / "smart_meter_readings_5.csv"),
        pd.read_csv(ORIGINAL_SM_DIR / "smart_meter_readings_6_R.csv"),
    ]

    sm_df = pd.concat(dfs, ignore_index=True)
    return sm_df


def pull_weather_data(mode: str, regions_df: pd.DataFrame):
    if mode == "backtest":
        historical_weather = fetch_and_save_weather_for_regions(
            mode="historical",
            regions=regions_df,
            output_dir=None,
            start_date="2017-04-01",
            end_date="2020-04-01",
        )
        wearther_df = pd.concat(historical_weather, ignore_index=True)
        return wearther_df


def write_sm_metadata_raw(df: pd.DataFrame) -> Path:
    """
    Write smart meter metadata to raw parquet.
    """
    output_path = RAW_DATA_DIR / "smart_meter_metadata.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


def write_sm_readings_raw(df: pd.DataFrame) -> Path:
    """
    Write smart meter readings to raw parquet.
    """
    output_path = RAW_DATA_DIR / "smart_meter_readings.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


def write_weather_raw(df: pd.DataFrame) -> Path:
    """
    Write weather data to raw parquet.
    """
    output_path = RAW_DATA_DIR / "weather.parquet"
    df.to_parquet(output_path, index=False)
    return output_path


def check_duplicates(df, pk: str):
    dup_stats = df.groupby([pk, "timestamp"]).size().reset_index(name="count")

    dup_stats = dup_stats[dup_stats["count"] > 1]

    # LOG  This later
    print(
        f"{dup_stats.head(1)} \n duplicated timestamps: \
        {len(dup_stats['timestamp'])} , amount of rows  \
        affected:{sum(dup_stats['count'])}"
    )
    return len(dup_stats["timestamp"])


def handle_duplicates(df, dataset_type: str):
    """
    Canonicalize duplicates for different datasets.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    dataset_type : str
        One of {"smart_meter", "weather"}.

    Returns
    -------
    pd.DataFrame
        Canonicalized dataframe with duplicates handled.
    """

    if dataset_type == "smart_meter":
        mask_sum_window = (df["timestamp"].dt.month == 10) & (
            df["timestamp"].dt.day.between(25, 30)
        )

        df_sum = df[mask_sum_window]
        df_rest = df[~mask_sum_window]

        df_sum_canonical = df_sum.groupby(
            ["customer_id", "timestamp"], as_index=False
        ).agg(
            {
                "energy_import_kwh": "sum",
                "energy_export_kwh": "sum",
                **{
                    col: "first"
                    for col in df_sum.columns
                    if col
                    not in {
                        "customer_id",
                        "timestamp",
                        "energy_import_kwh",
                        "energy_export_kwh",
                    }
                },
            }
        )

        df_rest_canonical = df_rest.sort_values("timestamp").drop_duplicates(
            subset=["customer_id", "timestamp"],
            keep="first",
        )

        canonical_df = pd.concat(
            [df_sum_canonical, df_rest_canonical],
            ignore_index=True,
        ).sort_values(["customer_id", "timestamp"])

    elif dataset_type == "weather":
        canonical_df = df.sort_values("timestamp").drop_duplicates(
            subset=["region_id", "timestamp"],
            keep="first",
        )

    else:
        raise ValueError(
            "dataset_type must be one of {'smart_meter', 'weather'}"
        )

    return canonical_df.reset_index(drop=True)


def check_time_continuity(
    df,
    id_col="object_id",
    time_col="timestamp",
    expected_step_hours=1,
):
    """
    Enforce temporal continuity of time-series data per entity.

    This function validates that time-indexed data is continuous and
    strictly ordered per entity. It raises an exception if any gaps
    or time reversals are detected.

    Duplicate detection is intentionally excluded and must be handled
    by a separate validation function.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe containing time-series data.
    id_col : str, default "object_id"
        Column identifying individual time series (e.g. meter/object).
    time_col : str, default "timestamp"
        Datetime column representing event time.
    expected_step_hours : int, default 1
        Expected time delta between consecutive records.

    Returns
    -------
    continuity_summary : pandas.DataFrame
        Per-entity summary with counts of:
        - ok
        - gap
        - time_reversal

    Raises
    ------
    ValueError
        If any gap or time reversal is detected.
    """

    # Deterministic ordering
    df_sorted = df.sort_values([id_col, time_col])

    prev_ts = df_sorted.groupby(id_col)[time_col].shift(1)

    delta_hours = (df_sorted[time_col] - prev_ts).dt.total_seconds() / 3600

    # Initialize flags
    continuity_flag = "ok"
    flags = pd.Series(continuity_flag, index=df_sorted.index)

    flags.loc[delta_hours > expected_step_hours] = "gap"
    flags.loc[delta_hours < 0] = "time_reversal"

    # Build summary
    continuity_summary = (
        pd.concat(
            [df_sorted[id_col], flags.rename("continuity_flag")],
            axis=1,
        )
        .groupby([id_col, "continuity_flag"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # Alert on violations
    violations = continuity_summary[
        continuity_summary.drop(columns=[id_col]).sum(axis=1)
        > continuity_summary.get("ok", 0)
    ]

    if not violations.empty:
        raise ValueError(
            "Time continuity check failed. "
            "Gaps or time reversals detected in smart meter data."
        )

    return continuity_summary


def add_ingestion_time(
    df: pd.DataFrame,
    mode: str,
    forecast_horizon: Optional[int] = None,
) -> pd.DataFrame:

    if mode == "smart_meter_backtest":
        df["ingested_at"] = df["timestamp"]
    elif mode == "weather":
        if forecast_horizon is None:
            raise ValueError(
                "forecast_horizon must be provided for weather mode"
            )
        df["ingested_at"] = df["timestamp"]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return df


def timestamp_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the `timestamp` column to timezone-naive pandas datetime.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"], dayfirst=True)

    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)

    return df
    return df
