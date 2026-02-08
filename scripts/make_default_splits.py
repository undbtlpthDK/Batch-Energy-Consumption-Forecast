import logging

from energycast.features import (
    calendar_features,
    statistical_features,
    weather_features,
)
from energycast.normalization_and_preprocessing import norm_utils
from energycast.utils import data_utils

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("Loading raw data...")
    df_readings = data_utils.load_parquet("raw", "smart_meter_readings")
    df_id = data_utils.load_parquet("raw", "smart_meter_metadata")
    df_weather = data_utils.load_parquet("raw", "weather")

    # Selecting only most neccessary columns for default dataset
    df_sm = df_readings[["timestamp", "object_id", "energy_import_kwh"]]
    df_sm = df_sm.rename(columns={"energy_import_kwh": "energy_kwh"})
    df_sm = data_utils.downcast_float_in_df(df_sm, columns_to_exclude=[])

    # Add stats features
    logger.info("Building statistical features...")
    df_sm = statistical_features.add_rolling_stats(
        df_sm, windows=[6, 12, 24], aggs=["mean", "std"]
    )
    df_sm = statistical_features.add_lagged_energy_values(
        df_sm, lags=[1, 6, 12, 24, 168]
    )
    df_sm = data_utils.downcast_float_in_df(df_sm, columns_to_exclude=[])
    df_sm = statistical_features.freeze_history_features(df_sm, 24, [])
    df_sm = statistical_features.add_customer_averages(df_sm)
    df_sm = data_utils.downcast_float_in_df(df_sm, columns_to_exclude=[])

    # Add calendar features
    logger.info("Building calendar features...")
    df_sm = calendar_features.add_calendar_features(df_sm)
    df_sm = data_utils.downcast_float_in_df(df_sm, columns_to_exclude=[])

    # Add weather features
    logger.info("Adding weather features...")
    df_sm = weather_features.join_weather(
        df=df_sm, df_id=df_id, df_weather=df_weather
    )
    df_sm = data_utils.downcast_float_in_df(df_sm, columns_to_exclude=[])

    # Normalization
    df_sm = norm_utils.normalize_bool_columns(df_sm)
    df_sm = norm_utils.normalize_categorical_columns(
        df_sm, categorical_cols=["weekday", "hour"]
    )

    # Splitting and storing
    logger.info("Splitting + saving...")
    splits, name = data_utils.load_splits("default_24")
    splits_dfs = data_utils.prepare_splits(df=df_sm, splits=splits)
    data_utils.save_splits(splits_dfs, name)
    logger.info("Done.")


if __name__ == "__main__":
    main()
