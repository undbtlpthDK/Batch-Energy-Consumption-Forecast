
import feature_adding
import feature_utils

df_id = feature_utils.load_raw_parquet(
    df_name="smart_meter_metadata"
)
df_sm = feature_utils.load_raw_parquet(
    df_name="smart_meter_readings"
)
df_weather = feature_utils.load_raw_parquet(
    df_name="weather"
)

# Downcast the column values
df_sm = feature_utils.downcast_float_in_df(
    df_sm,
    columns_to_exclude=[],
)

"""
df_norm = feature_adding.user_normalization(df_sm)
df_norm_lag = feature_adding.add_lagged_energy_values(
    df_norm,
    lags=[1, 2, 3, 4, 5, 24, 48, 168],
)
df_norm_lag_rol = feature_adding.add_rolling_stats(
    df_norm_lag,
    windows=[3, 5, 8, 10, 24, 36, 48, 96, 168, 192],
    stats=["sum", "mean", "std"],
)
df_norm_lag_rol_cal = feature_adding.add_calendar_features(
    df_norm_lag_rol
)
df_sm_complete = feature_adding.join_weather(
    df_norm_lag_rol_cal,
    df_id,
    df_weather,
)
print("features added, saving file")
feature_utils.write_processed_sm_readings(
    df_sm_complete,
    "smart_meter_readings",
)
"""

print("multi-horizon generation in process...")
# Creates a 24-horizon multi-horizon dataset
df_sm_multi_24 = df_sm[
    ["object_id", "timestamp", "energy_import_kwh", "energy_export_kwh"]
]
df_sm_multi_24 = feature_adding.add_lagged_energy_values(
    df_sm_multi_24,
    lags=[1, 24, 168],
)
df_sm_multi_24 = feature_adding.add_rolling_stats(
    df_sm_multi_24,
    windows=[24, 48, 72, 168],
    stats=["mean", "std"],
)

df_sm_multi_24 = feature_utils.downcast_float_in_df(df_sm_multi_24, [])

df_sm_multi_24 = feature_adding.convert_to_multi_horizon(
    df_sm_multi_24,
    24,
)
print("adding calendar features ...")
df_sm_multi_24 = feature_adding.add_calendar_features(
    df_sm_multi_24
)
df_sm_multi_24 = feature_adding.join_weather(
    df_sm_multi_24,
    df_id,
    df_weather,
)

df_sm_multi_24 = feature_utils.downcast_float_in_df(df_sm_multi_24, [])

print("24h multi-horizon df is ready, saving file")
feature_utils.write_processed_sm_readings(
    df_sm_multi_24,
    "multi_horizon_reading_h24",
)


# Creates a 168-horizon multi-horizon dataset
print("multi-horizon generation in process...")
df_sm_multi_168 = df_sm[
    ["object_id", "timestamp", "energy_import_kwh", "energy_export_kwh"]
]
df_sm_multi_168 = feature_adding.add_lagged_energy_values(
    df_sm_multi_168,
    lags=[1, 24, 168],
)
df_sm_multi_168 = feature_adding.add_rolling_stats(
    df_sm_multi_168,
    windows=[24, 48, 72, 168],
    stats=["mean", "std"],
)

df_sm_multi_168 = feature_utils.downcast_float_in_df(df_sm_multi_168, [])

df_sm_multi_168 = feature_adding.convert_to_multi_horizon(
    df_sm_multi_168,
    168,
)
print("adding calendar features ...")
df_sm_multi_168 = feature_adding.add_calendar_features(
    df_sm_multi_168
)
df_sm_multi_168 = feature_adding.join_weather(
    df_sm_multi_168,
    df_id,
    df_weather,
)

df_sm_multi_168 = feature_utils.downcast_float_in_df(df_sm_multi_168, [])

print("168h multi-horizon df is ready, saving file")
feature_utils.write_processed_sm_readings(
    df_sm_multi_168,
    "multi_horizon_reading_h168",
)
