import feature_adding
import feature_utils

df_id = feature_utils.load_raw_parquet(df_name='smart_meter_metadata')
df_sm = feature_utils.load_raw_parquet(df_name='smart_meter_readings')
df_weather = feature_utils.load_raw_parquet(df_name='weather')


df_norm = feature_adding.user_normalization(df_sm)
df_norm_lag = feature_adding.add_lagged_energy_values(df_norm)
df_norm_lag_rol = feature_adding.add_rolling_stats(df_norm_lag)
df_norm_lag_rol_cal = feature_adding.add_calendar_features(df_norm_lag_rol)
df_sm_complete = feature_adding.join_weather(
    df_norm_lag_rol_cal, df_id, df_weather
    )

print("eng done, saving file")

feature_utils.write_processed_sm_readings(df_sm_complete)
