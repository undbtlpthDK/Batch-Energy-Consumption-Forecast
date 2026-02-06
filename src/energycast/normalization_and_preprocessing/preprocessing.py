import norm_utils

# H - 168

splits = norm_utils.load_splits()

df_h168 = norm_utils.prepare_multi_horizon_dataset(
    dataset_name="multi_horizon_reading_h168",
    horizon=168,
    data_type="import"
)

# Naive and ARIMA (no scaling, reduced column amount)
stat_train_168, stat_dev_168, stat_test_168 = norm_utils.prepare_splits(
    df_h168, splits, model_type="naive"
)

norm_utils.write_normalized_parquet(
    stat_train_168, "stats_models_168_train"
)
norm_utils.write_normalized_parquet(
    stat_dev_168, "stats_models_168_dev"
)
norm_utils.write_normalized_parquet(
    stat_test_168, "stats_models_168_test"
)

# LightGBM (no scaling)
lgbm_train_168, lgbm_dev_168, lgbm_test_168 = norm_utils.prepare_splits(
    df_h168, splits, model_type="LightGBM"
)

norm_utils.write_normalized_parquet(
    lgbm_train_168, "lgbm_multi_horizon_168_train"
)
norm_utils.write_normalized_parquet(
    lgbm_dev_168, "lgbm_multi_horizon_168_dev"
)
norm_utils.write_normalized_parquet(
    lgbm_test_168, "lgbm_multi_horizon_168_test"
)

# Pooled Regression (z-score scaling)
pr_train_168_scaled, scaler_168 = norm_utils.zscore_scale_float_columns(
    lgbm_train_168
)
pr_dev_168_scaled, _ = norm_utils.zscore_scale_float_columns(
    lgbm_dev_168, stats=scaler_168
)
pr_test_168_scaled, _ = norm_utils.zscore_scale_float_columns(
    lgbm_test_168, stats=scaler_168
)

norm_utils.write_normalized_parquet(
    pr_train_168_scaled, "pr_multi_horizon_168_train"
)
norm_utils.write_normalized_parquet(
    pr_dev_168_scaled, "pr_multi_horizon_168_dev"
)
norm_utils.write_normalized_parquet(
    pr_test_168_scaled, "pr_multi_horizon_168_test"
)


# Horizon - 24

# Horizon is used for rows with missing values removal, and as the 168 rolling
# value is calculated for this dataset it needs to 168 rows to be removed
df_h24 = norm_utils.prepare_multi_horizon_dataset(
    dataset_name="multi_horizon_reading_h24",
    horizon=168,
    data_type='import'
)

# LightGBM (no scaling)
lgbm_train_24, lgbm_dev_24, lgbm_test_24 = norm_utils.prepare_splits(
    df_h24, splits, model_type="LightGBM"
)

norm_utils.write_normalized_parquet(
    lgbm_train_24, "lgbm_multi_horizon_24_train"
)
norm_utils.write_normalized_parquet(
    lgbm_dev_24, "lgbm_multi_horizon_24_dev"
)
norm_utils.write_normalized_parquet(
    lgbm_test_24, "lgbm_multi_horizon_24_test"
)

# Pooled Regression (z-score scaling)
pr_train_24_scaled, scaler_24 = norm_utils.zscore_scale_float_columns(
    lgbm_train_24
)
pr_dev_24_scaled, _ = norm_utils.zscore_scale_float_columns(
    lgbm_dev_24, stats=scaler_24
)
pr_test_24_scaled, _ = norm_utils.zscore_scale_float_columns(
    lgbm_test_24, stats=scaler_24
)

norm_utils.write_normalized_parquet(
    pr_train_24_scaled, "pr_multi_horizon_24_train"
)
norm_utils.write_normalized_parquet(
    pr_dev_24_scaled, "pr_multi_horizon_24_dev"
)
norm_utils.write_normalized_parquet(
    pr_test_24_scaled, "pr_multi_horizon_24_test"
)
