import pandas as pd
import pytest

from energycast.features import statistical_features


@pytest.fixture
def freeze_df():
    return pd.DataFrame(
        {
            "object_id": ["A", "A", "A", "B", "B", "B"],
            "timestamp": pd.date_range("2023-01-01", periods=6, freq="h"),
            "lag_1": [1, 2, 3, 10, 20, 30],
            "roll_3": [5, 6, 7, 50, 60, 70],
        }
    )


@pytest.fixture
def energy_df():
    return pd.DataFrame(
        {
            "object_id": ["A"] * 5 + ["B"] * 5,
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
            "energy_import": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
            "energy_export": [0, 1, 0, 1, 0, 5, 4, 3, 2, 1],
        }
    )


def test_add_customer_averages_creates_columns(energy_df):
    result = statistical_features.add_customer_averages(
        energy_df, shift_hours=1, windows_days=[1]
    )

    assert "customer_mean_energy_import" in result.columns
    assert "customer_1d_mean_energy_import" in result.columns
    assert "object_id" in result.columns


def test_add_customer_averages_shift_prevents_leakage(energy_df):
    result = statistical_features.add_customer_averages(
        energy_df, shift_hours=1, windows_days=[1]
    )

    first_a = result[result["object_id"] == "A"].iloc[0]
    assert pd.isna(first_a["customer_mean_energy_import"])


def test_add_rolling_stats_creates_columns(energy_df):
    result = statistical_features.add_rolling_stats(
        energy_df, windows=[2], aggs=["mean"]
    )

    assert "rolling_mean_energy_import_2" in result.columns
    assert "rolling_mean_energy_export_2" in result.columns


def test_add_rolling_stats_grouped_by_customer(energy_df):
    result = statistical_features.add_rolling_stats(
        energy_df, windows=[2], aggs=["mean"]
    )

    a_vals = result[result["object_id"] == "A"]["rolling_mean_energy_import_2"]
    b_vals = result[result["object_id"] == "B"]["rolling_mean_energy_import_2"]

    assert a_vals.notna().sum() > 0
    assert b_vals.notna().sum() > 0


def test_add_lagged_energy_values_creates_columns(energy_df):
    result = statistical_features.add_lagged_energy_values(
        energy_df, lags=[1, 2]
    )

    assert "lagged_1_energy_import" in result.columns
    assert "lagged_2_energy_export" in result.columns


def test_add_horizon_index(freeze_df):
    result = statistical_features.add_horizon_index(freeze_df, horizon=2)

    assert "horizon_index_2" in result.columns
    assert list(result[result["object_id"] == "A"]["horizon_index_2"]) == [
        0,
        1,
        0,
    ]
    assert list(result[result["object_id"] == "B"]["horizon_index_2"]) == [
        0,
        1,
        0,
    ]


def test_freeze_history_features_creates_frozen_columns(freeze_df):
    result = statistical_features.freeze_history_features(
        freeze_df, window=2, extra_freeze_cols=[]
    )

    assert "lag_1_frozen" in result.columns
    assert "roll_3_frozen" in result.columns

    a_block = result[result["object_id"] == "A"].iloc[:2]
    assert (a_block["lag_1_frozen"] == a_block["lag_1_frozen"].iloc[0]).all()


def test_freeze_history_features_preserves_object_id(freeze_df):
    result = statistical_features.freeze_history_features(
        freeze_df, window=2, extra_freeze_cols=[]
    )

    assert "object_id" in result.columns
    assert set(result["object_id"]) == {"A", "B"}


def test_convert_to_multi_horizon_smoke(freeze_df):
    result = statistical_features.convert_to_multi_horizon(
        freeze_df, window=2, extra_freeze_cols=[]
    )

    assert "horizon_index_2" in result.columns
    assert "lag_1_frozen" in result.columns
    assert "forecast_origin_time" in result.columns
    assert len(result) == len(freeze_df)
    assert "forecast_origin_time" in result.columns
    assert len(result) == len(freeze_df)
