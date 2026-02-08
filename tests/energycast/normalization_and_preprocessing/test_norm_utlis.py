import numpy as np
import pandas as pd

from energycast.normalization_and_preprocessing import norm_utils


def test_normalize_bool_columns_converts_bool_to_int():
    df = pd.DataFrame(
        {
            "flag": [True, False, True],
            "value": [1.0, 2.0, 3.0],
        }
    )

    result = norm_utils.normalize_bool_columns(df)

    assert result["flag"].dtype == "int8"
    assert result["flag"].tolist() == [1, 0, 1]
    assert result["value"].dtype == "float64"


def test_normalize_categorical_columns_adds_dummies_and_keeps_original():
    df = pd.DataFrame(
        {
            "categorical_value": ["1", "2", "1"],
            "value": [10, 20, 30],
        }
    )

    result = norm_utils.normalize_categorical_columns(
        df, ["categorical_value"]
    )

    assert "categorical_value" in result.columns
    assert "categorical_value_1" in result.columns
    assert "categorical_value_2" in result.columns

    assert result["categorical_value_1"].tolist() == [1, 0, 1]
    assert result["categorical_value_2"].tolist() == [0, 1, 0]

    assert result["categorical_value_1"].dtype == "int8"


def test_zscore_scale_computes_stats_and_scales():
    df = pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [10.0, 20.0, 30.0],
            "energy_kwh": [100.0, 200.0, 300.0],  # excluded
        }
    )

    df_scaled, stats = norm_utils.zscore_scale_float_columns(df)
    # stats computed
    assert "feature_1" in stats
    assert "feature_2" in stats
    assert "energy_kwh" not in stats

    # scaled columns ~ mean 0
    assert np.isclose(df_scaled["feature_1"].mean(), 0.0)
    assert np.isclose(df_scaled["feature_2"].mean(), 0.0)

    # excluded column untouched
    assert df_scaled["energy_kwh"].equals(df["energy_kwh"])
    assert df_scaled["energy_kwh"].equals(df["energy_kwh"])


def test_zscore_scale_reuses_existing_stats():
    df_train = pd.DataFrame(
        {
            "feature": [1.0, 2.0, 3.0],
        }
    )

    df_test = pd.DataFrame(
        {
            "feature": [4.0, 5.0],
        }
    )

    _, stats = norm_utils.zscore_scale_float_columns(df_train)
    df_scaled, _ = norm_utils.zscore_scale_float_columns(df_test, stats=stats)

    mean, std = stats["feature"]
    expected = (df_test["feature"] - mean) / std

    assert np.allclose(df_scaled["feature"], expected)
