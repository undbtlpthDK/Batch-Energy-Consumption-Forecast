import numpy as np
import pandas as pd
import pytest

from energycast.normalization_and_preprocessing import norm_utils


@pytest.fixture
def bool_df():
    return pd.DataFrame(
        {
            "flag": [True, False, True],
            "value": [1.0, 2.0, 3.0],
        }
    )


@pytest.fixture
def categorical_df():
    return pd.DataFrame(
        {
            "categorical_value": ["1", "2", "1"],
            "value": [10, 20, 30],
        }
    )


@pytest.fixture
def zscore_df():
    return pd.DataFrame(
        {
            "feature_1": [1.0, 2.0, 3.0],
            "feature_2": [10.0, 20.0, 30.0],
            "energy_kwh": [100.0, 200.0, 300.0],
        }
    )


@pytest.fixture
def warmup_df():
    return pd.DataFrame(
        {
            "object_id": ["A"] * 4 + ["B"] * 4,
            "value": [1, 2, 3, 4, 10, 20, 30, 40],
        }
    )


# Normalization


def test_normalize_bool_columns(bool_df):
    result = norm_utils.normalize_bool_columns(bool_df)

    assert result["flag"].dtype == "int8"
    assert result["flag"].tolist() == [1, 0, 1]
    assert result["value"].dtype == "float64"


def test_normalize_categorical_columns(categorical_df):
    result = norm_utils.normalize_categorical_columns(
        categorical_df, ["categorical_value"]
    )

    assert {"categorical_value_1", "categorical_value_2"}.issubset(
        result.columns
    )
    assert result["categorical_value_1"].tolist() == [1, 0, 1]
    assert result["categorical_value_1"].dtype == "int8"


# Scaling


def test_zscore_scale_computes_stats_and_scales(zscore_df):
    df_scaled, stats = norm_utils.zscore_scale_float_columns(zscore_df)

    assert set(stats.keys()) == {"feature_1", "feature_2"}
    assert np.isclose(df_scaled["feature_1"].mean(), 0.0)
    assert df_scaled["energy_kwh"].equals(zscore_df["energy_kwh"])


def test_zscore_scale_reuses_existing_stats():
    df_train = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
    df_test = pd.DataFrame({"feature": [4.0, 5.0]})

    _, stats = norm_utils.zscore_scale_float_columns(df_train)
    df_scaled, _ = norm_utils.zscore_scale_float_columns(df_test, stats=stats)

    mean, std = stats["feature"]
    assert np.allclose(df_scaled["feature"], (df_test["feature"] - mean) / std)


# Remove rows with nan values


def test_remove_warmup_and_validate_removes_rows(warmup_df):
    result = norm_utils.remove_warmup_and_validate(warmup_df, n=2)

    assert result["object_id"].value_counts().to_dict() == {"A": 2, "B": 2}


def test_remove_warmup_and_validate_raises_on_nan():
    df = pd.DataFrame(
        {
            "object_id": ["A", "A", "A"],
            "value": [1.0, None, 3.0],
        }
    )

    with pytest.raises(ValueError):
        norm_utils.remove_warmup_and_validate(df, n=1)
        norm_utils.remove_warmup_and_validate(df, n=1)
