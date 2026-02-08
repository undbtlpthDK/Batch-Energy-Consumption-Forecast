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
