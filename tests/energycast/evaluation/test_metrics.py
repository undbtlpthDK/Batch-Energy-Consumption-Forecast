import pandas as pd
import pytest

from energycast.evaluation.metrics import (
    calculate_MAE,
    calculate_RMSE,
    calculate_sMAPE,
    evaluate_forecasts,
)


@pytest.fixture
def simple_forecast_df():
    return pd.DataFrame(
        {
            "id": ["A", "A", "B", "B"],
            "actual": [10.0, 12.0, 20.0, 22.0],
            "pred": [11.0, 11.0, 18.0, 24.0],
        }
    )


def test_calculate_mae_per_id(simple_forecast_df):
    avg, per_id = calculate_MAE(simple_forecast_df, "pred", "actual", "id")

    assert per_id.loc["A"] == pytest.approx(1.0)
    assert per_id.loc["B"] == pytest.approx(2.0)

    assert avg == pytest.approx(1.5)

    assert per_id.name == "MAE"


def test_calculate_rmse_per_id(simple_forecast_df):
    avg, per_id = calculate_RMSE(simple_forecast_df, "pred", "actual", "id")

    assert per_id.loc["A"] == pytest.approx(1.0)
    assert per_id.loc["B"] == pytest.approx(2.0)

    assert avg == pytest.approx(1.5)
    assert per_id.name == "RMSE"


def test_calculate_smape_basic(simple_forecast_df):
    avg, per_id = calculate_sMAPE(simple_forecast_df, "pred", "actual", "id")

    assert per_id.index.tolist() == ["A", "B"]
    assert (per_id >= 0).all()
    assert avg >= 0
    assert per_id.name == "sMAPE"


def test_smape_zero_denominator():
    df = pd.DataFrame(
        {
            "id": ["A"],
            "actual": [0.0],
            "pred": [0.0],
        }
    )

    avg, per_id = calculate_sMAPE(df, "pred", "actual", "id")

    assert per_id.loc["A"] == 0.0
    assert avg == 0.0


def test_evaluate_forecasts_output_structure(simple_forecast_df):
    results, per_customer = evaluate_forecasts(
        simple_forecast_df,
        pred_col="pred",
        actual_col="actual",
        id_col="id",
    )

    assert set(results.keys()) == {
        "MAE_average",
        "RMSE_average",
        "sMAPE_average",
    }

    for value in results.values():
        assert isinstance(value, float)

    assert set(per_customer.columns) == {
        "id",
        "MAE",
        "RMSE",
        "sMAPE",
    }

    assert len(per_customer) == 2
    assert len(per_customer) == 2
