import pandas as pd
import pytest

from energycast.features import calendar_features


@pytest.fixture
def calendar_df():
    return pd.DataFrame(
        {
            "id": [1, 2],
            "timestamp": ["2017-04-01 01:00:00", "2025-06-24 09:00:00"],
        }
    )


@pytest.fixture
def sample_dates():
    return ["2017-04-01 01:00:00", "2025-06-24 09:00:00"]


def test_add_calendar_features(calendar_df):

    calendar_df["timestamp"] = pd.to_datetime(calendar_df["timestamp"])

    df = calendar_features.add_calendar_features(calendar_df)
    assert df.loc[df["id"] == 1, "hour"].iloc[0] == 1
    assert df.loc[df["id"] == 1, "day"].iloc[0] == 1
    assert df.loc[df["id"] == 1, "weekday"].iloc[0] == 5
    assert df.loc[df["id"] == 1, "is_weekend"].iloc[0]
    # Not a holiday
    assert not df.loc[df["id"] == 1, "is_holiday"].iloc[0]
    assert df.loc[df["id"] == 1, "week"].iloc[0] == 13
    assert df.loc[df["id"] == 1, "month"].iloc[0] == 4

    # Is a holiday
    assert df.loc[df["id"] == 2, "is_holiday"].iloc[0]


def test_is_lv_holiday(sample_dates):
    assert not calendar_features.is_lv_holiday(pd.to_datetime(sample_dates[0]))
    assert calendar_features.is_lv_holiday(pd.to_datetime(sample_dates[1]))
