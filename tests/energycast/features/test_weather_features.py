import pandas as pd
import pytest

from energycast.features import weather_features


@pytest.fixture
def weather_dfs():
    df = pd.DataFrame(
        {
            "object_id": [1, 2],
            "timestamp": [1, 1],
            "value": [10, 20],
        }
    )

    df_id = pd.DataFrame(
        {
            "object_id": ["1", "2"],
            "region_id": ["R1", "R2"],
        }
    )

    df_weather = pd.DataFrame(
        {
            "timestamp": [1, 1],
            "region_id": ["R1", "R2"],
            "temp": [5.0, 7.0],
        }
    )

    return df, df_id, df_weather


def test_join_weather_basic(weather_dfs):

    df, df_id, df_weather = weather_dfs
    result = weather_features.join_weather(df, df_id, df_weather)

    assert len(result) == 2
    assert "region_id" in result.columns
    assert "temp" in result.columns
    assert result.loc[0, "temp"] == 5.0
    assert result.loc[1, "temp"] == 7.0
