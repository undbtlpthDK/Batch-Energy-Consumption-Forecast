import pandas as pd
import pytest

from energycast.utils import data_utils


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        }
    )


# Parquet loading


def test_load_parquet_invalid_dir():
    with pytest.raises(ValueError, match="Invalid dir"):
        data_utils.load_parquet("wrong_dir", "test")


def test_load_parquet_success(tmp_path, monkeypatch, sample_df):
    file_path = tmp_path / "data.parquet"
    sample_df.to_parquet(file_path)

    monkeypatch.setattr(
        "energycast.utils.data_utils.RAW_DATA_DIR",
        tmp_path,
    )

    result = data_utils.load_parquet("raw", "data")

    pd.testing.assert_frame_equal(result, sample_df)


# Parquet saving


def test_write_parquet_invalid_dir(sample_df):
    with pytest.raises(ValueError, match="Invalid dir"):
        data_utils.write_parquet(sample_df, "wrong", "test")


def test_write_parquet_empty_df():
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="empty"):
        data_utils.write_parquet(empty_df, "raw", "test")


def test_write_parquet_success(tmp_path, monkeypatch, sample_df):
    monkeypatch.setattr(
        "energycast.utils.data_utils.RAW_DATA_DIR",
        tmp_path,
    )

    output_path = data_utils.write_parquet(sample_df, "raw", "data")

    assert output_path.exists()

    loaded = pd.read_parquet(output_path)
    pd.testing.assert_frame_equal(loaded, sample_df)
