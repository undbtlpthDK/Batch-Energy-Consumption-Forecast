import numpy as np
import pandas as pd
import pytest

from energycast.utils import data_utils


@pytest.fixture
def base_df():
    return pd.DataFrame(
        {
            "small_float": [0.1, 0.2, 0.3],  # should be converted to float 16
            "large_float": [1e6, 2e6, 3e6],  # shouldn't be converted
            "int_col": [1, 2, 3],
        }
    )


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {
            "a": [1, 2],
            "b": [3, 4],
        }
    )


@pytest.fixture
def time_df():
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=10, freq="h"),
            "value": range(10),
        }
    )


# Downcast test


def test_downcast_float_ranges(base_df):
    result = data_utils.downcast_float_in_df(base_df, columns_to_exclude=[])

    assert result["small_float"].dtype == np.float16
    assert result["large_float"].dtype == np.float32


def test_downcast_excluded_columns(base_df):
    result = data_utils.downcast_float_in_df(
        base_df, columns_to_exclude=["small_float"]
    )

    assert result["small_float"].dtype == np.float64
    assert result["large_float"].dtype == np.float32


def test_original_dataframe_not_modified(base_df):
    _ = data_utils.downcast_float_in_df(base_df, columns_to_exclude=[])

    assert base_df["small_float"].dtype == np.float64
    assert base_df["large_float"].dtype == np.float64


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


# Splits


def test_create_data_split_basic(time_df):
    result = data_utils._create_data_split(
        time_df,
        start="2023-01-01 02:00:00",
        end="2023-01-01 05:00:00",
    )

    assert len(result) == 3
    assert result["timestamp"].min() >= pd.Timestamp("2023-01-01 02:00:00")
    assert result["timestamp"].max() < pd.Timestamp("2023-01-01 05:00:00")


def test_prepare_splits_with_dev(time_df):
    splits = {
        "train": (
            pd.Timestamp("2023-01-01 00:00:00"),
            pd.Timestamp("2023-01-01 04:00:00"),
        ),
        "dev": (
            pd.Timestamp("2023-01-01 04:00:00"),
            pd.Timestamp("2023-01-01 07:00:00"),
        ),
        "test": (
            pd.Timestamp("2023-01-01 07:00:00"),
            pd.Timestamp("2023-01-01 10:00:00"),
        ),
    }

    train, dev, test = data_utils.prepare_splits(time_df, splits)

    assert len(train) == 4
    assert len(dev) == 3
    assert len(test) == 3


def test_prepare_splits_without_dev(time_df):
    splits = {
        "train": (
            pd.Timestamp("2023-01-01 00:00:00"),
            pd.Timestamp("2023-01-01 06:00:00"),
        ),
        "test": (
            pd.Timestamp("2023-01-01 06:00:00"),
            pd.Timestamp("2023-01-01 10:00:00"),
        ),
    }

    train, test = data_utils.prepare_splits(time_df, splits)

    assert len(train) == 6
    assert len(test) == 4
