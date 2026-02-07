import pandas as pd


def join_weather(
    df: pd.DataFrame, df_id: pd.DataFrame, df_weather: pd.DataFrame
) -> pd.DataFrame:
    """Adds the wether features to the smart meter readings dataset

    Parameters
    ----------
    df : pd.DataFrame
       smart meters readings
    df_id : pd.DataFrame
        customer metadata
    df_weather : pd.DataFrame
        weather readings

    Returns
    -------
    pd.DataFrame
        Smart meters dataset with weather features added
    """

    df = df.sort_values(["object_id", "timestamp"]).copy()

    df["object_id"] = df["object_id"].astype(str)
    df_id["object_id"] = df_id["object_id"].astype(str)

    df = df.join(
        df_id.set_index("object_id")[["region_id"]],
        how="left",
        on="object_id",
    )

    df = pd.merge(
        left=df,
        right=df_weather,
        how="left",
        left_on=["timestamp", "region_id"],
        right_on=["timestamp", "region_id"],
    )
    return df
