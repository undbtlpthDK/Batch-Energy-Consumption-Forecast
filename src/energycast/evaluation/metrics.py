import numpy as np
import pandas as pd


def evaluate_forecasts(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    id_col: str,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Calculates MAE, RMSE, sMAPE per prediction

    Parameters
    ----------
    df : pd.DataFrame
        df with the predicted column
    pred_col : str
        name of column where predicted value is stored
    actual_col : str
        name of column where original value is stored
    id_col : str
        name of column where id is stored

    Returns
    -------
    tuple[dict[str, float], pd.DataFrame]
        dictionary of average metrics values and
        DataFrame with metrics calculated by id
    """
    average_mae, mae_per_id = calculate_MAE(df, pred_col, actual_col, id_col)

    average_rmse, rmse_per_id = calculate_RMSE(
        df, pred_col, actual_col, id_col
    )

    average_smape, smape_per_id = calculate_sMAPE(
        df, pred_col, actual_col, id_col
    )

    per_customer = pd.concat(
        [mae_per_id, rmse_per_id, smape_per_id],  # type: ignore
        axis=1,
    ).reset_index()  # type: ignore

    results = {
        "MAE_average": float(average_mae),
        "RMSE_average": float(average_rmse),
        "sMAPE_average": float(average_smape),
    }

    return results, per_customer


def calculate_MAE(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    id_col: str,
) -> tuple[float, pd.Series]:
    """Calculates MAE Metric

    Parameters
    ----------
    df : pd.DataFrame
        df with the predicted column
    pred_col : str
        name of column where predicted value is stored
    actual_col : str
        name of column where original value is stored
    id_col : str
        name of column where id is stored

    Returns
    -------
    tuple[float, pd.Series]
       average MAE for different ids, MAE per id
    """

    df_mae = df.copy()
    df_mae["error"] = np.abs(df_mae[actual_col] - df_mae[pred_col])

    mae_per_id = df_mae.groupby(id_col)["error"].mean().rename("MAE")

    average_mae = mae_per_id.mean()

    return average_mae, mae_per_id


def calculate_RMSE(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    id_col: str,
) -> tuple[float, pd.Series]:
    """Calculates RMSE Metric

    Parameters
    ----------
    df : pd.DataFrame
        df with the predicted column
    pred_col : str
        name of column where predicted value is stored
    actual_col : str
        name of column where original value is stored
    id_col : str
        name of column where id is stored

    Returns
    -------
    tuple[float, pd.Series]
       average RMSE for different ids, RMSE per id
    """
    df_RMSE = df.copy()
    df_RMSE["error"] = np.power((df_RMSE[actual_col] - df_RMSE[pred_col]), 2)

    rmse_per_id = (
        df_RMSE.groupby(id_col)["error"].mean().rename("RMSE").pipe(np.sqrt)
    )

    average_rmse = rmse_per_id.mean()

    return average_rmse, rmse_per_id  # type: ignore


def calculate_sMAPE(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    id_col: str,
) -> tuple[float, pd.Series]:
    """Calculates sMAPE metric

    Calculates sMAPE by id and average between customers
    Parameters
    ----------
    df : pd.DataFrame
        df with the predicted column
    pred_col : str
        name of column where predicted value is stored
    actual_col : str
        name of column where original value is stored
    id_col : str
        name of column where id is stored

    Returns
    -------
    tuple[float, pd.Series]
        average sMAPE for different ids, sMAPE per id
    """
    df_smape = df.copy()

    denom = (np.abs(df_smape[actual_col]) + np.abs(df_smape[pred_col])) / 2

    df_smape["error"] = np.where(
        denom == 0,
        0.0,
        np.abs(df_smape[actual_col] - df_smape[pred_col]) / denom,
    )

    smape_per_id = df_smape.groupby(id_col)["error"].mean().rename("sMAPE")

    average_smape = smape_per_id.mean()

    return average_smape, smape_per_id
