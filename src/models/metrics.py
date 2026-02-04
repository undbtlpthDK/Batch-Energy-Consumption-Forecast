import numpy as np
import pandas as pd


def evaluate_forecasts(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    id_col: str,
):

    mae_score, mae_per_id = calculate_MAE(
        df, pred_col, actual_col, id_col
    )

    mean_rmse_per_id, global_rmse, rmse_per_id = calculate_RMSE(
        df, pred_col, actual_col, id_col
    )

    global_smape, smape_per_id = calculate_sMAPE(
        df, pred_col, actual_col, id_col
    )

    per_customer = (
        pd.concat(
            [mae_per_id, rmse_per_id, smape_per_id],
            axis=1,
        )
        .reset_index()
    )

    results = {
        "MAE_macro": mae_score,
        "RMSE_macro": mean_rmse_per_id,
        "RMSE_global": global_rmse,
        "sMAPE_macro": global_smape,
    }

    return results, per_customer


def calculate_MAE(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    id_col: str,
):
    df_mae = df.copy()
    df_mae["error"] = np.abs(df_mae[actual_col] - df_mae[pred_col])

    mae_per_id = (
        df_mae
        .groupby(id_col)["error"]
        .mean()
        .rename("MAE")
    )

    MAE_score = mae_per_id.mean()

    return MAE_score, mae_per_id


def calculate_RMSE(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    id_col: str,
):
    df_RMSE = df.copy()
    df_RMSE["error"] = np.power((df_RMSE[actual_col] - df_RMSE[pred_col]), 2)

    global_rmse = np.sqrt(df_RMSE["error"].mean())

    rmse_per_id = (
        df_RMSE
        .groupby(id_col)["error"]
        .mean()
        .rename("RMSE")
        .pipe(np.sqrt)
    )

    mean_rmse_score_per_id = rmse_per_id.mean()

    return mean_rmse_score_per_id, global_rmse, rmse_per_id


def calculate_sMAPE(
    df: pd.DataFrame,
    pred_col: str,
    actual_col: str,
    id_col: str,
):
    df_smape = df.copy()

    denom = (np.abs(df_smape[actual_col]) + np.abs(df_smape[pred_col])) / 2

    df_smape["error"] = np.where(
        denom == 0,
        0.0,
        np.abs(df_smape[actual_col] - df_smape[pred_col]) / denom,
    )

    smape_per_id = (
        df_smape
        .groupby(id_col)["error"]
        .mean()
        .rename("sMAPE")
    )

    global_smape = smape_per_id.mean()

    return global_smape, smape_per_id
