from typing import List

import lightgbm as lgb
import pandas as pd


def per_customer_data(
    train: pd.DataFrame,
    dev: pd.DataFrame,
    test: pd.DataFrame,
    features: List[str],
    target: str,
):
    """Splits data by customer

    Parameters
    ----------
    train : pd.DataFrame
        train set
    dev : pd.DataFrame
        dev set
    test : pd.DataFrame
        test set
    features : List[str]
        feature list
    target : str
        target list

    Returns
    -------
    dict
        returns 2 layer dictionary, first key - customer_id
        second key - data sets
    """
    customer_data = {
        customer: {
            "X_train": train.loc[train["object_id"] == customer, features],
            "y_train": train.loc[train["object_id"] == customer, target],
            "X_dev": dev.loc[dev["object_id"] == customer, features],
            "y_dev": dev.loc[dev["object_id"] == customer, target],
            "X_test": test.loc[test["object_id"] == customer, features],
            "y_test": test.loc[test["object_id"] == customer, target],
            "y_pred": 0,
        }
        for customer in train["object_id"].unique()
    }
    return customer_data


def train_models_per_customer(customer_data: dict, config: dict) -> dict:
    """Creates dictionary that stores trained model per customer

    Parameters
    ----------
    customer_data :
        customer datasets dictionary
    config : _type_
        model parameters from configuration yaml

    Returns
    -------
    _type_
        dictionary of models where key is customer id
    """
    customer_models = {}
    for customer, data in customer_data.items():
        model = lgb.LGBMRegressor(
            objective=config["objective"],
            boosting_type=config["boosting_type"],
            metric=config["metric"],
            random_state=config["random_state"],
            learning_rate=config["learning_rate"],
            num_leaves=config["num_leaves"],
            max_depth=config["max_depth"],
            min_data_in_leaf=config["min_data_in_leaf"],
            n_estimators=config["n_estimators"],
            lambda_l1=config["lambda_l1"],
            lambda_l2=config["lambda_l2"],
        )
        model.fit(
            data["X_train"],
            data["y_train"],
            eval_set=[(data["X_dev"], data["y_dev"])],
            eval_metric="mae",
        )
        customer_models[customer] = model

    return customer_models


def predict_per_customer(
    customer_models: dict, customer_data: dict
) -> pd.DataFrame:
    """Calculates prediction for customer test

    Parameters
    ----------
    customer_models : dict
        customer pretrained models
    customer_data : dict
        customer datasets

    Returns
    -------
    pd.DataFrame
        returns DataFrame with predictions and original values
    """
    for customer, model in customer_models.items():
        customer_data[customer]["y_pred"] = model.predict(
            customer_data[customer]["X_test"]
        )
        prediction = [
            pd.DataFrame(
                {
                    "object_id": customer,
                    "y_true": customer_data[customer]["y_test"]
                    .to_numpy()
                    .reshape(-1),
                    "y_pred": customer_data[customer]["y_pred"],
                }
            )
            for customer in customer_data.keys()
        ]

    return pd.concat(prediction)
