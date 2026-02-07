from pathlib import Path

import lightgbm as lgb
import mlflow
import pandas as pd

import energycast.utils.model_artifacts as model_artifacts
import src.energycast.evaluation.metrics as metrics

ROOT = Path.cwd()
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"

config = {
    "model_type": "lgbm",
    "strategy": "per_customer",
    "horizon": 24,
    "target_col": "energy_kwh",
    "evaluation_protocol": "fixed_train_dev_test",
    "lgbm_params": {
        "objective": "regression",
        "boosting_type": "goss",
        "metric": "mae",
        "learning_rate": 0.06,
        "num_leaves": 8,
        "max_depth": 10,
        "min_data_in_leaf": 100,
        "n_estimators": 100,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "random_state": 42,
    },
}


# Should be moved after
def per_customer_data():
    train = pd.read_parquet(PROCESSED / "lgbm_multi_horizon_24_train.parquet")
    dev = pd.read_parquet(PROCESSED / "lgbm_multi_horizon_24_dev.parquet")
    test = pd.read_parquet(PROCESSED / "lgbm_multi_horizon_24_test.parquet")

    train = pd.get_dummies(train, columns=["horizon_index"], drop_first=True)
    dev = pd.get_dummies(dev, columns=["horizon_index"], drop_first=True)
    test = pd.get_dummies(test, columns=["horizon_index"], drop_first=True)

    bool_cols = train.select_dtypes(include="bool").columns
    train[bool_cols] = train[bool_cols].astype("int8")
    dev[bool_cols] = dev[bool_cols].astype("int8")
    test[bool_cols] = test[bool_cols].astype("int8")

    train["y_day_ago"] = train["energy_kwh"].shift(24)
    train["y_week_ago"] = train["energy_kwh"].shift(168)
    dev["y_day_ago"] = dev["energy_kwh"].shift(24)
    dev["y_week_ago"] = dev["energy_kwh"].shift(168)
    test["y_day_ago"] = test["energy_kwh"].shift(24)
    test["y_week_ago"] = test["energy_kwh"].shift(168)
    train = train.dropna()
    dev = dev.dropna()
    test = test.dropna()

    features = [
        "day",
        "hour_sin",
        "hour_cos",
        "week_sin",
        "week_cos",
        "weekday",
        "is_weekend",
        "month_sin",
        "month_cos",
        "is_holiday",
        "temperature_2m",
        "is_day",
        "relative_humidity_2m",
        "apparent_temperature",
        "precipitation",
        "lag_24",
        "lag_168",
        "lag_1",
        "rolling_mean_24",
        "rolling_mean_168",
        "rolling_std_168",
        "horizon_index_1",
        "horizon_index_2",
        "horizon_index_3",
        "horizon_index_4",
        "horizon_index_5",
        "horizon_index_6",
        "horizon_index_7",
        "horizon_index_8",
        "horizon_index_9",
        "horizon_index_10",
        "horizon_index_11",
        "horizon_index_12",
        "horizon_index_13",
        "horizon_index_14",
        "horizon_index_15",
        "horizon_index_16",
        "horizon_index_17",
        "horizon_index_18",
        "horizon_index_19",
        "horizon_index_20",
        "horizon_index_21",
        "horizon_index_22",
        "horizon_index_23",
        "y_day_ago",
        "y_week_ago",
    ]

    target = ["energy_kwh"]

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


def train_models_per_customer(customer_data):
    customer_models = {}
    for customer, data in customer_data.items():
        model = lgb.LGBMRegressor(
            objective="regression",
            boosting_type="goss",
            metric="mae",
            random_state=42,
            learning_rate=0.06,
            num_leaves=8,
            max_depth=10,
            min_data_in_leaf=100,
            n_estimators=100,
            lambda_l1=0.5,
            lambda_l2=0.5,
        )
        model.fit(
            data["X_train"],
            data["y_train"],
            eval_set=[(data["X_dev"], data["y_dev"])],
            eval_metric="mae",
        )
        customer_models[customer] = model

    return customer_models


def predict_per_customer(customer_models, customer_data):
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


def main():

    data_tracking = {
        "train_data": str(PROCESSED / "lgbm_multi_horizon_24_train.parquet"),
        "train_dvc": str(
            PROCESSED / f"{"lgbm_multi_horizon_24_train.parquet"}.dvc"
        ),
        "dev_data": str(PROCESSED / "lgbm_multi_horizon_24_dev.parquet"),
        "dev_dvc": str(
            PROCESSED / f"{"lgbm_multi_horizon_24_dev.parquet"}.dvc"
        ),
        "test_data": str(PROCESSED / "lgbm_multi_horizon_24_test.parquet"),
        "test_dvc": str(
            PROCESSED / f"{"lgbm_multi_horizon_24_test.parquet"}.dvc"
        ),
    }

    with mlflow.start_run(run_name="LightGBM"):
        mlflow.set_experiment("EnergyCast-Per-Customer-Models")

        mlflow.log_params(
            {
                "train_data": data_tracking["train_data"],
                "dev_data": data_tracking["dev_data"],
                "test_data": data_tracking["test_data"],
            }
        )
        config["data_tracking"] = data_tracking
        customer_data = per_customer_data()
        customer_models = train_models_per_customer(customer_data)
        prediction = predict_per_customer(customer_models, customer_data)
        results, per_customer = metrics.evaluate_forecasts(
            prediction, "y_pred", "y_true", "object_id"
        )

        mlflow.log_metrics(results)

        run_dir = model_artifacts.make_run_dir(
            artifacts_root=ARTIFACTS,
            category="model",
            model_name="lgbm_per_customer",
        )

        model_artifacts.save_run_artifacts(
            run_dir=run_dir,
            metrics=results,
            per_customer_df=per_customer,
            forecasts_df=prediction,
            config=config,
        )

        mlflow.log_artifacts(str(run_dir))
        mlflow.end_run()


if __name__ == "__main__":
    main()
