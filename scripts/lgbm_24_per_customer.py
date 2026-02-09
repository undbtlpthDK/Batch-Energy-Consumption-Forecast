import mlflow

from energycast.evaluation import metrics
from energycast.models import lgbm, model_utils
from energycast.utils import data_utils, model_artifacts


def main():

    conf = model_utils.load_model_config("lgbm_per_customer_24")
    train = data_utils.load_parquet("PROCESSED", conf.train_set)
    dev = data_utils.load_parquet("PROCESSED", conf.dev_set)
    test = data_utils.load_parquet("PROCESSED", conf.test_set)

    data_tracking = {
        "train_data": conf.train_set,
        "dev_data": conf.dev_set,
        "test_data": conf.test_set,
    }
    mlflow.set_experiment(conf.mlflow_experiment)
    with mlflow.start_run(run_name=conf.mlflow_run_name):

        mlflow.log_params(
            {
                "train_data": data_tracking["train_data"],
                "dev_data": data_tracking["dev_data"],
                "test_data": data_tracking["test_data"],
            }
        )
        # config["data_tracking"] = data_tracking
        customer_data = lgbm.per_customer_data(
            train=train,
            dev=dev,
            test=test,
            features=conf.features,
            target=conf.target_col,
        )
        customer_models = lgbm.train_models_per_customer(
            customer_data, config=conf.model_params
        )
        prediction = lgbm.predict_per_customer(customer_models, customer_data)

        results, per_customer = metrics.evaluate_forecasts(
            prediction, "y_pred", "y_true", "object_id"
        )

        mlflow.log_metrics(results)

        run_dir = model_artifacts.make_run_dir(
            category=conf.artifact_category,
            model_name=conf.artifact_name,
        )

        model_artifacts.save_run_artifacts(
            run_dir=run_dir,
            metrics=results,
            per_customer_df=per_customer,
            forecasts_df=prediction,
        )

        mlflow.log_artifacts(str(run_dir))
        mlflow.log_params(model_artifacts.conf_to_params(conf))

        mlflow.end_run()


if __name__ == "__main__":
    main()
