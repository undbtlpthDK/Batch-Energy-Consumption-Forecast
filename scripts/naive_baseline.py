import mlflow

from energycast.models import baseline, model_utils
from energycast.utils import data_utils, model_artifacts


def main():
    conf = model_utils.load_model_config("naive_24")
    df_train = data_utils.load_parquet("PROCESSED", conf.train_set)
    df_test = data_utils.load_parquet("PROCESSED", conf.test_set)

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
            }
        )
        results, per_customer, forecasts = (
            baseline.run_seasonal_naive_baseline(
                df_train,
                df_test,
                target_col="energy_kwh",
            )
        )

        mlflow.log_metrics(results)

        run_dir = model_artifacts.make_run_dir(
            category=conf.artifact_category,  # type: ignore
            model_name=conf.artifact_name,  # type: ignore
        )

        model_artifacts.save_run_artifacts(  # type: ignore
            run_dir=run_dir,
            metrics=results,
            per_customer_df=per_customer,
            forecasts_df=forecasts,
        )

        mlflow.log_artifacts(str(run_dir))
        mlflow.log_params(model_artifacts.conf_to_params(conf))
        mlflow.end_run()
    print(results)


if __name__ == "__main__":
    main()
