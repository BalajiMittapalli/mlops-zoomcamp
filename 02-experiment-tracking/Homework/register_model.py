import os
import pickle
import click
import mlflow
import numpy as np

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

HPO_EXPERIMENT_NAME = "random-forest-hyperopt"
EXPERIMENT_NAME = "random-forest-best-models"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)


def load_pickle(filename):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


def train_and_log_model(data_path, params):
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    X_test, y_test = load_pickle(os.path.join(data_path, "test.pkl"))


    n_estimators = min(int(params.get("n_estimators", 10)), 10)
    max_depth = min(int(params.get("max_depth", 10)), 10)
    min_samples_split = int(params.get("min_samples_split", 2))
    min_samples_leaf = int(params.get("min_samples_leaf", 1))
    random_state = int(params.get("random_state", 42))

    with mlflow.start_run():
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        rf.fit(X_train, y_train)

        val_rmse = np.sqrt(mean_squared_error(y_val, rf.predict(X_val)))

        mlflow.log_metric("val_rmse", val_rmse)

        test_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
        mlflow.log_metric("test_rmse", test_rmse)

        
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "random_state": random_state
        })

        mlflow.sklearn.log_model(rf, artifact_path="model")


@click.command()
@click.option(
    "--data_path",
    default="../output",  
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--top_n",
    default=5,
    type=int,
    help="Number of top models to evaluate before promoting the best one"
)
def run_register_model(data_path: str, top_n: int):
    client = MlflowClient()

    
    hpo_experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    hpo_runs = client.search_runs(
        experiment_ids=hpo_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"]
    )

    for run in hpo_runs:
        train_and_log_model(data_path=data_path, params=run.data.params)

    best_model_experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=best_model_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"

    model_name = "random-forest-regressor"
    mlflow.register_model(model_uri=model_uri, name=model_name)

    print(f"Registered model: {model_name}")
    print(f"Best run ID: {run_id}")
    print(f"Test RMSE of best model: {best_run.data.metrics['test_rmse']:.3f}")


if __name__ == '__main__':
    run_register_model()
