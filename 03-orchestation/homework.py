from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import mlflow

# ─── Utility Functions ─────────────────────────────────────────────────────────

def read_and_count(filename, **kwargs):
    df = pd.read_parquet(filename)
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    df[['PULocationID','DOLocationID']] = df[['PULocationID','DOLocationID']].astype(str)
    count = len(df)
    print(f"Loaded records: {count}")           # Q3: raw count
    # push DataFrame via XCom
    kwargs['ti'].xcom_push(key='df_raw', value=df)
    kwargs['ti'].xcom_push(key='raw_count', value=count)

def prepare_and_count(**kwargs):
    df = kwargs['ti'].xcom_pull(key='df_raw', task_ids='read_data')
    df[['PULocationID','DOLocationID']] = df[['PULocationID','DOLocationID']].astype(str)
    count = df.shape[0]
    print(f"Prepared records: {count}")        # Q4: filtered count
    kwargs['ti'].xcom_push(key='df_prep', value=df)
    kwargs['ti'].xcom_push(key='prep_count', value=count)

def train_and_log(**kwargs):
    df = kwargs['ti'].xcom_pull(key='df_prep', task_ids='prepare_data')
    # vectorize
    dicts = df[['PULocationID','DOLocationID']].to_dict(orient='records')
    dv = DictVectorizer()
    X = dv.fit_transform(dicts)
    y = df['duration'].values

    # train
    lr = LinearRegression()
    lr.fit(X, y)

    # MLflow logging
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("nyc-taxi-experiment")
    run_date = kwargs['ts'][:10]
    with mlflow.start_run(run_name=run_date):
        mlflow.log_param("orchestrator", "Airflow")
        mlflow.log_param("orchestrator_version", "2.10.3")
        mlflow.log_metric("intercept", float(lr.intercept_))   # Q5

        # save vectorizer & model
        Path("models").mkdir(exist_ok=True)
        vec_path = f"models/vectorizer_{run_date}.pkl"
        mdl_path = f"models/model_{run_date}.pkl"
        with open(vec_path, "wb") as fv:
            pickle.dump(dv, fv)
        with open(mdl_path, "wb") as fm:
            pickle.dump(lr, fm)

        mlflow.log_artifact(vec_path, artifact_path="preprocessor")
        mlflow.log_artifact(mdl_path, artifact_path="model")

    run_id = mlflow.active_run().info.run_id
    print(f"Intercept: {lr.intercept_}")
    print(f"MLflow run_id: {run_id}")

# ─── DAG Definition ────────────────────────────────────────────────────────────

DEFAULT_ARGS = {
    'owner': 'you',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='nyc_taxi_training',
    default_args=DEFAULT_ARGS,
    description='Monthly pipeline: load, prepare, train & register Yellow taxi model',
    schedule='@monthly',                    # ← changed here
    start_date=datetime(2023, 3, 1),
    catchup=False,
    tags=['homework', 'mlflow', 'airflow'],
) as dag:

    read_data    = PythonOperator(task_id='read_data', python_callable=read_and_count,
                                  op_kwargs={'filename': '/workspaces/mlops-zoomcamp/data/yellow_tripdata_2023-03.parquet'})
    prepare_data = PythonOperator(task_id='prepare_data', python_callable=prepare_and_count)
    train_model  = PythonOperator(task_id='train_model', python_callable=train_and_log)

    read_data >> prepare_data >> train_model
