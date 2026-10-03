from datetime import datetime

from airflow.sdk import dag, task


@dag(
    dag_id="forecasting_pipeline",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1},
    tags=["forecast"],
)
def forecasting_pipeline():

    @task
    def ingest_data():
        from src.ingest import run
        return run()

    @task
    def clean_data():
        from src.clean import run
        return run()

    @task
    def engineer_features():
        from src.features import run
        return run()

    @task
    def tune_hyperparameters():
        from src.tune import run
        return run()

    @task
    def train_models():
        from src.train import run
        return run()

    @task
    def cross_validate(training: dict):
        from src.cross_validate import run
        return run(training["run_id"])

    @task
    def evaluate_models(training: dict):
        from src.evaluate import run
        return run(training["run_id"])

    @task
    def register_model(evaluation: dict):
        from src.register import run
        return run(evaluation["run_id"])

    trained = train_models()
    ingest_data() >> clean_data() >> engineer_features() >> tune_hyperparameters() >> trained

    validated = cross_validate(trained)
    evaluated = evaluate_models(trained)
    validated >> evaluated
    register_model(evaluated)


forecasting_pipeline()