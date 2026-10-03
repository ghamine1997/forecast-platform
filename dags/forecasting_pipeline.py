"""
### Weekly store sales forecasting pipeline

Ingests sales from MinIO, cleans and engineers features, tunes Prophet and LightGBM
with Optuna, trains, cross-validates and evaluates both, registers the winner
(quality gate: test SMAPE < 10%), forecasts 365 days and refreshes the dashboard.

Runbook: docs/RUNBOOK.md in the project repository.
"""
from datetime import datetime, timedelta

from airflow.sdk import dag, task


@dag(
    dag_id="forecasting_pipeline",
    schedule="0 2 * * 1",          # every Monday at 02:00
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "execution_timeout": timedelta(minutes=30),
    },
    tags=["forecast"],
    doc_md=__doc__,
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

    @task
    def generate_forecast(registration: dict):
        from src.forecast import run
        return run()

    @task
    def update_dashboard(forecast: dict):
        from src.publish import run
        return run()

    trained = train_models()
    ingest_data() >> clean_data() >> engineer_features() >> tune_hyperparameters() >> trained

    validated = cross_validate(trained)
    evaluated = evaluate_models(trained)
    validated >> evaluated

    registered = register_model(evaluated)
    update_dashboard(generate_forecast(registered))


forecasting_pipeline()