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

    ingest_data() >> clean_data()


forecasting_pipeline()