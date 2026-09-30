import os
from datetime import datetime

from airflow.sdk import dag, task


@dag(
    dag_id="hello_forecast",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["test"],
)
def hello_forecast():

    @task
    def check_postgres():
        import psycopg2

        conn = psycopg2.connect(
            host=os.environ["POSTGRES_HOST"],
            dbname=os.environ["POSTGRES_DB"],
            user=os.environ["POSTGRES_USER"],
            password=os.environ["POSTGRES_PASSWORD"],
        )
        with conn.cursor() as cur:
            cur.execute("SELECT version();")
            print("Connected to:", cur.fetchone()[0])
        conn.close()

    @task
    def check_minio():
        import boto3

        s3 = boto3.client(
            "s3",
            endpoint_url=os.environ["MINIO_ENDPOINT"],
            aws_access_key_id=os.environ["MINIO_USER"],
            aws_secret_access_key=os.environ["MINIO_PASSWORD"],
        )
        objects = s3.list_objects_v2(Bucket="raw").get("Contents", [])
        for obj in objects:
            print(f"Found {obj['Key']} ({obj['Size'] / 1e6:.1f} MB)")

    check_postgres() >> check_minio()


hello_forecast()