import os

import boto3
from sqlalchemy import create_engine

RAW_BUCKET = "raw"
RAW_KEY = "train.csv"
MODELS_BUCKET = "models"


def get_engine():
    user = os.getenv("POSTGRES_USER", "forecast")
    password = os.getenv("POSTGRES_PASSWORD", "forecast123")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "30432")
    db = os.getenv("POSTGRES_DB", "forecast")
    return create_engine(f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}")


def get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:30900"),
        aws_access_key_id=os.getenv("MINIO_USER", "minioadmin"),
        aws_secret_access_key=os.getenv("MINIO_PASSWORD", "minioadmin123"),
        region_name="us-east-1",
    )