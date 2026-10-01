"""Save and load model artifacts in the MinIO `models` bucket."""
import json
import pickle

from src.config import MODELS_BUCKET, get_s3_client


def put_object(key, data):
    get_s3_client().put_object(Bucket=MODELS_BUCKET, Key=key, Body=data)


def get_object(key):
    return get_s3_client().get_object(Bucket=MODELS_BUCKET, Key=key)["Body"].read()


def save_prophet_models(models, key):
    """Prophet's official JSON format: safer across versions than pickle."""
    from prophet.serialize import model_to_json

    payload = {str(store): model_to_json(model) for store, model in models.items()}
    put_object(key, json.dumps(payload).encode("utf-8"))


def load_prophet_models(key):
    from prophet.serialize import model_from_json

    payload = json.loads(get_object(key))
    return {int(store): model_from_json(text) for store, text in payload.items()}


def save_pickle(obj, key):
    put_object(key, pickle.dumps(obj))


def load_pickle(key):
    return pickle.loads(get_object(key))