"""Stage 5 - Training: fit the tuned models on all training data, store them in MinIO."""
import json
import logging
import time
from datetime import datetime, timezone

import pandas as pd

from src.config import get_engine
from src.models import fit_lightgbm, fit_prophet, last_train_date, load_best_params, load_data
from src.storage import save_pickle, save_prophet_models

log = logging.getLogger(__name__)


def _record(run_id, name, params, key, data, started):
    return {
        "run_id": run_id,
        "model_name": name,
        "trained_at": datetime.now(timezone.utc),
        "train_start": data["date"].min(),
        "train_end": data["date"].max(),
        "n_rows": int(len(data)),
        "params": json.dumps(params),
        "artifact_key": key,
        "duration_s": round(time.time() - started, 1),
    }


def run():
    engine = get_engine()
    sales, features = load_data(engine)
    params = load_best_params(engine)

    train_end = last_train_date(sales)
    train_sales = sales[sales["date"] <= train_end]
    train_features = features[features["date"] <= train_end].dropna()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    records = []

    started = time.time()
    prophet_models = {
        int(store): fit_prophet(group, params["prophet"])
        for store, group in train_sales.groupby("store")
    }
    key = f"candidates/{run_id}/prophet.json"
    save_prophet_models(prophet_models, key)
    records.append(_record(run_id, "prophet", params["prophet"], key, train_sales, started))
    log.info("Prophet: %d store models saved to %s", len(prophet_models), key)

    started = time.time()
    lgbm = fit_lightgbm(train_features, params["lightgbm"])
    key = f"candidates/{run_id}/lightgbm.pkl"
    save_pickle(lgbm, key)
    records.append(_record(run_id, "lightgbm", params["lightgbm"], key, train_features, started))
    log.info("LightGBM: global model saved to %s", key)

    pd.DataFrame(records).to_sql("training_runs", engine, if_exists="append", index=False)

    summary = {
        "run_id": run_id,
        "train_end": str(train_end.date()),
        "models": {r["model_name"]: r["artifact_key"] for r in records},
    }
    log.info("Training summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run())