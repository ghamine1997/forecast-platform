"""Stage 8 - Model registry: quality gate, refit the winner on all data, version it."""
import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import inspect, text

from src.config import get_engine
from src.models import fit_lightgbm, fit_prophet, load_data, residual_quantiles
from src.storage import put_object, save_pickle, save_prophet_models

log = logging.getLogger(__name__)

MAX_SMAPE = float(os.getenv("MAX_SMAPE", "10"))   # quality gate, in percent
REGISTRY_PREFIX = "registry/store-sales"


def query(engine, sql, **params):
    return pd.read_sql(text(sql), engine, params=params)


def dataset_version(sales):
    """Fingerprint of the exact training data: changes if any value changes."""
    hashed = pd.util.hash_pandas_object(sales[["date", "store", "sales"]], index=False).values
    return hashlib.md5(hashed.tobytes()).hexdigest()[:12]


def next_version(engine):
    if not inspect(engine).has_table("model_registry"):
        return 1
    return int(query(engine, "SELECT COALESCE(MAX(version), 0) + 1 AS v FROM model_registry")["v"][0])


def run(run_id=None):
    engine = get_engine()
    if run_id is None:
        run_id = query(engine, "SELECT MAX(run_id) AS r FROM evaluation_results")["r"][0]

    ev = query(engine, """SELECT * FROM evaluation_results
                          WHERE run_id = :r AND is_winner
                          ORDER BY evaluated_at DESC LIMIT 1""", r=run_id)
    if ev.empty:
        raise ValueError(f"No evaluation results for run {run_id}")
    winner = ev.iloc[0]
    name = winner["model_name"]

    # Quality gate
    if winner["smape"] > MAX_SMAPE:
        raise ValueError(f"Quality gate failed: {name} test SMAPE {winner['smape']:.2f}% "
                         f"> {MAX_SMAPE}%. Model NOT registered.")

    params = json.loads(query(engine, """SELECT params FROM training_runs
                                         WHERE run_id = :r AND model_name = :m""",
                              r=run_id, m=name)["params"][0])
    cv_smape = float(query(engine, """SELECT AVG(smape) AS s FROM cv_fold_metrics
                                      WHERE run_id = :r AND model_name = :m""",
                           r=run_id, m=name)["s"][0])

    # Refit the winner on ALL data (including the test period)
    sales, features = load_data(engine)
    version = next_version(engine)
    prefix = f"{REGISTRY_PREFIX}/v{version}"
    quantiles = None

    if name == "prophet":
        models = {int(s): fit_prophet(g, params) for s, g in sales.groupby("store")}
        artifact_key = f"{prefix}/model.json"
        save_prophet_models(models, artifact_key)
    else:
        model = fit_lightgbm(features, params)
        artifact_key = f"{prefix}/model.pkl"
        save_pickle(model, artifact_key)
        cv = query(engine, """SELECT sales, yhat, horizon_day FROM cv_predictions
                              WHERE run_id = :r AND model_name = 'lightgbm'""", r=run_id)
        quantiles = residual_quantiles(cv)

    entry = {
        "version": version,
        "model_name": name,
        "stage": "production",
        "registered_at": datetime.now(timezone.utc),
        "run_id": run_id,
        "artifact_key": artifact_key,
        "metadata_key": f"{prefix}/metadata.json",
        "params": json.dumps(params),
        "train_start": sales["date"].min(),
        "train_end": sales["date"].max(),
        "n_rows": int(len(sales)),
        "dataset_version": dataset_version(sales),
        "cv_smape": cv_smape,
        **{f"test_{m}": float(winner[m]) for m in ("smape", "rmse", "mae", "mape", "r2", "coverage_80")},
        "interval_quantiles": json.dumps(quantiles) if quantiles else None,
    }

    # Human-readable copy of the metadata next to the model file
    metadata = {k: (str(v) if isinstance(v, (pd.Timestamp, datetime)) else v) for k, v in entry.items()}
    put_object(entry["metadata_key"], json.dumps(metadata, indent=2).encode("utf-8"))

    # Archive the old production model and register the new one, in ONE transaction
    with engine.begin() as conn:
        if version > 1:
            conn.execute(text("UPDATE model_registry SET stage = 'archived' WHERE stage = 'production'"))
        pd.DataFrame([entry]).to_sql("model_registry", conn, if_exists="append", index=False)

    summary = {"version": version, "model_name": name, "test_smape": round(float(winner["smape"]), 3),
               "dataset_version": entry["dataset_version"], "artifact_key": artifact_key}
    log.info("Registered: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run(sys.argv[1] if len(sys.argv) > 1 else None))