"""Stage 7 - Evaluation on the untouched test set, residual analysis, model selection."""
import logging
import sys
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from src.config import get_engine
from src.metrics import all_metrics, smape
from src.models import (add_empirical_intervals, forecast_lightgbm, forecast_prophet,
                        last_train_date, load_data, residual_quantiles)
from src.storage import load_pickle, load_prophet_models

log = logging.getLogger(__name__)


def query(engine, sql, **params):
    return pd.read_sql(text(sql), engine, params=params)


def latest_run_id(engine):
    return query(engine, "SELECT MAX(run_id) AS run_id FROM training_runs")["run_id"][0]


def predict_test(run_id, engine, sales):
    runs = query(engine, "SELECT model_name, artifact_key FROM training_runs WHERE run_id = :r",
                 r=run_id)
    keys = dict(zip(runs["model_name"], runs["artifact_key"]))
    train_end = last_train_date(sales)
    test = sales[sales["date"] > train_end]
    horizon = test["date"].nunique()
    frames = []

    for store, model in load_prophet_models(keys["prophet"]).items():
        fc = forecast_prophet(model, horizon)
        fc["store"], fc["model_name"] = store, "prophet"
        frames.append(fc)

    lgbm = load_pickle(keys["lightgbm"])
    fc = forecast_lightgbm(lgbm, sales[sales["date"] <= train_end], horizon)
    fc["horizon_day"] = (fc["date"] - train_end).dt.days
    cv = query(engine, """SELECT sales, yhat, horizon_day FROM cv_predictions
                          WHERE run_id = :r AND model_name = 'lightgbm'""", r=run_id)
    fc = add_empirical_intervals(fc, residual_quantiles(cv))
    fc["model_name"] = "lightgbm"
    frames.append(fc)

    cols = ["model_name", "date", "store", "yhat", "yhat_lower", "yhat_upper"]
    preds = pd.concat([f[cols] for f in frames], ignore_index=True)
    preds = preds.merge(test, on=["date", "store"])
    preds["horizon_day"] = (preds["date"] - train_end).dt.days
    return preds, test


def residual_analysis(g):
    g = g.sort_values(["store", "date"])
    resid = g["sales"] - g["yhat"]
    autocorr = resid.groupby(g["store"]).apply(lambda r: r.autocorr(lag=1)).mean()
    return {
        "bias_pct": float(resid.mean() / g["sales"].mean() * 100),   # > 0 means under-forecasting
        "resid_std": float(resid.std()),
        "resid_autocorr_1": float(autocorr),
    }


def interval_analysis(g):
    inside = (g["sales"] >= g["yhat_lower"]) & (g["sales"] <= g["yhat_upper"])
    width = (g["yhat_upper"] - g["yhat_lower"]) / g["yhat"]
    return {"coverage_80": float(inside.mean() * 100),
            "interval_width_pct": float(width.mean() * 100)}


def run(run_id=None):
    engine = get_engine()
    run_id = run_id or latest_run_id(engine)
    sales, _ = load_data(engine)
    preds, test = predict_test(run_id, engine, sales)
    preds["run_id"] = run_id

    rows = []
    for name, g in preds.groupby("model_name"):
        rows.append({
            "run_id": run_id, "model_name": name,
            "evaluated_at": datetime.now(timezone.utc),
            "test_start": test["date"].min(), "test_end": test["date"].max(),
            **all_metrics(g["sales"], g["yhat"]),
            **residual_analysis(g),
            **interval_analysis(g),
        })
    results = pd.DataFrame(rows)
    results["is_winner"] = results["smape"] == results["smape"].min()
    winner = results.loc[results["is_winner"], "model_name"].iloc[0]

    per_store = (preds.groupby(["model_name", "store"])
                 .apply(lambda g: smape(g["sales"], g["yhat"]), include_groups=False)
                 .unstack(0).round(2))
    log.info("Test SMAPE by store:\n%s", per_store.to_string())

    preds.to_sql("test_predictions", engine, if_exists="append", index=False)
    results.to_sql("evaluation_results", engine, if_exists="append", index=False)

    keys = ["smape", "rmse", "mae", "mape", "r2", "bias_pct", "resid_autocorr_1",
            "coverage_80", "interval_width_pct"]
    summary = {"run_id": run_id, "winner": winner}
    for r in rows:
        summary[r["model_name"]] = {k: round(float(r[k]), 3) for k in keys}
    log.info("Evaluation summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run(sys.argv[1] if len(sys.argv) > 1 else None))