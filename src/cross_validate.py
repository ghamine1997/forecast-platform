"""Stage 6 - Rolling-origin cross-validation on all stores, with error by horizon."""
import logging
import sys
import time

import pandas as pd

from src.config import get_engine
from src.metrics import all_metrics, smape
from src.models import (cv_predict_lightgbm, cv_predict_prophet, last_train_date,
                        load_best_params, load_data, make_folds)

log = logging.getLogger(__name__)

N_FOLDS = 4
HORIZON_BINS = [0, 7, 30, 90]
HORIZON_LABELS = ["1-7 days", "8-30 days", "31-90 days"]


def latest_run_id(engine):
    return pd.read_sql("SELECT MAX(run_id) AS run_id FROM training_runs", engine)["run_id"][0]


def run(run_id=None):
    engine = get_engine()
    run_id = run_id or latest_run_id(engine)
    sales, features = load_data(engine)
    params = load_best_params(engine)

    folds = make_folds(last_train_date(sales), N_FOLDS)
    stores = sorted(int(s) for s in sales["store"].unique())
    log.info("CV folds: %s", [(str(a.date()), str(b.date())) for a, b in folds])

    candidates = [
        ("prophet", lambda: cv_predict_prophet(params["prophet"], sales, folds, stores)),
        ("lightgbm", lambda: cv_predict_lightgbm(params["lightgbm"], features, sales, folds, stores)),
    ]
    all_preds = []
    for name, predict in candidates:
        started = time.time()
        preds = predict()
        preds["model_name"] = name
        all_preds.append(preds)
        log.info("%s cross-validated in %.1f s", name, time.time() - started)

    preds = pd.concat(all_preds, ignore_index=True)
    fold_start = {k: start for k, (start, _) in enumerate(folds, 1)}
    preds["horizon_day"] = (preds["date"] - preds["fold"].map(fold_start)).dt.days + 1
    preds["run_id"] = run_id

    # Metrics per model and fold
    fold_rows = []
    for (name, fold), g in preds.groupby(["model_name", "fold"]):
        start, end = folds[fold - 1]
        fold_rows.append({"run_id": run_id, "model_name": name, "fold": int(fold),
                          "val_start": start, "val_end": end,
                          **all_metrics(g["sales"], g["yhat"])})
    fold_metrics = pd.DataFrame(fold_rows)

    # Error by forecast horizon
    preds["horizon_bucket"] = pd.cut(preds["horizon_day"], bins=HORIZON_BINS, labels=HORIZON_LABELS)
    by_horizon = (preds.groupby(["model_name", "horizon_bucket"], observed=True)
                  .apply(lambda g: smape(g["sales"], g["yhat"]), include_groups=False))

    # Save
    keep = ["run_id", "model_name", "fold", "date", "store", "sales", "yhat", "horizon_day"]
    preds[keep].to_sql("cv_predictions", engine, if_exists="append", index=False)
    fold_metrics.to_sql("cv_fold_metrics", engine, if_exists="append", index=False)

    summary = {"run_id": run_id}
    for name, g in fold_metrics.groupby("model_name"):
        summary[name] = {
            "smape_mean": round(float(g["smape"].mean()), 3),
            "smape_std": round(float(g["smape"].std()), 3),
            "rmse_mean": round(float(g["rmse"].mean()), 1),
            "mae_mean": round(float(g["mae"].mean()), 1),
            "mape_mean": round(float(g["mape"].mean()), 3),
            "r2_mean": round(float(g["r2"].mean()), 3),
            "smape_by_horizon": {k: round(float(v), 3) for k, v in by_horizon[name].items()},
        }
    log.info("Cross-validation summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run(sys.argv[1] if len(sys.argv) > 1 else None))