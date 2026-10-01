"""Stage 4 - Bayesian hyperparameter optimization with Optuna (TPE sampler)."""
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import optuna
import pandas as pd

from src.config import get_engine
from src.metrics import smape
from src.models import (cv_predict_lightgbm, cv_predict_prophet, last_train_date,
                        load_data, make_folds)

log = logging.getLogger(__name__)

N_TRIALS = int(os.getenv("N_TRIALS", "20"))
TUNING_FOLDS = 2
TIMEOUT_S = 900   # safety limit per model (15 minutes)


def pick_tuning_stores(sales):
    """Lowest, median and highest-volume stores: representative, 3x cheaper than all 10."""
    ranked = sales.groupby("store")["sales"].mean().sort_values().index.tolist()
    return [int(ranked[0]), int(ranked[len(ranked) // 2]), int(ranked[-1])]


def prophet_space(trial):
    return {
        "changepoint_prior_scale": trial.suggest_float("changepoint_prior_scale", 0.001, 0.5, log=True),
        "seasonality_prior_scale": trial.suggest_float("seasonality_prior_scale", 0.01, 10, log=True),
        "holidays_prior_scale": trial.suggest_float("holidays_prior_scale", 0.01, 10, log=True),
        "seasonality_mode": trial.suggest_categorical("seasonality_mode", ["additive", "multiplicative"]),
    }


def lightgbm_space(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "subsample_freq": 1,
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
    }


def log_trial(study, trial):
    log.info("%s trial %d: SMAPE %.3f (best so far %.3f)",
             study.study_name, trial.number, trial.value, study.best_value)


def run(n_trials=N_TRIALS):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    engine = get_engine()
    sales, features = load_data(engine)

    folds = make_folds(last_train_date(sales), TUNING_FOLDS)
    stores = pick_tuning_stores(sales)
    log.info("Tuning folds: %s", [(str(a.date()), str(b.date())) for a, b in folds])
    log.info("Tuning stores: %s", stores)

    candidates = [
        ("prophet", prophet_space,
         lambda p: cv_predict_prophet(p, sales, folds, stores)),
        ("lightgbm", lightgbm_space,
         lambda p: cv_predict_lightgbm(p, features, sales, folds, stores)),
    ]

    rows, summary = [], {}
    for name, space, evaluate in candidates:
        def objective(trial):
            preds = evaluate(space(trial))
            return smape(preds["sales"], preds["yhat"])

        started = time.time()
        study = optuna.create_study(direction="minimize", study_name=name,
                                    sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=n_trials, timeout=TIMEOUT_S, callbacks=[log_trial])

        best_params = space(optuna.trial.FixedTrial(study.best_params))
        duration = round(time.time() - started, 1)
        rows.append({
            "run_at": datetime.now(timezone.utc),
            "model_name": name,
            "best_smape": study.best_value,
            "best_params": json.dumps(best_params),
            "n_trials": len(study.trials),
            "duration_s": duration,
            "tuning_stores": json.dumps(stores),
        })
        summary[name] = {"best_smape": round(study.best_value, 3),
                         "trials": len(study.trials), "duration_s": duration}

    pd.DataFrame(rows).to_sql("hyperparameter_runs", engine, if_exists="append", index=False)
    log.info("Tuning summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run(int(sys.argv[1]) if len(sys.argv) > 1 else N_TRIALS))