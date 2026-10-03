"""Shared modelling code: data splits, Prophet and LightGBM fit/forecast.
Used by tuning (Day 6), cross-validation and training (Days 7-8) and forecasting (Day 9)."""
import logging

import numpy as np
import pandas as pd
import json

from src.features import HOLIDAY_COUNTRY, make_features

HORIZON = 90        # days predicted ahead in each validation window
TEST_DAYS = 90      # final hold-out period, never used for tuning or training
HISTORY_DAYS = 60   # history kept during recursive forecasting (>= longest lag + window)

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)


# ---------- Data and splits ----------

def load_data(engine):
    sales = pd.read_sql("SELECT date, store, sales FROM sales_daily_store", engine,
                        parse_dates=["date"])
    features = pd.read_sql("SELECT * FROM features_store", engine, parse_dates=["date"])
    return sales, features


def last_train_date(sales):
    """Everything after this date is the test set."""
    return sales["date"].max() - pd.Timedelta(days=TEST_DAYS)


def make_folds(end_date, n_folds, horizon=HORIZON):
    """Expanding-window folds. Each fold validates on `horizon` consecutive days;
    the last fold ends on `end_date`. The model trains on everything before each window."""
    folds = []
    for i in range(n_folds, 0, -1):
        val_end = end_date - pd.Timedelta(days=(i - 1) * horizon)
        val_start = val_end - pd.Timedelta(days=horizon - 1)
        folds.append((val_start, val_end))
    return folds


# ---------- Prophet (one model per store) ----------

def fit_prophet(train, params, intervals=True):
    from prophet import Prophet

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        uncertainty_samples=1000 if intervals else 0,
        **params,
    )
    model.add_country_holidays(country_name=HOLIDAY_COUNTRY)
    model.fit(train.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]])
    return model


def forecast_prophet(model, horizon):
    future = model.make_future_dataframe(periods=horizon, include_history=False)
    fc = model.predict(future).rename(columns={"ds": "date"})
    cols = [c for c in ["date", "yhat", "yhat_lower", "yhat_upper", "trend"] if c in fc.columns]
    fc = fc[cols].copy()
    for c in ("yhat", "yhat_lower", "yhat_upper"):
        if c in fc.columns:
            fc[c] = fc[c].clip(lower=0)
    return fc


def cv_predict_prophet(params, sales, folds, stores):
    rows = []
    for k, (start, end) in enumerate(folds, 1):
        for store in stores:
            s = sales[sales["store"] == store]
            model = fit_prophet(s[s["date"] < start], params, intervals=False)
            fc = forecast_prophet(model, (end - start).days + 1)
            fc["store"], fc["fold"] = store, k
            rows.append(fc)
    return pd.concat(rows, ignore_index=True).merge(sales, on=["date", "store"])


# ---------- LightGBM (one global model for all stores) ----------

def fit_lightgbm(train_features, params):
    import lightgbm as lgb

    train = train_features.dropna()
    X = train.drop(columns=["date", "sales"])
    model = lgb.LGBMRegressor(objective="regression", n_jobs=2, verbose=-1,
                              random_state=42, **params)
    model.fit(X, train["sales"])
    return model


def forecast_lightgbm(model, history, horizon):
    """Recursive forecast: predict one day, feed it back as history, repeat."""
    hist = history[["date", "store", "sales"]].copy()
    hist = hist[hist["date"] > hist["date"].max() - pd.Timedelta(days=HISTORY_DAYS)]
    origin = hist["date"].max()
    stores = sorted(hist["store"].unique())
    results = []

    for step in range(1, horizon + 1):
        day = origin + pd.Timedelta(days=step)
        new_rows = pd.DataFrame({"date": day, "store": stores, "sales": np.nan})
        hist = pd.concat([hist, new_rows], ignore_index=True)

        today = make_features(hist).query("date == @day")
        yhat = np.clip(model.predict(today[model.feature_name_]), 0, None)

        predicted = dict(zip(today["store"], yhat))
        mask = hist["date"] == day
        hist.loc[mask, "sales"] = hist.loc[mask, "store"].map(predicted)
        results.append(pd.DataFrame({"date": day, "store": today["store"].values, "yhat": yhat}))

    return pd.concat(results, ignore_index=True)


def cv_predict_lightgbm(params, features, sales, folds, stores):
    rows = []
    for k, (start, end) in enumerate(folds, 1):
        model = fit_lightgbm(features[features["date"] < start], params)
        fc = forecast_lightgbm(model, sales[sales["date"] < start], (end - start).days + 1)
        fc["fold"] = k
        rows.append(fc[fc["store"].isin(stores)])
    return pd.concat(rows, ignore_index=True).merge(sales, on=["date", "store"])

# ---------- Tuned parameters ----------

def load_best_params(engine):
    """Most recent tuned parameters for each model, from the tuning stage."""
    df = pd.read_sql(
        """SELECT DISTINCT ON (model_name) model_name, best_params
           FROM hyperparameter_runs
           ORDER BY model_name, run_at DESC""",
        engine,
    )
    return {row.model_name: json.loads(row.best_params) for row in df.itertuples()}

# ---------- Prediction intervals for LightGBM (empirical, from CV errors) ----------

INTERVAL_BINS = [0, 7, 30, np.inf]
INTERVAL_LABELS = ["1-7 days", "8-30 days", "31+ days"]
INTERVAL_QUANTILES = (0.10, 0.90)   # 80% interval, same width as Prophet's default


def _horizon_bucket(horizon_day):
    return pd.cut(horizon_day, bins=INTERVAL_BINS, labels=INTERVAL_LABELS).astype(str)


def residual_quantiles(cv_preds):
    """Relative CV errors (actual / predicted - 1), 10th and 90th percentiles per horizon."""
    df = cv_preds[cv_preds["yhat"] > 0]
    rel_err = df["sales"] / df["yhat"] - 1
    low, high = INTERVAL_QUANTILES
    grouped = rel_err.groupby(_horizon_bucket(df["horizon_day"]))
    return {bucket: [float(s.quantile(low)), float(s.quantile(high))] for bucket, s in grouped}


def add_empirical_intervals(fc, quantiles):
    """fc needs columns yhat and horizon_day."""
    fc = fc.copy()
    buckets = _horizon_bucket(fc["horizon_day"])
    low = buckets.map(lambda b: quantiles[b][0])
    high = buckets.map(lambda b: quantiles[b][1])
    fc["yhat_lower"] = (fc["yhat"] * (1 + low)).clip(lower=0)
    fc["yhat_upper"] = fc["yhat"] * (1 + high)
    return fc