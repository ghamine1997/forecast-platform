"""Stage 3 - Feature engineering: sales_daily_store -> features_store."""
import logging

import holidays
import numpy as np
import pandas as pd

from src.config import get_engine
from src.db import write_table

log = logging.getLogger(__name__)

HOLIDAY_COUNTRY = "US"
LAGS = [1, 7, 28]
WINDOWS = [7, 28]


def add_calendar_features(df):
    d = df["date"]
    df["day"] = d.dt.day
    df["day_of_week"] = d.dt.dayofweek          # 0 = Monday
    df["week"] = d.dt.isocalendar().week.astype(int)
    df["month"] = d.dt.month
    df["quarter"] = d.dt.quarter
    df["year"] = d.dt.year
    df["day_of_year"] = d.dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    return df


def add_holiday_features(df):
    years = range(df["date"].dt.year.min(), df["date"].dt.year.max() + 1)
    calendar = holidays.country_holidays(HOLIDAY_COUNTRY, years=years)
    hol = pd.to_datetime(list(calendar.keys()))
    near = hol.union(hol - pd.Timedelta(days=1)).union(hol + pd.Timedelta(days=1))

    df["is_holiday"] = df["date"].isin(hol).astype(int)
    df["near_holiday"] = df["date"].isin(near).astype(int)

    # Proxy for promotional periods (this dataset has no promotion column):
    # the retail peak from late November to the end of December
    month, day = df["date"].dt.month, df["date"].dt.day
    df["is_peak_season"] = ((month == 12) | ((month == 11) & (day >= 20))).astype(int)
    return df


def add_fourier_features(df):
    """Smooth sine/cosine waves that describe yearly and weekly cycles."""
    t_year = df["date"].dt.dayofyear / 365.25
    for k in (1, 2, 3):
        df[f"year_sin_{k}"] = np.sin(2 * np.pi * k * t_year)
        df[f"year_cos_{k}"] = np.cos(2 * np.pi * k * t_year)
    t_week = df["date"].dt.dayofweek / 7
    for k in (1, 2):
        df[f"week_sin_{k}"] = np.sin(2 * np.pi * k * t_week)
        df[f"week_cos_{k}"] = np.cos(2 * np.pi * k * t_week)
    return df


def add_lag_features(df):
    """Past values only: every feature is shifted so it never sees the current day."""
    df = df.sort_values(["store", "date"]).reset_index(drop=True)
    by_store = df.groupby("store")["sales"]

    for lag in LAGS:
        df[f"lag_{lag}"] = by_store.shift(lag)

    for w in WINDOWS:
        df[f"roll_mean_{w}"] = by_store.transform(lambda s: s.shift(1).rolling(w).mean())
        df[f"roll_median_{w}"] = by_store.transform(lambda s: s.shift(1).rolling(w).median())
        df[f"roll_std_{w}"] = by_store.transform(lambda s: s.shift(1).rolling(w).std())
    return df


def make_features(df):
    """Build all features from a frame with columns: date, store, sales.
    Reused on Day 9 for future dates, so training and forecasting match exactly."""
    df = df[["date", "store", "sales"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df = add_calendar_features(df)
    df = add_holiday_features(df)
    df = add_fourier_features(df)
    df = add_lag_features(df)
    return df


def run():
    engine = get_engine()
    sales = pd.read_sql("SELECT date, store, sales FROM sales_daily_store", engine)

    features = make_features(sales)
    write_table(features, "features_store", engine)

    feature_cols = [c for c in features.columns if c not in ("date", "store", "sales")]
    summary = {
        "rows": int(len(features)),
        "features": len(feature_cols),
        "complete_rows": int(features.dropna().shape[0]),
    }
    log.info("Feature summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run())