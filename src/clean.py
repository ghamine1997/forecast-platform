"""Stage 2 - Cleaning: raw_sales -> clean_sales (item level) + sales_daily_store (store level)."""
import logging

import pandas as pd

from src.config import get_engine
from src.db import write_table

log = logging.getLogger(__name__)


def clean_item_level(df):
    df = df.dropna(subset=["date", "store", "item"])
    df = df.drop_duplicates(subset=["date", "store", "item"], keep="last")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["store"] = df["store"].astype(int)
    df["item"] = df["item"].astype(int)
    df["sales"] = df["sales"].clip(lower=0)

    # Frequency normalization: every (date, store, item) combination must exist
    full_index = pd.MultiIndex.from_product(
        [
            pd.date_range(df["date"].min(), df["date"].max(), freq="D"),
            sorted(df["store"].unique()),
            sorted(df["item"].unique()),
        ],
        names=["date", "store", "item"],
    )
    df = df.set_index(["date", "store", "item"]).reindex(full_index).reset_index()
    missing = int(df["sales"].isna().sum())

    # Fill gaps by interpolating within each series
    df = df.sort_values(["store", "item", "date"])
    df["sales"] = df.groupby(["store", "item"])["sales"].transform(
        lambda s: s.interpolate(limit_direction="both")
    )
    df["sales"] = df["sales"].fillna(0)
    return df, missing


def build_store_level(df):
    store = df.groupby(["date", "store"], as_index=False)["sales"].sum()
    store = store.sort_values(["store", "date"]).reset_index(drop=True)

    # Outliers: compare each day to its local typical level (29-day rolling median)
    baseline = store.groupby("store")["sales"].transform(
        lambda s: s.rolling(29, center=True, min_periods=1).median()
    )
    resid = store["sales"] - baseline
    q1 = resid.groupby(store["store"]).transform(lambda r: r.quantile(0.25))
    q3 = resid.groupby(store["store"]).transform(lambda r: r.quantile(0.75))
    lower, upper = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)

    store["is_outlier"] = (resid < lower) | (resid > upper)
    store["sales"] = baseline + resid.clip(lower, upper)
    return store


def run():
    engine = get_engine()
    raw = pd.read_sql("SELECT date, store, item, sales FROM raw_sales", engine)
    raw_rows = len(raw)

    items, missing_filled = clean_item_level(raw)
    stores = build_store_level(items)

    write_table(items, "clean_sales", engine)
    write_table(stores, "sales_daily_store", engine)

    summary = {
        "raw_rows": int(raw_rows),
        "clean_rows": int(len(items)),
        "missing_filled": missing_filled,
        "store_rows": int(len(stores)),
        "outliers_capped": int(stores["is_outlier"].sum()),
    }
    log.info("Cleaning summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run())