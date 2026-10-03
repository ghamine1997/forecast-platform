"""Stage 9 - Forecast generation with the production model from the registry."""
import json
import logging
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from src.config import get_engine
from src.db import write_table
from src.models import add_empirical_intervals, forecast_lightgbm, forecast_prophet, load_data
from src.storage import load_pickle, load_prophet_models

log = logging.getLogger(__name__)

FORECAST_DAYS = int(os.getenv("FORECAST_DAYS", "365"))  # covers 7, 30, 90 days and 12 months
TREND_WINDOW = 28     # days in the moving average used as trend
SHARE_DAYS = 90       # recent period used to split store forecasts into items
HORIZONS = (7, 30, 90, 365)


def load_production_model(engine):
    reg = pd.read_sql("""SELECT * FROM model_registry
                         WHERE stage = 'production' ORDER BY version DESC LIMIT 1""", engine)
    if reg.empty:
        raise ValueError("No production model in the registry")
    return reg.iloc[0]


def predict(entry, sales):
    origin = sales["date"].max()

    if entry["model_name"] == "prophet":
        frames = []
        for store, model in load_prophet_models(entry["artifact_key"]).items():
            fc = forecast_prophet(model, FORECAST_DAYS)
            fc["store"] = store
            frames.append(fc)
        fc = pd.concat(frames, ignore_index=True)
        fc["horizon_day"] = (fc["date"] - origin).dt.days
    else:
        model = load_pickle(entry["artifact_key"])
        fc = forecast_lightgbm(model, sales, FORECAST_DAYS)
        fc["horizon_day"] = (fc["date"] - origin).dt.days
        fc = add_empirical_intervals(fc, json.loads(entry["interval_quantiles"]))

    return fc[["date", "store", "horizon_day", "yhat", "yhat_lower", "yhat_upper"]]


def add_trend_seasonality(fc, sales):
    """Model-agnostic decomposition: trend = 28-day centered moving average, seasonal = the rest.
    The last 28 days of real sales are prepended so the trend has no edge effect on day 1."""
    recent = sales[sales["date"] > sales["date"].max() - pd.Timedelta(days=TREND_WINDOW)]
    recent = recent[["date", "store", "sales"]].rename(columns={"sales": "yhat"})
    combined = pd.concat([recent, fc[["date", "store", "yhat"]]]).sort_values(["store", "date"])
    combined["trend"] = combined.groupby("store")["yhat"].transform(
        lambda s: s.rolling(TREND_WINDOW, center=True, min_periods=1).mean()
    )
    fc = fc.merge(combined[["date", "store", "trend"]], on=["date", "store"], how="left")
    fc["seasonal"] = fc["yhat"] - fc["trend"]
    return fc


def disaggregate_to_items(fc, engine, origin):
    """Top-down: split each store forecast across items by their recent share of store sales."""
    cutoff = (origin - pd.Timedelta(days=SHARE_DAYS)).to_pydatetime()
    items = pd.read_sql(text("""SELECT store, item, SUM(sales) AS sales FROM clean_sales
                                WHERE date > :cutoff GROUP BY store, item"""),
                        engine, params={"cutoff": cutoff})
    items["share"] = items["sales"] / items.groupby("store")["sales"].transform("sum")

    cols = ["forecast_id", "date", "store", "horizon_day", "yhat", "yhat_lower", "yhat_upper"]
    out = fc[cols].merge(items[["store", "item", "share"]], on="store")
    for c in ("yhat", "yhat_lower", "yhat_upper"):
        out[c] = (out[c] * out["share"]).round(2)
    return out.drop(columns="share")


def run():
    engine = get_engine()
    sales, _ = load_data(engine)
    entry = load_production_model(engine)
    origin = sales["date"].max()
    forecast_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    log.info("Forecasting %d days from %s with %s v%d",
             FORECAST_DAYS, origin.date(), entry["model_name"], entry["version"])

    fc = add_trend_seasonality(predict(entry, sales), sales)
    fc.insert(0, "forecast_id", forecast_id)
    fc["model_name"] = entry["model_name"]
    fc["model_version"] = int(entry["version"])
    fc["generated_at"] = datetime.now(timezone.utc)
    fc.to_sql("forecasts", engine, if_exists="append", index=False,
              method="multi", chunksize=1000)

    items = disaggregate_to_items(fc, engine, origin)
    write_table(items, "forecasts_item", engine)

    summary = {
        "forecast_id": forecast_id,
        "model": f"{entry['model_name']} v{int(entry['version'])}",
        "start": str(fc["date"].min().date()),
        "end": str(fc["date"].max().date()),
        "store_rows": int(len(fc)),
        "item_rows": int(len(items)),
        **{f"total_next_{h}_days": round(float(fc.loc[fc["horizon_day"] <= h, "yhat"].sum()))
           for h in HORIZONS},
    }
    log.info("Forecast summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run())