"""Final stage - Update dashboard: refresh the views and KPIs the dashboard reads."""
import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import text

from src.config import get_engine

log = logging.getLogger(__name__)

VIEWS = {
    "latest_forecast":
        "SELECT * FROM forecasts WHERE forecast_id = (SELECT MAX(forecast_id) FROM forecasts)",
    "production_model":
        "SELECT * FROM model_registry WHERE stage = 'production'",
}

KPI_SQL = """
SELECT f.forecast_id, f.model_name, f.model_version,
       MIN(f.date) AS forecast_start, MAX(f.date) AS forecast_end,
       SUM(f.yhat) FILTER (WHERE f.horizon_day <= 7)   AS forecast_7d,
       SUM(f.yhat) FILTER (WHERE f.horizon_day <= 30)  AS forecast_30d,
       SUM(f.yhat) FILTER (WHERE f.horizon_day <= 90)  AS forecast_90d,
       SUM(f.yhat) FILTER (WHERE f.horizon_day <= 365) AS forecast_365d,
       (SELECT SUM(sales) FROM sales_daily_store
        WHERE date > (SELECT MAX(date) FROM sales_daily_store) - INTERVAL '365 days') AS sales_last_365d
FROM latest_forecast f
GROUP BY f.forecast_id, f.model_name, f.model_version
"""


def run():
    engine = get_engine()
    with engine.begin() as conn:
        for name, sql in VIEWS.items():
            conn.execute(text(f"DROP VIEW IF EXISTS {name}"))
            conn.execute(text(f"CREATE VIEW {name} AS {sql}"))

    kpis = pd.read_sql(KPI_SQL, engine)
    kpis["growth_vs_last_year_pct"] = (kpis["forecast_365d"] / kpis["sales_last_365d"] - 1) * 100
    kpis["published_at"] = datetime.now(timezone.utc)
    kpis.to_sql("dashboard_kpis", engine, if_exists="replace", index=False)

    row = kpis.iloc[0]
    summary = {
        "forecast_id": row["forecast_id"],
        "model": f"{row['model_name']} v{int(row['model_version'])}",
        "forecast_365d": round(float(row["forecast_365d"])),
        "sales_last_365d": round(float(row["sales_last_365d"])),
        "growth_vs_last_year_pct": round(float(row["growth_vs_last_year_pct"]), 2),
    }
    log.info("Dashboard data published: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run())