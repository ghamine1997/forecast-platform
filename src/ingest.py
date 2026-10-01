"""Stage 1 - Ingestion: MinIO raw bucket -> PostgreSQL raw_sales table."""
import io
import logging
from datetime import datetime, timezone

import pandas as pd

from src.config import RAW_BUCKET, RAW_KEY, get_engine, get_s3_client
from src.db import write_table

log = logging.getLogger(__name__)
EXPECTED_COLUMNS = ["date", "store", "item", "sales"]


def read_raw_csv():
    obj = get_s3_client().get_object(Bucket=RAW_BUCKET, Key=RAW_KEY)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def validate(df):
    """Run data quality checks. Critical failures raise; others are logged."""
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols or df.empty:
        raise ValueError(f"Critical: empty file or missing columns {sorted(missing_cols)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    checks = []

    bad_dates = int(df["date"].isna().sum())
    checks.append(("parsable_dates", bad_dates == 0, f"{bad_dates} unparsable dates"))

    nulls = int(df[EXPECTED_COLUMNS].isna().sum().sum())
    checks.append(("no_missing_values", nulls == 0, f"{nulls} missing values"))

    negatives = int((df["sales"] < 0).sum())
    checks.append(("non_negative_sales", negatives == 0, f"{negatives} negative sales"))

    dups = int(df.duplicated(subset=["date", "store", "item"]).sum())
    checks.append(("unique_date_store_item", dups == 0, f"{dups} duplicate rows"))

    expected_days = (df["date"].max() - df["date"].min()).days + 1
    rows_per_series = df.groupby(["store", "item"]).size()
    gaps = int((rows_per_series < expected_days).sum())
    checks.append(("complete_daily_series", gaps == 0, f"{gaps} series with missing days"))

    for name, passed, detail in checks:
        log.info("%-24s %s  (%s)", name, "PASS" if passed else "WARN", detail)
    return df, checks


def run():
    df = read_raw_csv()
    df, checks = validate(df)

    engine = get_engine()
    write_table(df, "raw_sales", engine)

    quality = pd.DataFrame(checks, columns=["check_name", "passed", "detail"])
    quality.insert(0, "run_at", datetime.now(timezone.utc))
    quality.to_sql("data_quality_log", engine, if_exists="append", index=False)

    summary = {
        "rows": int(len(df)),
        "stores": int(df["store"].nunique()),
        "items": int(df["item"].nunique()),
        "start": str(df["date"].min().date()),
        "end": str(df["date"].max().date()),
        "checks_passed": int(sum(c[1] for c in checks)),
        "checks_total": len(checks),
    }
    log.info("Ingestion summary: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(run())